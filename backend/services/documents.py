from pathlib import Path
import shutil
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.create_engine import engine
from backend.models import MetaData
from backend.services.indexer.chunker import chunk_documents
from backend.services.indexer.embedder import (
    create_collection,
    generate_embeddings,
    store_embeddings,
)
from backend.services.indexer.pdf_parser import parse_pdf


def unique_file_path(directory: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    candidate = directory / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


async def save_uploaded_file(file: UploadFile, document_dir: Path) -> Path:
    document_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_file_path(document_dir, file.filename or "")

    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not save uploaded document.") from exc
    finally:
        await file.close()

    return destination


def create_document_metadata(
    db: Session,
    destination: Path,
    content_type: str | None,
) -> MetaData:
    doc_info = MetaData(
        doc_name=destination.name,
        doc_type=content_type,
        file_path=str(destination),
        indexing_status="indexing",
        indexed_chunks=0,
    )

    try:
        db.add(doc_info)
        db.commit()
        db.refresh(doc_info)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not save document metadata.") from exc

    return doc_info


def cleanup_failed_document(db: Session, doc_info: MetaData, destination: Path) -> None:
    db.delete(doc_info)
    db.commit()
    destination.unlink(missing_ok=True)


def mark_document_failed(
    db: Session,
    doc_info: MetaData,
    error: Exception,
) -> None:
    doc_info.indexing_status = "failed"
    doc_info.indexing_error = str(error)
    doc_info.indexed_chunks = 0
    db.commit()


def mark_document_indexed(
    db: Session,
    doc_info: MetaData,
    indexed_chunks: int,
) -> None:
    doc_info.indexing_status = "indexed"
    doc_info.indexing_error = None
    doc_info.indexed_chunks = indexed_chunks
    db.commit()


def index_document(
    destination: Path,
    doc_id: int,
    embedding_model: Any,
) -> tuple[int, Any]:
    parsed = parse_pdf(destination)
    chunks = chunk_documents([parsed])
    if not chunks:
        raise ValueError("No extractable text was found. This PDF may be scanned or image-only.")

    for chunk in chunks:
        chunk["chunk_id"] = f"{doc_id}-{chunk['chunk_id']}"
        chunk["doc_id"] = str(doc_id)

    chroma_collection = create_collection()
    embeddings = generate_embeddings(chunks, embedding_model)
    store_embeddings(chunks, embeddings, chroma_collection)

    return len(chunks), chroma_collection


async def upload_and_index_document(
    file: UploadFile,
    db: Session,
    document_dir: Path,
    embedding_model: Any,
) -> tuple[MetaData, int, Any]:
    destination = await save_uploaded_file(file, document_dir)
    doc_info = create_document_metadata(db, destination, file.content_type)

    try:
        indexed_chunks, chroma_collection = index_document(
            destination=destination,
            doc_id=doc_info.id,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        cleanup_failed_document(db, doc_info, destination)
        raise HTTPException(status_code=500, detail=f"Could not index document: {exc}") from exc

    return doc_info, indexed_chunks, chroma_collection


async def save_document_for_background_indexing(
    file: UploadFile,
    db: Session,
    document_dir: Path,
) -> MetaData:
    destination = await save_uploaded_file(file, document_dir)
    return create_document_metadata(db, destination, file.content_type)


def index_document_background(
    doc_id: int,
    embedding_model: Any,
) -> Any | None:
    with Session(bind=engine) as db:
        doc_info = db.get(MetaData, doc_id)
        if doc_info is None:
            return None

        try:
            indexed_chunks, chroma_collection = index_document(
                destination=Path(doc_info.file_path),
                doc_id=doc_info.id,
                embedding_model=embedding_model,
            )
            mark_document_indexed(db, doc_info, indexed_chunks)
            return chroma_collection
        except Exception as exc:
            mark_document_failed(db, doc_info, exc)
            return None


def build_document_response(document: MetaData, indexed_chunks: int = 0) -> dict:
    return {
        "id": document.id,
        "doc_name": document.doc_name,
        "doc_type": document.doc_type,
        "file_path": document.file_path,
        "indexed_chunks": document.indexed_chunks or indexed_chunks,
        "indexing_status": document.indexing_status or "indexed",
        "indexing_error": document.indexing_error,
    }


def delete_document(db: Session, doc_id: int) -> dict:
    document = db.get(MetaData, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    response = build_document_response(document)
    file_path = Path(document.file_path)
    try:
        collection = create_collection()
        collection.delete(where={"doc_id": str(doc_id)})
    except Exception:
        # Older indexed documents may not have doc_id metadata; file and DB cleanup still proceed.
        pass

    try:
        db.delete(document)
        db.commit()
        file_path.unlink(missing_ok=True)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not remove document.") from exc

    return response
