from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[3]


def load_chunks(path: Path) -> list:
    """Load chunks from JSON."""

    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_collection():
    """Create or load a ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

    collection = client.get_or_create_collection(
        name="research_documents"
    )

    return collection


def generate_embeddings(chunks: list, model: SentenceTransformer) -> list:
    """Generate embeddings for every chunk."""

    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        return []

    embeddings = model.encode(texts)

    return embeddings.tolist()


def store_embeddings(
    chunks: list,
    embeddings: list,
    collection,
) -> None:
    """Store chunks and embeddings in ChromaDB."""

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))

        documents.append(chunk["text"])

        metadatas.append(
            {
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "doc_id": chunk.get("doc_id", ""),
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    INPUT_PATH = BASE_DIR / "chunks.json"

    chunks = load_chunks(INPUT_PATH)

    model = load_embedding_model()

    embeddings = generate_embeddings(chunks, model)

    collection = create_collection()

    store_embeddings(
        chunks,
        embeddings,
        collection,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
