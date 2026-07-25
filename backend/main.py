from pathlib import Path
import sys

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.create_engine import create_tables, get_db
from backend.models import MetaData
from backend.services.documents import (
    build_document_response,
    delete_document,
    index_document_background,
    save_document_for_background_indexing,
)
from backend.services.querying.LLM import generate_response, load_client
from backend.services.querying.prompt_builder import build_prompt
from backend.services.querying.retriever import (
    load_collection,
    load_embedding_model,
    retrieve_documents,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = BASE_DIR / "RAG_Documents"

embedding_model = None
collection = None
llm_client = None


class DocumentResponse(BaseModel):
    id: int
    doc_name: str
    doc_type: str | None
    file_path: str
    indexed_chunks: int = 0
    indexing_status: str = "indexing"
    indexing_error: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceChunk(BaseModel):
    document_name: str
    page_number: int
    distance: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.on_event("startup")
def startup() -> None:
    create_tables()
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return "Dense RAG"


@app.get("/health")
def status():
    return {"Status":"Healthy"}


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(MetaData).order_by(MetaData.uploaded_at.desc()).all()
    return [
        build_document_response(document)
        for document in documents
    ]


@app.post("/documents", response_model=DocumentResponse)
async def post_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    doc_info = await save_document_for_background_indexing(
        file=file,
        db=db,
        document_dir=DOCUMENT_DIR,
    )
    background_tasks.add_task(index_uploaded_document, doc_info.id)
    return build_document_response(doc_info)


@app.delete("/documents/{doc_id}", response_model=DocumentResponse)
def remove_document(doc_id: int, db: Session = Depends(get_db)):
    return delete_document(db, doc_id)


def index_uploaded_document(doc_id: int) -> None:
    global embedding_model, collection

    if embedding_model is None:
        embedding_model = load_embedding_model()

    indexed_collection = index_document_background(doc_id, embedding_model)
    if indexed_collection is not None:
        collection = indexed_collection


def get_rag_services():
    global embedding_model, collection, llm_client

    if embedding_model is None:
        embedding_model = load_embedding_model()
    if collection is None:
        collection = load_collection()
    if llm_client is None:
        llm_client = load_client()

    return embedding_model, collection, llm_client


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        model, vector_collection, client = get_rag_services()
        active_document_names = {
            name
            for (name,) in db.query(MetaData.doc_name)
            .filter(MetaData.indexing_status == "indexed")
            .all()
        }
        retrieved_chunks = retrieve_documents(
            question=request.question,
            model=model,
            collection=vector_collection,
            top_k=request.top_k,
            allowed_document_names=active_document_names,
        )
        prompt = build_prompt(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
        )
        answer = generate_response(prompt=prompt, client=client)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer, sources=retrieved_chunks)
