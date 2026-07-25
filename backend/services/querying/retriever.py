from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


BASE_DIR = Path(__file__).resolve().parents[3]


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""

    return SentenceTransformer("all-MiniLM-L6-v2")


def load_collection():
    """Load or create the ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

    collection = client.get_or_create_collection(
        name="research_documents"
    )

    return collection


def retrieve_documents(
    question: str,
    model: SentenceTransformer,
    collection,
    top_k: int = 5,
    allowed_document_names: set[str] | None = None,
) -> list:
    """Retrieve the most relevant chunks."""

    if allowed_document_names is not None and not allowed_document_names:
        return []

    query_embedding = model.encode(question).tolist()
    search_limit = top_k if allowed_document_names is None else max(top_k * 5, 25)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=search_limit,
    )

    retrieved_chunks=[]
    documents= results["documents"][0]
    metadatas= results["metadatas"][0]
    distances= results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        document_name = metadata["document_name"]
        if allowed_document_names is not None and document_name not in allowed_document_names:
            continue

        retrieved_chunks.append(
            {
                "text": document,
                "document_name": document_name,
                "page_number": metadata["page_number"],
                "distance": distance,
            }
        )
        if len(retrieved_chunks) >= top_k:
            break

    return retrieved_chunks

if __name__ == "__main__":
    model = load_embedding_model()

    collection = load_collection()

    question = input("Enter your question: ")

    results = retrieve_documents(
        question,
        model,
        collection,
    )

    print(results)
