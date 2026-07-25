from pathlib import Path
import math
import re

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[3]


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""

    return SentenceTransformer("all-MiniLM-L6-v2")


def load_collection():
    """Load or create the ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

    return client.get_or_create_collection(
        name="research_documents"
    )


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical retrieval."""

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to the 0..1 range."""

    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if math.isclose(min_score, max_score):
        return {key: 1.0 for key in scores}

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def load_candidate_chunks(
    collection,
    allowed_document_names: set[str] | None = None,
) -> list[dict]:
    """Load chunks from Chroma so BM25 can search the same corpus."""

    results = collection.get(include=["documents", "metadatas"])
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    chunks = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        document_name = metadata.get("document_name", "")
        if allowed_document_names is not None and document_name not in allowed_document_names:
            continue

        chunks.append(
            {
                "id": str(chunk_id),
                "text": document,
                "document_name": document_name,
                "page_number": metadata.get("page_number"),
                "doc_id": metadata.get("doc_id", ""),
            }
        )

    return chunks


def bm25_search(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rank candidate chunks with BM25."""

    query_terms = tokenize(question)
    if not query_terms or not chunks:
        return []

    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    doc_count = len(tokenized_docs)
    avg_doc_length = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
    document_frequency = {}

    for tokens in tokenized_docs:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    k1 = 1.5
    b = 0.75
    scored_chunks = []

    for chunk, tokens in zip(chunks, tokenized_docs):
        if not tokens:
            continue

        term_frequency = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        score = 0.0
        doc_length = len(tokens)

        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue

            frequency_in_docs = document_frequency.get(term, 0)
            idf = math.log(1 + ((doc_count - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)))
            denominator = frequency + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * ((frequency * (k1 + 1)) / denominator)

        if score > 0:
            scored_chunks.append({**chunk, "bm25_score": score})

    return sorted(scored_chunks, key=lambda chunk: chunk["bm25_score"], reverse=True)[:top_k]


def vector_search(
    question: str,
    model: SentenceTransformer,
    collection,
    top_k: int,
    allowed_document_names: set[str] | None = None,
) -> list[dict]:
    """Retrieve semantic candidates from Chroma."""

    collection_count = collection.count()
    if collection_count == 0:
        return []

    query_embedding = model.encode(question).tolist()
    search_limit = min(collection_count, top_k if allowed_document_names is None else max(top_k * 5, 25))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=search_limit,
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results.get("ids", [[]])[0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        document_name = metadata.get("document_name", "")
        if allowed_document_names is not None and document_name not in allowed_document_names:
            continue

        chunks.append(
            {
                "id": str(chunk_id),
                "text": document,
                "document_name": document_name,
                "page_number": metadata.get("page_number"),
                "distance": distance,
                "vector_score": 1 / (1 + max(distance, 0)),
            }
        )

        if allowed_document_names is not None and len(chunks) >= top_k:
            break

    return chunks


def rerank_candidates(
    question: str,
    vector_candidates: list[dict],
    bm25_candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """Merge vector and BM25 candidates, then rerank with a hybrid score."""

    merged = {}
    for candidate in vector_candidates:
        merged[candidate["id"]] = {**candidate}

    for candidate in bm25_candidates:
        current = merged.get(candidate["id"], {})
        merged[candidate["id"]] = {
            **candidate,
            **current,
            "bm25_score": candidate["bm25_score"],
        }

    vector_scores = normalize_scores({
        chunk_id: candidate.get("vector_score", 0.0)
        for chunk_id, candidate in merged.items()
    })
    bm25_scores = normalize_scores({
        chunk_id: candidate.get("bm25_score", 0.0)
        for chunk_id, candidate in merged.items()
    })

    query_terms = set(tokenize(question))
    reranked = []
    for chunk_id, candidate in merged.items():
        chunk_terms = set(tokenize(candidate["text"]))
        overlap_score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        rerank_score = (
            0.55 * vector_scores.get(chunk_id, 0.0)
            + 0.35 * bm25_scores.get(chunk_id, 0.0)
            + 0.10 * overlap_score
        )

        reranked.append(
            {
                "text": candidate["text"],
                "document_name": candidate["document_name"],
                "page_number": candidate["page_number"],
                "distance": candidate.get("distance", 1.0),
                "vector_score": candidate.get("vector_score", 0.0),
                "bm25_score": candidate.get("bm25_score", 0.0),
                "rerank_score": rerank_score,
            }
        )

    return sorted(reranked, key=lambda chunk: chunk["rerank_score"], reverse=True)[:top_k]


def retrieve_documents(
    question: str,
    model: SentenceTransformer,
    collection,
    top_k: int = 5,
    allowed_document_names: set[str] | None = None,
) -> list:
    """Retrieve chunks using hybrid vector + BM25 retrieval and reranking."""

    if allowed_document_names is not None and not allowed_document_names:
        return []

    candidate_limit = max(top_k * 5, 25)
    vector_candidates = vector_search(
        question=question,
        model=model,
        collection=collection,
        top_k=candidate_limit,
        allowed_document_names=allowed_document_names,
    )
    lexical_chunks = load_candidate_chunks(
        collection=collection,
        allowed_document_names=allowed_document_names,
    )
    bm25_candidates = bm25_search(
        question=question,
        chunks=lexical_chunks,
        top_k=candidate_limit,
    )

    return rerank_candidates(
        question=question,
        vector_candidates=vector_candidates,
        bm25_candidates=bm25_candidates,
        top_k=top_k,
    )


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
