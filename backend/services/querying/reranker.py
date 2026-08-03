import math
import re


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical scoring."""

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
