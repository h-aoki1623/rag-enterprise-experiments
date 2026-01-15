"""Retrieval layer for vector search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import Chunk, RetrievalResult

# Module-level cache for loaded resources
_index: faiss.Index | None = None
_chunks: list[Chunk] | None = None
_model: SentenceTransformer | None = None


def load_index(force_reload: bool = False) -> tuple[faiss.Index, list[Chunk]]:
    """
    Load FAISS index and docstore from disk.

    Args:
        force_reload: If True, reload even if already cached.

    Returns:
        Tuple of (FAISS index, list of Chunk objects).

    Raises:
        FileNotFoundError: If index files don't exist.
    """
    global _index, _chunks

    if _index is not None and _chunks is not None and not force_reload:
        return _index, _chunks

    # Check if files exist
    if not settings.faiss_index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {settings.faiss_index_path}. Run ingestion first."
        )
    if not settings.docstore_path.exists():
        raise FileNotFoundError(
            f"Docstore not found at {settings.docstore_path}. Run ingestion first."
        )

    # Load FAISS index
    _index = faiss.read_index(str(settings.faiss_index_path))
    print(f"Loaded FAISS index with {_index.ntotal} vectors")

    # Load docstore
    with open(settings.docstore_path, encoding="utf-8") as f:
        docstore = json.load(f)

    _chunks = [Chunk(**chunk_data) for chunk_data in docstore["chunks"]]
    print(f"Loaded {len(_chunks)} chunks from docstore")

    return _index, _chunks


def get_model() -> SentenceTransformer:
    """Get or load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_query(query: str, model: SentenceTransformer | None = None) -> np.ndarray:
    """
    Generate embedding for a query.

    Args:
        query: Query string to embed.
        model: Optional pre-loaded model.

    Returns:
        Numpy array of shape (1, embedding_dim).
    """
    if model is None:
        model = get_model()

    embedding = model.encode([query], convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding)

    return embedding


def retrieve(
    query: str,
    k: int | None = None,
    min_score: float = 0.0,
) -> list[RetrievalResult]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        query: Query string.
        k: Number of results to return. Defaults to settings.default_top_k.
        min_score: Minimum relevance score threshold (0-1 for cosine similarity).

    Returns:
        List of RetrievalResult objects sorted by relevance.
    """
    # Clamp k to allowed range
    if k is None:
        k = settings.default_top_k
    k = min(k, settings.max_top_k)

    # Load index and chunks
    index, chunks = load_index()

    # Get query embedding
    query_embedding = embed_query(query)

    # Search
    scores, indices = index.search(query_embedding, k)

    # Build results
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        # Skip invalid indices (can happen if k > index size)
        if idx < 0 or idx >= len(chunks):
            continue

        # Skip low scores
        if score < min_score:
            continue

        results.append(
            RetrievalResult(
                chunk=chunks[idx],
                score=float(score),
                rank=rank,
            )
        )

    return results


def retrieve_with_debug(
    query: str,
    k: int | None = None,
) -> dict:
    """
    Retrieve with debug information for development.

    Args:
        query: Query string.
        k: Number of results.

    Returns:
        Dict with results and debug info.
    """
    results = retrieve(query, k)

    return {
        "query": query,
        "k": k or settings.default_top_k,
        "num_results": len(results),
        "results": [
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "classification": r.chunk.metadata.classification.value,
                "text_preview": r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text,
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    # Quick test
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is the company policy?"
    debug_results = retrieve_with_debug(query)

    print(f"\nQuery: {debug_results['query']}")
    print(f"Results: {debug_results['num_results']}\n")

    for r in debug_results["results"]:
        print(f"[{r['rank']}] Score: {r['score']:.4f} | {r['chunk_id']} ({r['classification']})")
        print(f"    {r['text_preview']}\n")
