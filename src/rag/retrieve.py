"""Retrieval layer for vector search."""

import json
import time
from typing import TYPE_CHECKING

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .audit import (
    RetrievalEvent,
    create_actor_from_user_context,
    generate_request_id,
    get_audit_logger,
    hash_query,
)
from .config import settings
from .models import Chunk, HierarchicalChunk, HierarchicalRetrievalResult, RetrievalResult

if TYPE_CHECKING:
    from .models import UserContext

# Module-level cache for loaded resources
_index: faiss.Index | None = None
_chunks: list[Chunk] | None = None
_model: SentenceTransformer | None = None

# Hierarchical chunking cache
_parents: dict[str, HierarchicalChunk] | None = None
_children: list[HierarchicalChunk] | None = None
_is_hierarchical: bool | None = None


def load_index(force_reload: bool = False) -> tuple[faiss.Index, list[Chunk]]:
    """
    Load FAISS index and docstore from disk (flat chunking mode).

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

    # Handle v1 and v2 formats for backward compatibility
    version = docstore.get("version", "1.0")
    if version == "2.0":
        # V2: hierarchical format - extract children as flat chunks for compatibility
        _chunks = [Chunk(**c) for c in docstore["chunks"]["children"]]
    else:
        # V1: flat format
        _chunks = [Chunk(**chunk_data) for chunk_data in docstore["chunks"]]
    print(f"Loaded {len(_chunks)} chunks from docstore")

    return _index, _chunks


def load_hierarchical_index(
    force_reload: bool = False,
) -> tuple[faiss.Index, dict[str, HierarchicalChunk], list[HierarchicalChunk], bool]:
    """
    Load FAISS index and hierarchical docstore from disk.

    Args:
        force_reload: If True, reload even if already cached.

    Returns:
        Tuple of (FAISS index, parents dict, children list, is_hierarchical).

    Raises:
        FileNotFoundError: If index files don't exist.
    """
    global _index, _parents, _children, _is_hierarchical

    if (
        _index is not None
        and _parents is not None
        and _children is not None
        and _is_hierarchical is not None
        and not force_reload
    ):
        return _index, _parents, _children, _is_hierarchical

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

    # Handle version migration
    version = docstore.get("version", "1.0")

    if version == "2.0":
        # V2: hierarchical format
        _is_hierarchical = True
        _parents = {
            p["chunk_id"]: HierarchicalChunk(**p) for p in docstore["chunks"]["parents"]
        }
        _children = [HierarchicalChunk(**c) for c in docstore["chunks"]["children"]]
        print(
            f"Loaded hierarchical docstore: {len(_parents)} parents, {len(_children)} children"
        )
    else:
        # V1: flat format - no parents available
        _is_hierarchical = False
        _parents = {}
        _children = []
        print("Loaded flat docstore (v1) - hierarchical retrieval not available")

    return _index, _parents, _children, _is_hierarchical


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
    user_context: "UserContext | None" = None,
    request_id: str | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve top-k most relevant chunks for a query with RBAC filtering.

    Args:
        query: Query string.
        k: Number of results to return. Defaults to settings.default_top_k.
        min_score: Minimum relevance score threshold (0-1 for cosine similarity).
        user_context: User context for RBAC filtering. None = public-only access.
        request_id: Optional correlation ID for audit logging.

    Returns:
        List of RetrievalResult objects sorted by relevance (filtered by RBAC).
    """
    from .rbac import filter_retrieval_results

    start_time = time.perf_counter()

    # Generate request_id if not provided
    if request_id is None:
        request_id = generate_request_id()

    # Get audit logger
    audit_logger = get_audit_logger()

    # Clamp k to allowed range
    if k is None:
        k = settings.default_top_k
    k = min(k, settings.max_top_k)

    # Over-fetch strategy: fetch k*3 to account for filtering
    fetch_k = min(k * 3, settings.max_top_k * 3)

    # Load index and chunks
    index, chunks = load_index()

    # Get query embedding
    query_embedding = embed_query(query)

    # Search (fetch more candidates)
    scores, indices = index.search(query_embedding, fetch_k)

    # Build unfiltered results
    unfiltered_results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        # Skip invalid indices (can happen if k > index size)
        if idx < 0 or idx >= len(chunks):
            continue

        # Skip low scores
        if score < min_score:
            continue

        unfiltered_results.append(
            RetrievalResult(
                chunk=chunks[idx],
                score=float(score),
                rank=rank,
            )
        )

    results_before_filter = len(unfiltered_results)

    # Apply RBAC filtering with audit logging
    filtered_results = filter_retrieval_results(
        unfiltered_results,
        user_context,
        request_id=request_id,
        audit_logger=audit_logger,
    )

    filter_applied = ["rbac"]

    # If filtered results < k and we haven't hit max fetch limit, expand search
    if len(filtered_results) < k and fetch_k < settings.max_top_k * 5:
        expand_fetch_k = min(k * 5, settings.max_top_k * 5)
        print(f"RBAC: Expanding search from {fetch_k} to {expand_fetch_k} due to filtering")

        # Re-search with expanded k
        scores, indices = index.search(query_embedding, expand_fetch_k)

        # Rebuild unfiltered results
        unfiltered_results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            if score < min_score:
                continue

            unfiltered_results.append(
                RetrievalResult(
                    chunk=chunks[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        results_before_filter = len(unfiltered_results)

        # Re-apply filtering with audit logging
        filtered_results = filter_retrieval_results(
            unfiltered_results,
            user_context,
            request_id=request_id,
            audit_logger=audit_logger,
        )

    # Get final results
    final_results = filtered_results[:k]

    # Log retrieval event
    latency_ms = (time.perf_counter() - start_time) * 1000
    classifications_accessed = list(
        set(r.chunk.metadata.classification.value for r in final_results)
    )
    pii_accessed = any(r.chunk.metadata.pii_flag for r in final_results)

    actor = create_actor_from_user_context(user_context, auth_method="cli")
    event = RetrievalEvent(
        request_id=request_id,
        actor=actor,
        query_hash=hash_query(query),
        embedding_model=settings.embedding_model,
        k_requested=k,
        results_before_filter=results_before_filter,
        results_after_filter=len(filtered_results),
        top_k_returned=len(final_results),
        filter_applied=filter_applied,
        classifications_accessed=classifications_accessed,
        pii_accessed=pii_accessed,
        policy_decision="allowed" if final_results else "no_results",
        resource_type="chunk",
        resource_ids=[r.chunk.chunk_id for r in final_results],
        latency_ms=latency_ms,
    )
    audit_logger.log(event)

    return final_results


def retrieve_with_debug(
    query: str,
    k: int | None = None,
    user_context: "UserContext | None" = None,
) -> dict:
    """
    Retrieve with debug information for development.

    Args:
        query: Query string.
        k: Number of results.
        user_context: User context for RBAC filtering.

    Returns:
        Dict with results and debug info.
    """
    results = retrieve(query, k, user_context=user_context)

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
                "text_preview": r.chunk.text[:200] + "..."
                if len(r.chunk.text) > 200
                else r.chunk.text,
            }
            for r in results
        ],
    }


def retrieve_hierarchical(
    query: str,
    k: int | None = None,
    return_parents: int = 3,
    min_score: float = 0.0,
    user_context: "UserContext | None" = None,
    request_id: str | None = None,
) -> list[HierarchicalRetrievalResult]:
    """
    Retrieve with hierarchical parent context resolution and RBAC filtering.

    Searches child chunks and returns parent chunks with their matched children.
    Filters at parent level based on user context.

    Args:
        query: Query string.
        k: Number of child chunks to search. Defaults to settings.default_top_k * 3.
        return_parents: Maximum number of parent chunks to return.
        min_score: Minimum relevance score threshold.
        user_context: User context for RBAC filtering. None = public-only access.
        request_id: Optional correlation ID for audit logging.

    Returns:
        List of HierarchicalRetrievalResult objects (filtered by RBAC).
    """
    from .rbac import filter_hierarchical_results

    start_time = time.perf_counter()

    # Generate request_id if not provided
    if request_id is None:
        request_id = generate_request_id()

    # Get audit logger
    audit_logger = get_audit_logger()

    # Clamp k to allowed range (over-fetch for better parent coverage)
    if k is None:
        k = settings.default_top_k * 3
    k = min(k, settings.max_top_k * 3)

    # Load hierarchical index
    index, parents, children, is_hierarchical = load_hierarchical_index()

    if not is_hierarchical:
        print("Warning: Docstore is not hierarchical. Use retrieve() instead.")
        return []

    # Get query embedding
    query_embedding = embed_query(query)

    # Search children
    scores, indices = index.search(query_embedding, k)

    # Group by parent_id
    parent_groups: dict[str, list[tuple[HierarchicalChunk, float]]] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(children):
            continue
        if score < min_score:
            continue

        child = children[idx]
        parent_id = child.parent_id
        if parent_id is None:
            continue

        if parent_id not in parent_groups:
            parent_groups[parent_id] = []
        parent_groups[parent_id].append((child, float(score)))

    # Build unfiltered results for each parent
    unfiltered_results = []
    for parent_id, child_matches in parent_groups.items():
        parent_chunk = parents.get(parent_id)
        if parent_chunk is None:
            continue

        # Sort children by score
        child_matches.sort(key=lambda x: x[1], reverse=True)
        matched_children = [c for c, _ in child_matches]
        child_scores = [s for _, s in child_matches]

        # Aggregate score: use max child score
        aggregate_score = max(child_scores) if child_scores else 0.0

        unfiltered_results.append(
            HierarchicalRetrievalResult(
                parent_chunk=parent_chunk,
                matched_children=matched_children,
                child_scores=child_scores,
                aggregate_score=aggregate_score,
                rank=0,  # Will be set after filtering
            )
        )

    # Sort by aggregate score
    unfiltered_results.sort(key=lambda x: x.aggregate_score, reverse=True)

    results_before_filter = len(unfiltered_results)

    # Apply RBAC filtering at parent level with audit logging
    filtered_results = filter_hierarchical_results(
        unfiltered_results,
        user_context,
        request_id=request_id,
        audit_logger=audit_logger,
    )

    # Get final results
    final_results = filtered_results[:return_parents]

    # Log retrieval event
    latency_ms = (time.perf_counter() - start_time) * 1000
    classifications_accessed = list(
        set(r.parent_chunk.metadata.classification.value for r in final_results)
    )
    pii_accessed = any(r.parent_chunk.metadata.pii_flag for r in final_results)

    filter_applied = ["rbac", "hierarchical"]

    actor = create_actor_from_user_context(user_context, auth_method="cli")
    event = RetrievalEvent(
        request_id=request_id,
        actor=actor,
        query_hash=hash_query(query),
        index_name="hierarchical",
        embedding_model=settings.embedding_model,
        k_requested=return_parents,
        results_before_filter=results_before_filter,
        results_after_filter=len(filtered_results),
        top_k_returned=len(final_results),
        filter_applied=filter_applied,
        classifications_accessed=classifications_accessed,
        pii_accessed=pii_accessed,
        policy_decision="allowed" if final_results else "no_results",
        resource_type="parent_chunk",
        resource_ids=[r.parent_chunk.chunk_id for r in final_results],
        latency_ms=latency_ms,
    )
    audit_logger.log(event)

    return final_results


def _get_parent_preview(text: str, header: str | None) -> str:
    """
    Get a preview of parent chunk: header + first N chars.

    Args:
        text: Full parent chunk text.
        header: Section header.

    Returns:
        Preview string with header and initial content.
    """
    preview_size = settings.parent_preview_size
    if len(text) <= preview_size:
        return text

    # Return header context + truncated preview
    preview = text[:preview_size]
    # Try to end at a sentence or line break
    last_break = max(preview.rfind("\n"), preview.rfind(". "), preview.rfind("。"))
    if last_break > preview_size // 2:
        preview = preview[: last_break + 1]
    return preview + "..."


def retrieve_hierarchical_with_debug(
    query: str,
    k: int | None = None,
    return_parents: int = 3,
    include_full_parent: bool = False,
    user_context: "UserContext | None" = None,
) -> dict:
    """
    Retrieve hierarchically with debug information.

    Provides two-stage context:
    - preview: Section header + initial N chars
    - full_text: Complete parent content (optional, for when preview is insufficient)

    Args:
        query: Query string.
        k: Number of child chunks to search.
        return_parents: Maximum number of parent chunks to return.
        include_full_parent: If True, include full parent text in results.
        user_context: User context for RBAC filtering.

    Returns:
        Dict with hierarchical results and debug info.
    """
    results = retrieve_hierarchical(query, k, return_parents, user_context=user_context)

    return {
        "query": query,
        "k": k or settings.default_top_k * 3,
        "return_parents": return_parents,
        "num_results": len(results),
        "results": [
            {
                "rank": r.rank,
                "aggregate_score": round(r.aggregate_score, 4),
                "parent": {
                    "chunk_id": r.parent_chunk.chunk_id,
                    "doc_id": r.parent_chunk.doc_id,
                    "section_header": r.parent_chunk.section_header,
                    "classification": r.parent_chunk.metadata.classification.value,
                    # Two-stage context: preview first
                    "preview": _get_parent_preview(
                        r.parent_chunk.text, r.parent_chunk.section_header
                    ),
                    # Full text available on demand
                    "full_text": r.parent_chunk.text if include_full_parent else None,
                    "full_text_length": len(r.parent_chunk.text),
                },
                "matched_children": [
                    {
                        "chunk_id": c.chunk_id,
                        "section_header": c.section_header,
                        "score": round(s, 4),
                        "text": c.text,
                    }
                    for c, s in zip(r.matched_children[:3], r.child_scores[:3])
                ],
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
