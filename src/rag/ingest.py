"""Document ingestion pipeline: load, chunk, embed, and index."""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .audit import (
    Actor,
    IngestionEvent,
    generate_request_id,
    get_audit_logger,
)
from .config import settings
from .hierarchy import chunk_documents_hierarchical
from .models import Chunk, Document, DocumentMetadata, HierarchicalChunk


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """
    Load documents from directory with their metadata.

    Each document should have a .md file and a corresponding .meta.json file.

    Args:
        docs_dir: Directory containing documents. Defaults to settings.docs_dir.

    Returns:
        List of Document objects with content and metadata.
    """
    docs_dir = docs_dir or settings.docs_dir
    documents = []

    # Find all markdown files
    for md_file in docs_dir.rglob("*.md"):
        # Skip if it's a README or other non-document file
        if md_file.name.lower() == "readme.md":
            continue

        # Look for corresponding metadata file
        meta_file = md_file.with_suffix(".meta.json")
        if not meta_file.exists():
            print(f"Warning: No metadata file for {md_file}, skipping")
            continue

        # Load content
        content = md_file.read_text(encoding="utf-8")

        # Load metadata
        with open(meta_file, encoding="utf-8") as f:
            meta_dict = json.load(f)

        metadata = DocumentMetadata(**meta_dict)

        documents.append(
            Document(
                content=content,
                metadata=metadata,
                file_path=str(md_file),
            )
        )

    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split documents into chunks with inherited metadata.

    Args:
        documents: List of documents to chunk.

    Returns:
        List of Chunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Split the document content
        texts = splitter.split_text(doc.content)

        for i, text in enumerate(texts):
            chunk_id = f"{doc.metadata.doc_id}-chunk-{i:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=doc.metadata.doc_id,
                    metadata=doc.metadata,
                )
            )

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer | None = None) -> np.ndarray:
    """
    Generate embeddings for chunks.

    Args:
        chunks: List of chunks to embed.
        model: Optional pre-loaded model. If None, loads from settings.

    Returns:
        Numpy array of embeddings with shape (num_chunks, embedding_dim).
    """
    if model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        model = SentenceTransformer(settings.embedding_model)

    texts = [chunk.text for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_index(embeddings: np.ndarray, chunks: list[Chunk]) -> None:
    """
    Build FAISS index and save docstore (flat chunking mode).

    Args:
        embeddings: Numpy array of embeddings.
        chunks: List of chunks corresponding to embeddings.
    """
    # Ensure index directory exists
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    # Build FAISS index (using L2 distance, will convert to similarity later)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(settings.faiss_index_path))
    print(f"Saved FAISS index to {settings.faiss_index_path}")

    # Build and save docstore (v1 format for flat chunking)
    docstore = {
        "version": "1.0",
        "chunks": [chunk.model_dump() for chunk in chunks],
        "metadata": {
            "total_chunks": len(chunks),
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
            "hierarchy_enabled": False,
        },
    }

    with open(settings.docstore_path, "w", encoding="utf-8") as f:
        json.dump(docstore, f, indent=2, ensure_ascii=False)
    print(f"Saved docstore to {settings.docstore_path}")


def build_hierarchical_index(
    embeddings: np.ndarray,
    parents: list[HierarchicalChunk],
    children: list[HierarchicalChunk],
) -> None:
    """
    Build FAISS index and save hierarchical docstore.

    Only children are indexed in FAISS for retrieval.
    Parents are stored for context resolution.

    Args:
        embeddings: Numpy array of embeddings for children.
        parents: List of parent chunks.
        children: List of child chunks corresponding to embeddings.
    """
    # Ensure index directory exists
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    # Build FAISS index (children only)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(settings.faiss_index_path))
    print(f"Saved FAISS index to {settings.faiss_index_path}")

    # Build and save hierarchical docstore (v2 format)
    docstore = {
        "version": "2.0",
        "chunks": {
            "parents": [p.model_dump() for p in parents],
            "children": [c.model_dump() for c in children],
        },
        "index_mapping": {str(i): children[i].chunk_id for i in range(len(children))},
        "metadata": {
            "total_parents": len(parents),
            "total_children": len(children),
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
            "parent_chunk_size": settings.parent_chunk_size,
            "child_chunk_size": settings.child_chunk_size,
            "hierarchy_enabled": True,
        },
    }

    with open(settings.docstore_path, "w", encoding="utf-8") as f:
        json.dump(docstore, f, indent=2, ensure_ascii=False)
    print(f"Saved hierarchical docstore to {settings.docstore_path}")


def embed_hierarchical_chunks(
    children: list[HierarchicalChunk], model: SentenceTransformer | None = None
) -> np.ndarray:
    """
    Generate embeddings for child chunks (only children are embedded).

    Args:
        children: List of child chunks to embed.
        model: Optional pre-loaded model. If None, loads from settings.

    Returns:
        Numpy array of embeddings with shape (num_children, embedding_dim).
    """
    if model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        model = SentenceTransformer(settings.embedding_model)

    texts = [child.text for child in children]
    print(f"Generating embeddings for {len(texts)} child chunks...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def ingest_all(docs_dir: Path | None = None, use_hierarchy: bool | None = None) -> dict:
    """
    Run full ingestion pipeline.

    Args:
        docs_dir: Optional directory containing documents.
        use_hierarchy: Whether to use hierarchical chunking.
                       If None, uses settings.hierarchy_enabled.

    Returns:
        Statistics about the ingestion.
    """
    start_time = time.perf_counter()
    request_id = generate_request_id()
    audit_logger = get_audit_logger()

    # Determine chunking mode
    if use_hierarchy is None:
        use_hierarchy = settings.hierarchy_enabled

    # Resolve docs_dir
    resolved_docs_dir = docs_dir or settings.docs_dir
    source_path = str(resolved_docs_dir)

    # Track failure count
    failure_count = 0
    documents_processed = 0
    chunks_created = 0
    doc_ids_sample: list[str] = []

    try:
        # Load documents
        documents = load_documents(docs_dir)
        if not documents:
            print("No documents found to ingest")
            stats = {"documents": 0, "chunks": 0}

            # Log empty ingestion
            latency_ms = (time.perf_counter() - start_time) * 1000
            event = IngestionEvent(
                request_id=request_id,
                actor=Actor(auth_method="cli"),
                source_type="filesystem",
                source_path=source_path,
                documents_processed=0,
                chunks_created=0,
                failure_count=0,
                doc_ids_sample=[],
                policy_decision="no_documents",
                latency_ms=latency_ms,
            )
            audit_logger.log(event)

            return stats

        documents_processed = len(documents)
        doc_ids_sample = [doc.metadata.doc_id for doc in documents[:5]]

        # Load embedding model once
        model = SentenceTransformer(settings.embedding_model)

        if use_hierarchy:
            # Hierarchical chunking
            print("Using hierarchical chunking...")
            parents, children = chunk_documents_hierarchical(documents)

            # Generate embeddings for children only
            embeddings = embed_hierarchical_chunks(children, model)

            # Build hierarchical index
            build_hierarchical_index(embeddings, parents, children)

            chunks_created = len(children)
            stats = {
                "documents": len(documents),
                "parents": len(parents),
                "children": len(children),
                "hierarchy_enabled": True,
                "index_path": str(settings.faiss_index_path),
                "docstore_path": str(settings.docstore_path),
            }
        else:
            # Flat chunking (legacy mode)
            print("Using flat chunking...")
            chunks = chunk_documents(documents)

            # Generate embeddings
            embeddings = embed_chunks(chunks, model)

            # Build flat index
            build_index(embeddings, chunks)

            chunks_created = len(chunks)
            stats = {
                "documents": len(documents),
                "chunks": len(chunks),
                "hierarchy_enabled": False,
                "index_path": str(settings.faiss_index_path),
                "docstore_path": str(settings.docstore_path),
            }

        print(f"\nIngestion complete: {stats}")

    except Exception as e:
        failure_count = 1
        raise
    finally:
        # Always log the ingestion event
        latency_ms = (time.perf_counter() - start_time) * 1000
        event = IngestionEvent(
            request_id=request_id,
            actor=Actor(auth_method="cli"),
            source_type="filesystem",
            source_path=source_path,
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            failure_count=failure_count,
            doc_ids_sample=doc_ids_sample,
            policy_decision="completed" if failure_count == 0 else "failed",
            resource_type="index",
            latency_ms=latency_ms,
        )
        audit_logger.log(event)

    return stats


if __name__ == "__main__":
    ingest_all()
