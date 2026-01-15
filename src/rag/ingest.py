"""Document ingestion pipeline: load, chunk, embed, and index."""

import json
import uuid
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import Chunk, Document, DocumentMetadata


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
    Build FAISS index and save docstore.

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

    # Build and save docstore
    docstore = {
        "chunks": [chunk.model_dump() for chunk in chunks],
        "metadata": {
            "total_chunks": len(chunks),
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
        },
    }

    with open(settings.docstore_path, "w", encoding="utf-8") as f:
        json.dump(docstore, f, indent=2, ensure_ascii=False)
    print(f"Saved docstore to {settings.docstore_path}")


def ingest_all(docs_dir: Path | None = None) -> dict:
    """
    Run full ingestion pipeline.

    Args:
        docs_dir: Optional directory containing documents.

    Returns:
        Statistics about the ingestion.
    """
    # Load documents
    documents = load_documents(docs_dir)
    if not documents:
        print("No documents found to ingest")
        return {"documents": 0, "chunks": 0}

    # Chunk documents
    chunks = chunk_documents(documents)

    # Load embedding model once
    model = SentenceTransformer(settings.embedding_model)

    # Generate embeddings
    embeddings = embed_chunks(chunks, model)

    # Build and save index
    build_index(embeddings, chunks)

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(settings.faiss_index_path),
        "docstore_path": str(settings.docstore_path),
    }

    print(f"\nIngestion complete: {stats}")
    return stats


if __name__ == "__main__":
    ingest_all()
