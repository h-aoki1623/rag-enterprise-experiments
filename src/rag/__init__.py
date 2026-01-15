"""RAG module for document ingestion and retrieval."""

from .config import settings
from .ingest import ingest_all
from .models import Chunk, DocumentMetadata, RetrievalResult
from .retrieve import retrieve

__all__ = [
    "settings",
    "ingest_all",
    "retrieve",
    "Chunk",
    "DocumentMetadata",
    "RetrievalResult",
]
