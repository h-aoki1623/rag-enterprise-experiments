"""Pydantic models for RAG system."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Classification(str, Enum):
    """Document classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    doc_id: str = Field(..., description="Unique document identifier")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
    classification: Classification = Field(..., description="Security classification")
    allowed_roles: list[str] = Field(
        default_factory=list, description="Roles allowed to access this document"
    )
    pii_flag: bool = Field(default=False, description="Whether document contains PII")
    source: str = Field(default="unknown", description="Source system (confluence, pdf, etc.)")


class Chunk(BaseModel):
    """A chunk of text from a document with inherited metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Parent document ID")
    metadata: DocumentMetadata = Field(..., description="Inherited document metadata")


class RetrievalResult(BaseModel):
    """Result from retrieval with relevance score."""

    chunk: Chunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., description="Relevance score (higher is more relevant)")
    rank: int = Field(..., description="Rank in retrieval results (1-indexed)")


class Document(BaseModel):
    """A full document with content and metadata."""

    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    file_path: Optional[str] = Field(default=None, description="Original file path")
