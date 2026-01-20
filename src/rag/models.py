"""Pydantic models for RAG system."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Classification(str, Enum):
    """Document classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class PolicyFlag(str, Enum):
    """Policy flags for generation results."""

    NO_CONTEXT = "no_context"  # No basis in documents
    PII_REFERENCED = "pii_referenced"  # Referenced PII-containing data
    CONFIDENTIAL = "confidential"  # Referenced confidential document
    UNCERTAIN = "uncertain"  # Low confidence in answer


class ChunkLevel(str, Enum):
    """Hierarchy level of a chunk."""

    PARENT = "parent"
    CHILD = "child"


class UserContext(BaseModel):
    """User context for RBAC filtering."""

    user_roles: list[str] = Field(
        default_factory=list, description="List of roles assigned to user"
    )
    user_id: Optional[str] = Field(
        default=None, description="User identifier for audit logging"
    )


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    doc_id: str = Field(..., description="Unique document identifier")
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


class HierarchicalChunk(BaseModel):
    """A chunk with hierarchy relationships for hierarchical chunking."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Root document ID")
    metadata: DocumentMetadata = Field(..., description="Inherited document metadata")
    level: ChunkLevel = Field(..., description="Hierarchy level of this chunk")
    parent_id: Optional[str] = Field(
        default=None, description="ID of parent chunk (None for parent-level chunks)"
    )
    children_ids: list[str] = Field(
        default_factory=list, description="IDs of child chunks (empty for child-level)"
    )
    section_header: Optional[str] = Field(
        default=None, description="Section header if this is a parent chunk"
    )


class RetrievalResult(BaseModel):
    """Result from retrieval with relevance score."""

    chunk: Chunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., description="Relevance score (higher is more relevant)")
    rank: int = Field(..., description="Rank in retrieval results (1-indexed)")


class HierarchicalRetrievalResult(BaseModel):
    """Result from hierarchical retrieval with parent context."""

    parent_chunk: HierarchicalChunk = Field(..., description="Parent chunk for context")
    matched_children: list[HierarchicalChunk] = Field(
        default_factory=list, description="Child chunks that matched the query"
    )
    child_scores: list[float] = Field(
        default_factory=list, description="Relevance scores for each matched child"
    )
    aggregate_score: float = Field(..., description="Combined relevance score")
    rank: int = Field(..., description="Rank in results (1-indexed)")


class Document(BaseModel):
    """A full document with content and metadata."""

    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    file_path: Optional[str] = Field(default=None, description="Original file path")


class Citation(BaseModel):
    """Citation information for a generated answer."""

    doc_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")
    text_snippet: str = Field(..., description="Excerpt from the source for verification")


class GenerationResult(BaseModel):
    """Result from answer generation."""

    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    policy_flags: list[PolicyFlag] = Field(default_factory=list, description="Policy flags")
    raw_context_used: bool = Field(default=True, description="Whether raw context was used")
