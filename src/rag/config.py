"""Configuration settings for RAG system."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class AuditSettings(BaseModel):
    """Audit logging configuration.

    This model configures the enterprise audit logging system with
    support for different handlers and environments.
    """

    enabled: bool = Field(
        default=True,
        description="Enable/disable audit logging",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory for log files",
    )
    log_file: str = Field(
        default="audit.log",
        description="Audit log filename",
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,
        description="Max log file size before rotation (10MB default)",
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup files to keep",
    )
    console_output: bool = Field(
        default=False,
        description="Also output to console (default OFF for production)",
    )
    mask_sensitive_data: bool = Field(
        default=True,
        description="Mask queries and PII in logs",
    )
    handler_type: Literal["rotating_file", "stdout_json", "memory"] = Field(
        default="rotating_file",
        description="Handler type: rotating_file (default), stdout_json (containers), memory (tests)",
    )

    @property
    def log_path(self) -> Path:
        """Full path to audit log file."""
        return self.log_dir / self.log_file


class Settings(BaseSettings):
    """RAG system configuration."""

    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimension (384 for MiniLM)",
    )

    # Chunking settings (flat chunking)
    chunk_size: int = Field(
        default=1500,
        description="Target chunk size in characters (~500 tokens)",
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between chunks in characters",
    )

    # Hierarchical chunking settings
    hierarchy_enabled: bool = Field(
        default=True,
        description="Enable hierarchical chunking (parent-child structure)",
    )
    parent_chunk_size: int = Field(
        default=3000,
        description="Parent chunk size in characters (~1000 tokens)",
    )
    child_chunk_size: int = Field(
        default=500,
        description="Child chunk size in characters (~165 tokens)",
    )
    child_chunk_overlap: int = Field(
        default=50,
        description="Overlap between child chunks in characters",
    )
    parent_preview_size: int = Field(
        default=500,
        description="Initial preview size for parent context (chars)",
    )

    # Retrieval settings
    default_top_k: int = Field(
        default=5,
        description="Default number of chunks to retrieve",
    )
    max_top_k: int = Field(
        default=10,
        description="Maximum allowed top_k for retrieval",
    )

    # Paths
    docs_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "docs",
        description="Directory containing source documents",
    )
    index_dir: Path = Field(
        default=PROJECT_ROOT / "indexes",
        description="Directory for FAISS index storage",
    )

    # Index file names
    faiss_index_file: str = Field(
        default="faiss.index",
        description="FAISS index file name",
    )
    docstore_file: str = Field(
        default="docstore.json",
        description="Document store JSON file name",
    )

    # Anthropic API settings
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key",
    )
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model to use for generation",
    )
    generation_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for generation",
    )
    generation_temperature: float = Field(
        default=0.0,
        description="Temperature for generation (0.0 for deterministic)",
    )

    # Audit logging settings (nested)
    audit: AuditSettings = Field(
        default_factory=AuditSettings,
        description="Audit logging configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",  # Allows AUDIT__ENABLED=false
        case_sensitive=False,
    )

    @property
    def faiss_index_path(self) -> Path:
        """Full path to FAISS index file."""
        return self.index_dir / self.faiss_index_file

    @property
    def docstore_path(self) -> Path:
        """Full path to docstore JSON file."""
        return self.index_dir / self.docstore_file


# Global settings instance
settings = Settings()
