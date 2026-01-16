"""Configuration settings for RAG system."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


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

    # Anthropic API (for future generation step)
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key",
    )

    class Config:
        env_prefix = ""
        case_sensitive = False

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
