"""Configuration settings for RAG system."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class GuardrailSettings(BaseModel):
    """Guardrails configuration with classification-based thresholds.

    This model configures the input/output guardrails for detecting
    prompt injection and data leakage attacks.
    """

    # Feature flags
    input_guardrail_enabled: bool = Field(
        default=True,
        description="Enable input guardrail (injection detection)",
    )
    output_guardrail_enabled: bool = Field(
        default=True,
        description="Enable output guardrail (leakage detection)",
    )
    log_guardrail_events: bool = Field(
        default=True,
        description="Log guardrail events to audit log",
    )

    # Input guardrail settings
    max_query_length: int = Field(
        default=2000,
        description="Maximum query length before anomaly score increases",
    )

    # Input guardrail thresholds (fixed - not classification-based)
    # Note: Classification-based thresholds were removed as a security improvement.
    # If a user account is compromised, varying thresholds by role or classification
    # would be exploitable. Using a fixed threshold protects against this.
    #
    # Action determination based on score:
    #   score < injection_allow_threshold  → ALLOW
    #   score < injection_warn_threshold   → WARN
    #   score < injection_block_threshold  → REDACT
    #   score >= injection_block_threshold → BLOCK
    injection_allow_threshold: float = Field(
        default=0.25,
        description="Threshold below which queries are allowed (no action)",
    )
    injection_warn_threshold: float = Field(
        default=0.40,
        description="Threshold below which queries trigger a warning",
    )
    injection_block_threshold: float = Field(
        default=0.50,
        description="Threshold at or above which queries are blocked",
    )

    # Debug/logging settings
    log_raw_content: bool = Field(
        default=False,
        description="Log raw content in guardrail events (debug only)",
    )


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
        description="Handler type: rotating_file, stdout_json, or memory (tests)",
    )

    @property
    def log_path(self) -> Path:
        """Full path to audit log file."""
        return self.log_dir / self.log_file


class EvalSettings(BaseModel):
    """Evaluation framework configuration.

    This model configures thresholds and parameters for the evaluation
    framework across different perspectives (retrieval, context quality,
    groundedness, safety, pipeline).
    """

    # Groundedness evaluation settings - algorithm parameters
    claim_overlap_threshold: float = Field(
        default=0.3,
        description="Minimum n-gram overlap ratio for claim-context match (assertions)",
    )
    inference_threshold_ratio: float = Field(
        default=0.7,
        description="Ratio applied to claim_overlap_threshold for inference claims (more lenient)",
    )

    # Hybrid lexical matching settings (3-stage claim verification)
    min_key_terms: int = Field(
        default=3,
        description="Minimum key terms required to confirm support at stage 1",
    )
    key_term_threshold: float = Field(
        default=0.5,
        description="Minimum key term match ratio (weighted) for support",
    )
    number_weight: float = Field(
        default=2.0,
        description="Weight multiplier for numbers in key term matching",
    )
    jaccard_threshold: float = Field(
        default=0.25,
        description="Minimum Jaccard similarity for sentence-level matching",
    )
    ngram_fallback_threshold: float = Field(
        default=0.15,
        description="N-gram overlap threshold for fallback matching (lowered from 0.3)",
    )

    # Groundedness evaluation settings - success criteria thresholds
    min_claim_support_rate: float = Field(
        default=0.85,
        description="Minimum claim support rate for success (0.0-1.0)",
    )
    min_citation_validity_form: float = Field(
        default=0.95,
        description="Minimum citation validity (form) rate for success (0.0-1.0)",
    )

    # Context quality evaluation settings
    tfidf_similarity_threshold: float = Field(
        default=0.8,
        description="TF-IDF cosine similarity threshold for redundancy detection",
    )

    # Retrieval evaluation settings
    retrieval_k_values: list[int] = Field(
        default=[1, 3, 5, 10],
        description="Values of k for @k metrics (Recall@k, Precision@k, NDCG@k)",
    )


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
        default=800,
        description="Child chunk size in characters (~265 tokens)",
    )
    child_chunk_overlap: int = Field(
        default=100,
        description="Overlap between child chunks in characters",
    )
    parent_preview_size: int = Field(
        default=1500,
        description="Initial preview size for parent context (chars, 50% of parent)",
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
        default="claude-haiku-4-5-20251001",
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

    # Guardrails settings (nested)
    guardrails: GuardrailSettings = Field(
        default_factory=GuardrailSettings,
        description="Guardrails configuration for security",
    )

    # Evaluation settings (nested)
    evals: EvalSettings = Field(
        default_factory=EvalSettings,
        description="Evaluation framework configuration",
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
