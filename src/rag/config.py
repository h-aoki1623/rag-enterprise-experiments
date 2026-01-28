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

    # Output guardrail settings
    ngram_size: int = Field(
        default=5,
        description="N-gram size for verbatim detection",
    )
    max_verbatim_ratio: float = Field(
        default=0.4,
        description="Maximum allowed verbatim overlap ratio",
    )
    max_lcs_ratio: float = Field(
        default=0.5,
        description="Maximum allowed longest common subsequence ratio",
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

    # Output guardrail thresholds (classification-based)
    # Each classification has its own set of action thresholds.
    # Structure: {"public": {"allow": 0.4, "warn": 0.64, "block": 0.8}, ...}
    #
    # Action determination based on score:
    #   score < allow_threshold  → ALLOW
    #   score < warn_threshold   → WARN
    #   score < block_threshold  → REDACT
    #   score >= block_threshold → BLOCK
    leakage_thresholds: dict[str, dict[str, float]] = Field(
        default={
            "public": {"allow": 0.40, "warn": 0.64, "block": 0.80},
            "internal": {"allow": 0.30, "warn": 0.48, "block": 0.60},
            "confidential": {"allow": 0.20, "warn": 0.32, "block": 0.40},
        },
        description="Action thresholds for leakage detection by classification",
    )

    # Default thresholds (fallback when classification not found)
    default_leakage_allow_threshold: float = Field(
        default=0.30,
        description="Default allow threshold if classification not found",
    )
    default_leakage_warn_threshold: float = Field(
        default=0.48,
        description="Default warn threshold if classification not found",
    )
    default_leakage_block_threshold: float = Field(
        default=0.60,
        description="Default block threshold if classification not found",
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
    redundancy_threshold: float = Field(
        default=0.5,
        description="N-gram overlap threshold for detecting redundant chunks",
    )
    tfidf_similarity_threshold: float = Field(
        default=0.8,
        description="TF-IDF cosine similarity threshold for paraphrase redundancy",
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
        default=PROJECT_ROOT / "data" / "sample-docs",
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
