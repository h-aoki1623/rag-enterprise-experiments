"""Enterprise audit logging module for RAG system.

This module provides structured audit logging with:
- Trust boundary aware actor model
- Hash chain for tamper detection
- Configurable handlers (file, stdout, memory)
- Sensitive data masking
"""

import hashlib
import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .config import AuditSettings


# =============================================================================
# Enums
# =============================================================================


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Security Events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    # Operational Events
    RETRIEVAL_COMPLETE = "retrieval_complete"
    GENERATION_COMPLETE = "generation_complete"
    INGESTION_COMPLETE = "ingestion_complete"

    # Guardrail Events
    GUARDRAIL_EVENT = "guardrail_event"

    # Error Events (minimal info only)
    ERROR = "error"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditComponent(str, Enum):
    """System components for audit logging."""

    RBAC = "rbac"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    INGEST = "ingest"
    GUARDRAILS = "guardrails"


class AuditAction(str, Enum):
    """Actions being audited."""

    ACCESS_CHECK = "access_check"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    INGEST = "ingest"
    GUARDRAIL_CHECK = "guardrail_check"


class DenialReason(str, Enum):
    """Standardized denial reasons for RBAC."""

    NO_USER_CONTEXT = "no_user_context"
    ROLE_MISMATCH = "role_mismatch"
    NO_ALLOWED_ROLES = "no_allowed_roles"
    CLASSIFICATION_DENIED = "classification_denied"


# =============================================================================
# Models
# =============================================================================


class Actor(BaseModel):
    """Authenticated principal with trust boundary distinction.

    This model distinguishes between verified (authenticated) identity
    and asserted (claimed) identity from requests.
    """

    # Verified identity (after authentication)
    authenticated_user_id: Optional[str] = None

    # Asserted values (from request, may differ)
    asserted_user_id: Optional[str] = None
    asserted_roles: list[str] = Field(default_factory=list)

    # Authentication context
    auth_method: Optional[str] = None  # "api_key", "jwt", "cli", etc.


class AuditEvent(BaseModel):
    """Base audit event with required fields.

    All audit events inherit from this base class and include
    common fields for correlation, classification, and tracking.
    """

    # Identity
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(..., description="Correlation ID for request tracing")

    # Timestamps
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Event classification
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO

    # Actor (trust boundary aware)
    actor: Actor

    # Common structured fields
    component: AuditComponent
    action: AuditAction
    resource_type: Optional[str] = None  # "document", "chunk", "index", "prompt"
    resource_ids: list[str] = Field(default_factory=list)

    # Policy/Security flags
    pii_accessed: bool = False
    policy_decision: Optional[str] = None  # "allowed", "blocked", "allowed_with_redaction"

    # Performance
    latency_ms: Optional[float] = None

    # Schema versioning
    schema_version: str = "1.0"

    # Extended details (with defined common keys)
    details: dict[str, Any] = Field(default_factory=dict)

    # Tamper detection (hash chain)
    prev_event_hash: Optional[str] = None  # Hash of the previous event
    event_hash: Optional[str] = None  # Hash of this event

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class AccessDecisionEvent(AuditEvent):
    """RBAC access decision event."""

    component: AuditComponent = AuditComponent.RBAC
    action: AuditAction = AuditAction.ACCESS_CHECK

    # Access-specific fields
    doc_id: str
    chunk_id: Optional[str] = None
    classification: str
    decision: str  # "granted" or "denied"
    denial_reason: Optional[DenialReason] = None
    decision_basis: list[str] = Field(default_factory=list)  # ["role_match", "tenant_match"]


class RetrievalEvent(AuditEvent):
    """Retrieval operation event."""

    event_type: AuditEventType = AuditEventType.RETRIEVAL_COMPLETE
    component: AuditComponent = AuditComponent.RETRIEVE
    action: AuditAction = AuditAction.RETRIEVE

    # Retrieval-specific fields
    query_hash: str  # SHA256[:16] of query
    index_name: str = "default"
    embedding_model: str

    # Results
    k_requested: int
    results_before_filter: int
    results_after_filter: int
    top_k_returned: int
    filter_applied: list[str] = Field(default_factory=list)  # ["rbac", "tenant"]

    # Classifications accessed
    classifications_accessed: list[str] = Field(default_factory=list)


class GenerationEvent(AuditEvent):
    """Generation operation event."""

    event_type: AuditEventType = AuditEventType.GENERATION_COMPLETE
    component: AuditComponent = AuditComponent.GENERATE
    action: AuditAction = AuditAction.GENERATE

    # Generation-specific fields
    query_hash: str
    model: str

    # Token estimates
    context_chunks: int
    context_token_estimate: Optional[int] = None
    output_token_estimate: Optional[int] = None

    # Results
    policy_flags: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    refusal: bool = False
    refusal_reason: Optional[str] = None


class IngestionEvent(AuditEvent):
    """Ingestion operation event."""

    event_type: AuditEventType = AuditEventType.INGESTION_COMPLETE
    component: AuditComponent = AuditComponent.INGEST
    action: AuditAction = AuditAction.INGEST

    # Ingestion-specific fields
    source_type: str  # "filesystem", "s3", "db"
    source_path: str

    # Results
    documents_processed: int
    chunks_created: int
    failure_count: int = 0
    doc_ids_sample: list[str] = Field(default_factory=list)  # First few doc_ids


class GuardrailAuditEvent(AuditEvent):
    """Guardrail-related audit event.

    Logs input/output guardrail checks for security monitoring.
    Does not log raw content by default - uses hashes and statistics.
    """

    event_type: AuditEventType = AuditEventType.GUARDRAIL_EVENT
    component: AuditComponent = AuditComponent.GUARDRAILS
    action: AuditAction = AuditAction.GUARDRAIL_CHECK

    # Required fields
    guardrail_type: str  # "input" or "output"
    threat_type: Optional[str] = None
    threat_score: float = 0.0
    action_taken: str  # "allow" | "warn" | "redact" | "block"

    # Score breakdown (for post-analysis)
    score_breakdown: dict = Field(default_factory=dict)

    # Context info (no raw data)
    input_hash: Optional[str] = None  # Query hash
    input_length: Optional[int] = None
    doc_set_fingerprint: Optional[str] = None  # Hash of doc_ids
    doc_count: int = 0
    classifications_involved: list[str] = Field(default_factory=list)

    # Model info
    model_id: Optional[str] = None

    # Detection details (masked)
    matched_pattern_count: int = 0
    pii_detected_count: int = 0
    verbatim_ratio: Optional[float] = None


# =============================================================================
# JSON Formatter
# =============================================================================


class AuditJsonFormatter(logging.Formatter):
    """JSON formatter that outputs pre-formatted JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON (already formatted by AuditLogger)."""
        return record.getMessage()


# =============================================================================
# Memory Handler (for testing)
# =============================================================================


class MemoryHandler(logging.Handler):
    """In-memory handler for testing audit logs."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted log message."""
        with self._lock:
            self.records.append(self.format(record))

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self.records.clear()

    def get_records(self) -> list[str]:
        """Get a copy of all stored records."""
        with self._lock:
            return list(self.records)


# =============================================================================
# Audit Logger
# =============================================================================


class AuditLogger:
    """Enterprise audit logger with hash chain and configurable handlers.

    This is a thread-safe singleton that provides:
    - Structured JSON logging
    - Hash chain for tamper detection
    - Configurable handlers (file, stdout, memory)
    - Sensitive data masking
    """

    _instance: Optional["AuditLogger"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, settings: "AuditSettings") -> None:
        self.settings = settings
        self._last_hash: Optional[str] = None
        self._hash_lock = threading.Lock()
        self._memory_handler: Optional[MemoryHandler] = None
        self._logger = self._setup_logger()

        # Load last hash from existing log file to continue chain across restarts
        if self.settings.handler_type == "rotating_file":
            self._load_last_hash_from_file()

    @classmethod
    def get_instance(cls, settings: Optional["AuditSettings"] = None) -> "AuditLogger":
        """Get or create the singleton instance (thread-safe)."""
        with cls._lock:
            if cls._instance is None:
                if settings is None:
                    from .config import settings as app_settings

                    settings = app_settings.audit
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                # Close handlers
                for handler in cls._instance._logger.handlers[:]:
                    handler.close()
                    cls._instance._logger.removeHandler(handler)
            cls._instance = None

    def _setup_logger(self) -> logging.Logger:
        """Configure the logger with appropriate handlers."""
        logger = logging.getLogger("rag.audit")
        logger.setLevel(getattr(logging, self.settings.log_level.upper(), logging.INFO))
        logger.propagate = False  # Don't propagate to root logger

        # Clear existing handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = AuditJsonFormatter()

        # Configure handler based on type
        if self.settings.handler_type == "memory":
            self._memory_handler = MemoryHandler()
            self._memory_handler.setFormatter(formatter)
            logger.addHandler(self._memory_handler)

        elif self.settings.handler_type == "stdout_json":
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        else:  # rotating_file (default)
            # Ensure log directory exists
            log_path = self.settings.log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=self.settings.max_file_size,
                backupCount=self.settings.backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Optional console output
        if self.settings.console_output and self.settings.handler_type != "stdout_json":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _load_last_hash_from_file(self) -> None:
        """Load the last event_hash from existing log file to continue chain.

        This ensures the hash chain continues across process restarts.
        Only applicable for file-based handlers.
        """
        import json

        log_path = self.settings.log_path
        if not log_path.exists():
            return

        try:
            # Read the last line of the log file efficiently
            last_line = None
            with open(log_path, "rb") as f:
                # Seek to end of file
                f.seek(0, 2)  # 0 offset from end (2)
                file_size = f.tell()

                if file_size == 0:
                    return

                # Read backwards to find last newline
                buffer_size = min(4096, file_size)
                f.seek(max(0, file_size - buffer_size))
                content = f.read().decode("utf-8")

                # Find the last non-empty line
                lines = content.strip().split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        last_line = line
                        break

            if last_line:
                event = json.loads(last_line)
                self._last_hash = event.get("event_hash")

        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # If we can't read the file, start fresh
            # This is acceptable as it only affects chain continuity, not integrity
            pass

    def _mask_sensitive(self, event: AuditEvent) -> AuditEvent:
        """Mask sensitive data in the event if masking is enabled."""
        # Create a copy to avoid modifying the original
        event_dict = event.model_dump()

        # Mask sensitive fields in details
        sensitive_keys = ["query", "answer", "content", "text", "api_key", "token", "password"]
        if "details" in event_dict:
            for key in sensitive_keys:
                if key in event_dict["details"]:
                    event_dict["details"][key] = "<MASKED>"

        return type(event).model_validate(event_dict)

    def log(self, event: AuditEvent) -> None:
        """Log an audit event with hash chain."""
        if not self.settings.enabled:
            return

        # Compute hash chain atomically
        with self._hash_lock:
            # Set prev_event_hash from the last logged event
            event.prev_event_hash = self._last_hash

            # Compute event_hash for this event (excluding hash fields)
            event_json_for_hash = event.model_dump_json(exclude={"prev_event_hash", "event_hash"})
            content = f"{event.prev_event_hash or 'GENESIS'}:{event_json_for_hash}"
            event.event_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Update last hash for next event
            self._last_hash = event.event_hash

        # Mask sensitive data if enabled
        if self.settings.mask_sensitive_data:
            event = self._mask_sensitive(event)

        # Log with appropriate severity
        final_json = event.model_dump_json()
        log_method = getattr(self._logger, event.severity.value.lower(), self._logger.info)
        log_method(final_json)

    def get_memory_records(self) -> list[str]:
        """Get records from memory handler (for testing)."""
        if self._memory_handler is not None:
            return self._memory_handler.get_records()
        return []

    def clear_memory_records(self) -> None:
        """Clear records from memory handler (for testing)."""
        if self._memory_handler is not None:
            self._memory_handler.clear()


# =============================================================================
# Helper Functions
# =============================================================================


def get_audit_logger() -> AuditLogger:
    """Get the singleton audit logger instance."""
    return AuditLogger.get_instance()


def generate_request_id() -> str:
    """Generate a new request ID for correlation."""
    return f"req-{uuid.uuid4().hex[:12]}"


def hash_query(query: str) -> str:
    """Hash a query string for privacy-preserving logging."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def create_actor_from_user_context(
    user_context: Optional[Any], auth_method: str = "cli"
) -> Actor:
    """Create an Actor from a UserContext object.

    Args:
        user_context: UserContext object or None.
        auth_method: Authentication method used.

    Returns:
        Actor instance with appropriate fields populated.
    """
    if user_context is None:
        return Actor(auth_method=auth_method)

    return Actor(
        # For CLI, asserted and authenticated are the same
        authenticated_user_id=getattr(user_context, "user_id", None),
        asserted_user_id=getattr(user_context, "user_id", None),
        asserted_roles=getattr(user_context, "user_roles", []),
        auth_method=auth_method,
    )


def verify_hash_chain(log_file: Path) -> tuple[bool, list[str]]:
    """Verify the hash chain integrity of an audit log file.

    Args:
        log_file: Path to the audit log file.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    import json

    errors: list[str] = []
    prev_event_hash: Optional[str] = None

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Verify prev_event_hash matches the previous event's event_hash
                recorded_prev_hash = event.get("prev_event_hash")
                if recorded_prev_hash != prev_event_hash:
                    errors.append(
                        f"Line {line_num}: prev_event_hash mismatch - "
                        f"expected {prev_event_hash}, got {recorded_prev_hash}"
                    )

                # Verify event_hash is correct
                recorded_event_hash = event.get("event_hash")
                if recorded_event_hash is None:
                    errors.append(f"Line {line_num}: Missing event_hash")
                    prev_event_hash = recorded_prev_hash  # Continue chain verification
                    continue

                # Compute expected event_hash
                event_copy = {
                    k: v for k, v in event.items() if k not in ("prev_event_hash", "event_hash")
                }
                event_json = json.dumps(event_copy, separators=(",", ":"), sort_keys=True)
                content = f"{prev_event_hash or 'GENESIS'}:{event_json}"
                expected_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                # Verify event_hash is correct
                if recorded_event_hash != expected_hash:
                    errors.append(
                        f"Line {line_num}: event_hash mismatch - "
                        f"expected {expected_hash}, got {recorded_event_hash}"
                    )

                # Update prev_event_hash for next iteration
                prev_event_hash = recorded_event_hash

    except FileNotFoundError:
        errors.append(f"Log file not found: {log_file}")
    except Exception as e:
        errors.append(f"Error reading log file: {e}")

    return len(errors) == 0, errors
