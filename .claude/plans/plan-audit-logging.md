# Audit Logging Implementation Plan (Step 4)

## Overview

Implement enterprise-grade audit logging for the RAG system. This audit log captures **who accessed what, when, and with what outcome** - distinct from application/debug logs.

### Audit Log vs Application Log Boundary

| Audit Log | Application Log |
|-----------|-----------------|
| Access decisions (granted/denied) | Stack traces, exceptions |
| Data asset access (read/write) | Internal state, debug info |
| Administrative operations | Performance tuning details |
| Minimal error info (type, component, correlation ID) | Full error details |

---

## Implementation Scope

### New Files

| File | Purpose |
|------|---------|
| `src/rag/audit.py` | Core audit logging module (models, logger, formatter) |
| `tests/test_audit.py` | Unit and integration tests for audit logging |
| `logs/.gitkeep` | Ensure logs directory exists in repo |

### Files to Modify

| File | Changes |
|------|---------|
| `src/rag/config.py` | Add AuditSettings class |
| `src/rag/rbac.py` | Add access decision logging |
| `src/rag/retrieve.py` | Add retrieval event logging |
| `src/rag/generate.py` | Add generation event logging |
| `src/rag/ingest.py` | Add ingestion event logging |

---

## Step 1: Core Audit Module (`src/rag/audit.py`)

### Enums

```python
class AuditEventType(str, Enum):
    """Types of audit events."""
    # Security Events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    # Operational Events
    RETRIEVAL_COMPLETE = "retrieval_complete"
    GENERATION_COMPLETE = "generation_complete"
    INGESTION_COMPLETE = "ingestion_complete"

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


class AuditAction(str, Enum):
    """Actions being audited."""
    ACCESS_CHECK = "access_check"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    INGEST = "ingest"


class DenialReason(str, Enum):
    """Standardized denial reasons for RBAC."""
    NO_USER_CONTEXT = "no_user_context"
    TENANT_MISMATCH = "tenant_mismatch"
    ROLE_MISMATCH = "role_mismatch"
    NO_ALLOWED_ROLES = "no_allowed_roles"
    CLASSIFICATION_DENIED = "classification_denied"
```

### Actor Model (Trust Boundary)

```python
class Actor(BaseModel):
    """Authenticated principal with trust boundary distinction."""

    # Verified identity (after authentication)
    authenticated_user_id: Optional[str] = None
    authenticated_tenant_id: Optional[str] = None

    # Asserted values (from request, may differ)
    asserted_user_id: Optional[str] = None
    asserted_tenant_id: Optional[str] = None
    asserted_roles: list[str] = Field(default_factory=list)

    # Authentication context
    auth_method: Optional[str] = None  # "api_key", "jwt", "cli", etc.
```

### Base AuditEvent Model

```python
class AuditEvent(BaseModel):
    """Base audit event with required fields."""

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
    prev_hash: Optional[str] = None
```

### Specialized Event Models

#### AccessDecisionEvent

```python
class AccessDecisionEvent(AuditEvent):
    """RBAC access decision event."""

    event_type: AuditEventType = AuditEventType.ACCESS_GRANTED
    component: AuditComponent = AuditComponent.RBAC
    action: AuditAction = AuditAction.ACCESS_CHECK

    # Access-specific fields
    doc_id: str
    chunk_id: Optional[str] = None
    classification: str
    decision: str  # "granted" or "denied"
    denial_reason: Optional[DenialReason] = None
    decision_basis: list[str] = Field(default_factory=list)  # ["role_match", "tenant_match"]
```

#### RetrievalEvent

```python
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
```

#### GenerationEvent

```python
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
```

#### IngestionEvent

```python
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
```

### AuditLogger Class

```python
class AuditLogger:
    """Enterprise audit logger with hash chain and configurable handlers."""

    _instance: Optional["AuditLogger"] = None
    _lock: threading.Lock = threading.Lock()
    _last_hash: Optional[str] = None

    def __init__(self, settings: "AuditSettings"):
        self.settings = settings
        self._logger = self._setup_logger()

    @classmethod
    def get_instance(cls, settings: Optional["AuditSettings"] = None) -> "AuditLogger":
        """Thread-safe singleton."""
        with cls._lock:
            if cls._instance is None:
                if settings is None:
                    from .config import settings as app_settings
                    settings = app_settings.audit
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
            cls._last_hash = None

    def _compute_hash(self, event_json: str) -> str:
        """Compute hash for tamper detection chain."""
        content = f"{self._last_hash or 'GENESIS'}:{event_json}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def log(self, event: AuditEvent) -> None:
        """Log an audit event with hash chain."""
        if not self.settings.enabled:
            return

        # Add hash chain
        event_json = event.model_dump_json(exclude={"prev_hash"})
        event.prev_hash = self._compute_hash(event_json)
        self._last_hash = event.prev_hash

        # Mask sensitive data if enabled
        if self.settings.mask_sensitive_data:
            event = self._mask_sensitive(event)

        # Log
        final_json = event.model_dump_json()
        log_method = getattr(self._logger, event.severity.value.lower())
        log_method(final_json)
```

---

## Step 2: Configuration (`src/rag/config.py`)

```python
class AuditSettings(BaseModel):
    """Audit logging configuration."""

    enabled: bool = Field(default=True, description="Enable/disable audit logging")
    log_level: str = Field(default="INFO", description="Log level")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", description="Log directory")
    log_file: str = Field(default="audit.log", description="Audit log filename")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="10MB max")
    backup_count: int = Field(default=5, description="Backup files to keep")
    console_output: bool = Field(default=False, description="Console output (default OFF)")
    mask_sensitive_data: bool = Field(default=True, description="Mask queries/PII")

    # Handler configuration for different environments
    handler_type: str = Field(
        default="rotating_file",
        description="Handler type: rotating_file, stdout_json, memory (for tests)"
    )

    @property
    def log_path(self) -> Path:
        return self.log_dir / self.log_file
```

---

## Step 3: RBAC Logging (`src/rag/rbac.py`)

Modify `check_access()`:

- Log ACCESS_GRANTED/ACCESS_DENIED with:
  - `decision_basis`: ["role_match", "tenant_match", "classification_ok"]
  - `denial_reason`: DenialReason enum value
  - `doc_id`, `chunk_id` as required resource_ids
  - `pii_accessed` flag

---

## Step 4: Retrieval Logging (`src/rag/retrieve.py`)

Add logging to `retrieve()` and `retrieve_hierarchical()`:

- `request_id`: Pass through from caller or generate
- `query_hash`: SHA256[:16]
- `embedding_model`: From settings
- `k_requested`, `results_before_filter`, `results_after_filter`, `top_k_returned`
- `filter_applied`: ["rbac", "tenant", "metadata"]
- `classifications_accessed`: List of unique classifications
- `pii_accessed`: True if any result has pii_flag

---

## Step 5: Generation Logging (`src/rag/generate.py`)

Add logging to `generate()`:

- `query_hash`: SHA256[:16]
- `model`: From settings
- `context_chunks`: Number of context chunks
- `context_token_estimate`: Rough estimate
- `output_token_estimate`: From response usage
- `policy_flags`: List of PolicyFlag values
- `confidence`: Score
- `refusal`: True if generation was blocked
- `refusal_reason`: If applicable

---

## Step 6: Ingestion Logging (`src/rag/ingest.py`)

Add logging to `ingest_all()`:

- `source_type`: "filesystem"
- `source_path`: docs_dir path
- `documents_processed`: Count
- `chunks_created`: Count
- `failure_count`: Error count
- `doc_ids_sample`: First 5 doc_ids

---

## Step 7: Tests (`tests/test_audit.py`)

### Unit Tests
- AuditEvent model serialization/validation
- AuditLogger singleton pattern (thread-safe)
- JSON formatter output
- Log file creation and rotation
- Hash chain integrity verification
- Sensitive data masking verification

### Security Tests
- **PII not in logs**: Query text, document content, generation output masked
- **Credentials not logged**: API keys, tokens filtered

### Integration Tests
- **request_id propagation**: retrieve → generate events share request_id
- **Disabled mode**: `AUDIT__ENABLED=false` produces no output
- **Multi-thread safety**: 100+ concurrent logs don't corrupt JSON

### Handler Tests
- MemoryHandler for test assertions
- StreamHandler switch for CI

---

## Sample Log Output

```json
{
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "request_id": "req-xyz-123",
  "timestamp": "2025-01-19T12:34:56.789Z",
  "event_type": "retrieval_complete",
  "severity": "INFO",
  "actor": {
    "authenticated_user_id": "user-123",
    "authenticated_tenant_id": "acme-corp",
    "asserted_roles": ["employee"],
    "auth_method": "cli"
  },
  "component": "retrieve",
  "action": "retrieve",
  "resource_type": "chunk",
  "resource_ids": ["doc1-chunk-001", "doc2-chunk-003"],
  "pii_accessed": false,
  "policy_decision": "allowed",
  "latency_ms": 127.5,
  "schema_version": "1.0",
  "prev_hash": "8f3a9b2c1d4e5f67",
  "details": {
    "query_hash": "abc123def456",
    "embedding_model": "all-MiniLM-L6-v2",
    "k_requested": 5,
    "results_before_filter": 15,
    "results_after_filter": 5,
    "top_k_returned": 5,
    "filter_applied": ["rbac", "tenant"],
    "classifications_accessed": ["public", "internal"]
  }
}
```

---

## Verification

1. **Unit Tests**: `pytest tests/test_audit.py -v`
2. **Hash Chain Verification**:
   ```bash
   python -c "from src.rag.audit import verify_hash_chain; verify_hash_chain('logs/audit.log')"
   ```
3. **Integration Test**:
   ```bash
   python -m src.app search "test query" --tenant acme-corp --roles employee
   cat logs/audit.log | jq .
   ```
4. **Full Pipeline Test**:
   ```bash
   python -m src.app ingest
   python -m src.app ask "What is the company policy?" --tenant acme-corp --roles employee
   # Verify request_id links retrieval and generation events
   cat logs/audit.log | jq 'select(.request_id == "...")'
   ```
5. **All Tests**: `make test`

---

## Critical Files

- [config.py](src/rag/config.py) - Add AuditSettings
- [rbac.py](src/rag/rbac.py) - Add access logging with decision_basis
- [retrieve.py](src/rag/retrieve.py) - Add retrieval logging with filter details
- [generate.py](src/rag/generate.py) - Add generation logging with token estimates
- [ingest.py](src/rag/ingest.py) - Add ingestion logging with failure tracking

---

## Future Considerations (Out of Scope)

- **Production**: Switch to `stdout_json` handler for container environments
- **External storage**: Forward to SIEM/log aggregator
- **Digital signatures**: For stronger tamper evidence
