# Enterprise RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to demonstrate failure mode handling, security controls, and audit capabilities.

## Features

- **Document Ingestion**: Load, chunk, and embed documents with metadata
- **Vector Search**: FAISS-based similarity search with relevance scoring
- **Role-Based Access Control**: Fine-grained access control via user roles
- **Classification Levels**: Public, Internal, and Confidential document handling
- **Hierarchical Chunking**: Parent-child chunk relationships for better context
- **Answer Generation**: Citations, confidence scores, and policy flags
- **Enterprise Audit Logging**: Hash-chained tamper-evident logs with mandatory access tracking

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-enterprise-experiments

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` to set your Anthropic API key (required for generation):

```
ANTHROPIC_API_KEY=your-api-key-here
```

### Usage

#### 1. Ingest Documents

```bash
python -m src.app ingest
```

This will:
- Load all documents from `data/docs/`
- Split them into chunks
- Generate embeddings using local model
- Build FAISS index

#### 2. Search Documents

```bash
# Basic search (no RBAC filtering)
python -m src.app search "What is the vacation policy?"

# Search with role-based access control
python -m src.app search "What is the vacation policy?" --roles employee

# Hierarchical search
python -m src.app search "What is the vacation policy?" --roles employee -H
```

#### 3. Ask Questions (RAG)

```bash
# Basic question with role-based access
python -m src.app ask "What is the vacation policy?" --roles employee

# JSON output format
python -m src.app ask "What is the vacation policy?" --roles employee --json

# Use hierarchical retrieval
python -m src.app ask "What is the vacation policy?" --roles employee -H

# Specify number of chunks to retrieve
python -m src.app ask "What is the vacation policy?" --roles employee -k 3

# Public-only access (no roles required)
python -m src.app ask "What products do we offer?"
```

Example output:
```
============================================================
Question: What is the vacation policy?
Retrieval mode: Flat
Top-k: 5
User Context: roles=['employee']
============================================================

--- Answer ---
Employees receive 15 days of paid vacation per year.

Confidence: 0.95

--- Citations ---
[1] test-internal-001 / test-internal-001-chunk-0
    "Employees receive 15 days of paid vacation per year."

--- Policy Flags ---
  - (none)
```

JSON output format:
```json
{
  "answer": "Employees receive 15 days of paid vacation per year.",
  "citations": [
    {
      "doc_id": "test-internal-001",
      "chunk_id": "test-internal-001-chunk-0",
      "text_snippet": "Employees receive 15 days of paid vacation per year."
    }
  ],
  "confidence": 0.95,
  "policy_flags": []
}
```

#### 4. View System Info

```bash
python -m src.app info
```

## Project Structure

```
rag-enterprise-experiments/
├── src/
│   ├── app.py                 # CLI entry point
│   └── rag/
│       ├── config.py          # Configuration settings (includes AuditSettings)
│       ├── models.py          # Pydantic data models
│       ├── ingest.py          # Document ingestion pipeline
│       ├── retrieve.py        # Vector search layer with RBAC
│       ├── generate.py        # Answer generation with citations
│       ├── rbac.py            # Role-Based Access Control with mandatory audit
│       ├── audit.py           # Enterprise audit logging (hash chain)
│       ├── guardrails.py      # Input/Output guardrails (injection/leakage)
│       └── prompts.py         # System/user prompt templates
├── data/
│   └── docs/                  # Source documents
│       ├── public/            # Public documents
│       ├── internal/          # Internal documents
│       └── confidential/      # Confidential documents
├── indexes/                   # Generated FAISS index
├── logs/                      # Audit logs (hash-chained JSON)
├── tests/                     # Test suite
│   ├── test_rbac.py           # RBAC filtering tests
│   ├── test_audit.py          # Audit logging tests
│   ├── test_guardrails_*.py   # Guardrails tests (detection, mitigation, reproduction)
│   └── ...
├── pyproject.toml             # Project configuration
├── Makefile                   # Common commands
└── README.md
```

## Document Format

Documents are stored as Markdown files with JSON metadata sidecars:

```
data/docs/public/
├── company-overview.md
└── company-overview.meta.json
```

Metadata schema:

```json
{
  "doc_id": "public-001",
  "classification": "public",
  "allowed_roles": ["employee", "contractor", "public"],
  "pii_flag": false,
  "source": "confluence"
}
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format
```

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make dev-install` | Install with dev dependencies |
| `make ingest` | Run document ingestion |
| `make search Q="query"` | Search with query |
| `make ask Q="query"` | Ask a question using RAG |
| `make info` | Show system information |
| `make test` | Run tests |
| `make lint` | Run linter |
| `make format` | Format code |
| `make clean` | Remove index files |

## Implementation Roadmap

- [x] **Step 1**: Ingest + Retrieval (no RBAC)
  - Hierarchical chunking with parent-child relationships
  - FAISS-based vector search
- [x] **Step 2**: Generation (with citations)
  - Answer generation with Claude API
  - Mandatory citations and confidence scores
  - Policy flags for PII/confidential access
- [x] **Step 3**: RBAC filter (role-based)
  - Role-based access control
  - Post-retrieval filtering with over-fetch strategy
- [x] **Step 4**: Audit logging
  - Hash-chained tamper-evident logs
  - Mandatory access decision logging
  - Trust boundary aware actor model
  - Sensitive data masking
  - Hash chain persistence across restarts
- [x] **Step 5**: Guardrails (Injection / Leakage)
  - Input guardrail: Composite scoring injection detection
  - Output guardrail: Two-lane architecture (sanitize + content)
  - PII/Secret/Metadata detection and redaction
  - Classification-based thresholds
- [ ] **Step 6**: Evals integration
- [ ] **Step 7**: Remaining failures (Hallucination / Cost / Rate Limiting)

## Architecture

### Ingestion Pipeline

```
Documents → Load → Chunk → Embed → FAISS Index
                              ↓
                         Docstore (JSON)
```

### Retrieval Flow (with RBAC)

```
Query + UserContext → Embed → FAISS Search (k×3) → RBAC Filter → Top-K Chunks → Results
                                                         ↓
                                                   (Role Check)
```

### Generation Flow (RAG with Guardrails)

```
Query + UserContext → [Input Guardrail] → Retrieve Top-K → Build Context → LLM Generation → [Output Guardrail] → Response
                            ↓                    ↓                                               ↓
                     Injection Detection    RBAC Filtering                            Leakage Detection/Redaction
                     (BLOCK/WARN/ALLOW)                                               {answer, citations, confidence, policy_flags}
```

**Security features:**
- **Input Guardrail**: Composite scoring injection detection (pattern, structural, delimiter, anomaly, jailbreak intent)
- **Output Guardrail**: Two-lane architecture for leakage detection and PII/secret redaction
- Mandatory citations for all answers
- Policy flags for PII/confidential data access visibility
- Confidence scoring for answer reliability

## Security Considerations

### Access Control (RBAC)

- **Role-Based Access**: Fine-grained permissions
  - `user_context=None` → No access to any documents
  - Public documents: Accessible to all authenticated users
  - Internal/Confidential: Requires specific roles in `allowed_roles`
- **Post-Retrieval Filtering**: Over-fetch strategy (k×3, expandable to k×5)
  - Ensures k results after RBAC filtering
  - No modification to FAISS index required

### Data Protection

- **PII flagging** for sensitive documents
- **Policy flags** for visibility into accessed data
- **Mandatory citations** for answer traceability
- **Confidence scoring** for answer reliability

### Audit Logging

Enterprise-grade audit logging with:

- **Hash Chain**: Each log event includes `prev_event_hash` and `event_hash` for tamper detection
- **Mandatory Logging**: All RBAC access decisions are logged (no opt-out)
- **Trust Boundary**: Actor model tracks `authenticated_id` vs `asserted_id`
- **Sensitive Data Masking**: Queries and PII can be masked in logs
- **Persistence**: Hash chain continues across process restarts
- **Configurable Handlers**: File rotation, stdout JSON (containers), or memory (tests)

Audit log format (JSON Lines):
```json
{
  "timestamp": "2026-01-20T10:30:00Z",
  "request_id": "req-abc123",
  "event_type": "access_granted",
  "severity": "INFO",
  "actor": {
    "user_id": "user-001",
    "roles": ["employee"],
    "auth_method": "cli"
  },
  "doc_id": "internal-001",
  "classification": "internal",
  "decision": "granted",
  "decision_basis": ["role_match"],
  "prev_event_hash": "a1b2c3d4e5f6",
  "event_hash": "f6e5d4c3b2a1"
}
```

Verify hash chain integrity:
```python
from src.rag.audit import verify_hash_chain
is_valid, errors = verify_hash_chain("logs/audit.log")
```

### Guardrails

The system includes comprehensive input/output guardrails for security:

#### Input Guardrail (Injection Detection)

Composite scoring approach with 5 detection components:

| Component | Weight | Detection Target |
|-----------|--------|------------------|
| Pattern | 0.25 | Known attack phrases ("ignore instructions", "reveal system prompt") |
| Structural | 0.25 | Instruction-like grammar (imperative verbs, role assignment) |
| Delimiter | 0.15 | Prompt boundary manipulation (`[system]`, fake markers) |
| Anomaly | 0.15 | Statistical outliers (excessive length, encoded payloads) |
| Jailbreak Intent | 0.20 | Bypass intent ("bypass filters", "DAN mode") |

Actions: `ALLOW` → `WARN` → `BLOCK` based on fixed thresholds.

#### Output Guardrail (Leakage Detection)

Two-lane architecture:

1. **Sanitize Lane**: Detects PII, secrets, and metadata
   - Sets `sanitize_needed` flag → triggers redaction
   - High-confidence secrets (PEM keys, AWS keys, JWT) → immediate `BLOCK`

2. **Content Lane**: Detects verbatim/substring overlap with source context
   - Classification-based thresholds (stricter for confidential)
   - Actions: `ALLOW` → `WARN` → `BLOCK`

Detected patterns are automatically redacted:
- Email → `[EMAIL REDACTED]`
- Phone → `[PHONE REDACTED]`
- SSN → `[SSN REDACTED]`
- API keys → `[AWS_KEY REDACTED]`, `[GITHUB_TOKEN REDACTED]`
- doc_id → `[DOC_ID REDACTED]`

### Planned

- **Hallucination** detection and mitigation
- **Cost explosion** prevention
- **Rate limiting** controls

## License

MIT License
