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

#### 4. Run Evaluations

```bash
# Run all evaluations (full suite)
python -m src.app eval

# Run specific perspectives
python -m src.app eval --perspective retrieval,safety

# Smoke test (fast, for CI/PR checks)
python -m src.app eval --suite smoke

# Save execution traces for failed cases
python -m src.app eval --save-trace -v

# Compare against baseline for regression detection
python -m src.app eval --baseline reports/evals/previous_report.json
```

Example output:
```
============================================================
Running Evaluation Suite
============================================================
  Perspectives: all
  Suite: full
  Save traces: False
============================================================

============================================================
Evaluation Summary
============================================================

  Overall: 45/50 (90.0%)

  ✓ retrieval: 9/10 (90.0%)
  ✓ context_quality: 8/10 (80.0%)
  ✓ groundedness: 9/10 (90.0%)
  ✓ safety: 10/10 (100.0%)
  ✓ pipeline: 9/10 (90.0%)

  Reports generated:
    JSON: reports/evals/eval_report_20260122_143052.json
    Markdown: reports/evals/eval_report_20260122_143052.md
============================================================
```

#### 5. View System Info

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
│       ├── prompts.py         # System/user prompt templates
│       └── evals/             # Evaluation framework
│           ├── models.py      # Eval data models (cases, results, traces)
│           ├── metrics.py     # Metric calculations (MRR, NDCG, AUC, etc.)
│           ├── retrieval_evals.py      # A. Retrieval evaluation
│           ├── context_quality_evals.py # B. Context quality evaluation
│           ├── groundedness_evals.py   # C. Groundedness evaluation
│           ├── safety_evals.py         # E. Safety evaluation
│           ├── pipeline_evals.py       # F. Pipeline evaluation
│           ├── runner.py      # EvalRunner orchestrator
│           └── report.py      # Report generation (JSON/Markdown)
├── data/
│   └── docs/                  # Source documents
│       ├── public/            # Public documents
│       ├── internal/          # Internal documents
│       └── confidential/      # Confidential documents
├── indexes/                   # Generated FAISS index
├── logs/                      # Audit logs (hash-chained JSON)
├── reports/evals/             # Generated evaluation reports
├── traces/                    # Execution traces for debugging
├── tests/                     # Test suite
│   ├── test_rbac.py           # RBAC filtering tests
│   ├── test_audit.py          # Audit logging tests
│   ├── test_guardrails_*.py   # Guardrails tests (detection, mitigation, reproduction)
│   ├── test_evals.py          # Evaluation framework tests
│   └── ...
│   └── fixtures/evals/        # Evaluation fixtures
│       ├── cases.jsonl        # Base evaluation cases
│       ├── retrieval_labels.jsonl
│       ├── context_labels.jsonl
│       ├── groundedness_labels.jsonl
│       └── pipeline_labels.jsonl
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
| `make eval` | Run full evaluation suite |
| `make eval-smoke` | Run smoke evaluation suite |

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
- [x] **Step 6**: Evals integration
  - Perspective-based evaluation framework (Retrieval, Context Quality, Groundedness, Safety, Pipeline)
  - Heuristic metrics (no LLM calls) for cost-free CI runs
  - JSON/Markdown report generation with regression detection
  - CLI integration with smoke/full suite modes
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

### Evaluation Framework

The system includes a comprehensive evaluation framework organized by **evaluation perspective** for clear diagnosis (e.g., "retrieval is good but groundedness is bad" → focus on prompt engineering).

#### Evaluation Perspectives

| Perspective | Description | Key Metrics |
|-------------|-------------|-------------|
| **A. Retrieval** | How well does search find relevant documents? | NDCG@k (primary), MRR, Recall@k, Precision@k |
| **B. Context Quality** | Is the retrieved context useful? | Redundancy ratio, fact dispersion, unique token ratio |
| **C. Groundedness** | Is the answer supported by context? | Claim support rate, citation validity, numeric fabrication |
| **E. Safety** | Are guardrails effective? | AUC, TPR@FPR=1%, TPR@FPR=5% |
| **F. Pipeline** | End-to-end integration | Outcome validation, latency budgets, RBAC impact |

Note: D. Answer Quality (LLM-as-Judge) is deferred to Step 7.

#### Metric Targets

| Perspective | Metric | Target |
|-------------|--------|--------|
| Retrieval | NDCG@5 | > 0.6 |
| Retrieval | Recall@5 | > 0.7 |
| Context Quality | Redundancy Ratio | < 0.2 |
| Groundedness | Claim Support Rate | > 0.85 |
| Groundedness | Citation Validity (form) | > 0.95 |
| Safety | AUC (injection) | > 0.85 |
| Safety | TPR@FPR=1% | > 0.7 |
| Pipeline | Pass Rate | > 0.9 |

#### Design Principles

1. **Heuristic-based**: All metrics use computational methods (no LLM calls) for cost-free CI runs
2. **Query-centric fixtures**: Shared `case_id` across perspective-specific label files
3. **Trace storage**: Failed cases save execution traces for debugging
4. **Regression detection**: Compare against baseline reports for detecting regressions
5. **Diagnostic guidance**: Reports include actionable recommendations based on pass/fail patterns

#### CLI Options

```bash
python -m src.app eval [OPTIONS]

Options:
  --perspective TEXT     Perspectives: retrieval,context_quality,groundedness,
                         safety,pipeline,all (default: all)
  --suite {smoke,full}   Suite mode: smoke (fast) or full (default: full)
  --save-trace           Save execution traces for failed cases
  --output TEXT          Report name (default: auto-generated timestamp)
  --baseline PATH        Path to baseline JSON report for regression detection
  -v, --verbose          Verbose output with detailed metrics
```

#### Configurable Thresholds (EvalSettings)

Evaluation thresholds can be tuned via environment variables:

```bash
# Algorithm parameters
EVALS__CLAIM_OVERLAP_THRESHOLD=0.3       # Minimum n-gram overlap for claim-context match
EVALS__INFERENCE_THRESHOLD_RATIO=0.7     # Relaxation ratio for inference claims

# Success criteria thresholds
EVALS__MIN_CLAIM_SUPPORT_RATE=0.85       # Minimum claim support rate for success
EVALS__MIN_CITATION_VALIDITY_FORM=0.95   # Minimum citation validity (form) for success

# Context quality parameters
EVALS__REDUNDANCY_THRESHOLD=0.5          # N-gram overlap threshold for redundancy detection
EVALS__TFIDF_SIMILARITY_THRESHOLD=0.8    # TF-IDF cosine similarity threshold

# Retrieval parameters
EVALS__RETRIEVAL_K_VALUES=[1,3,5,10]     # Values of k for @k metrics
```

Example usage:
```bash
# Run evals with stricter claim support rate
EVALS__MIN_CLAIM_SUPPORT_RATE=0.90 python -m src.app eval --perspective groundedness
```

## Future Extensibility

### Hallucination Detection

The current Groundedness evaluation (`groundedness_evals.py`) provides a foundation for hallucination detection through claim-context alignment analysis. Future enhancements may include:

- **Real-time hallucination detection**: Integrate claim support rate checking into the generation pipeline
- **Confidence scoring**: Assign confidence levels to generated claims based on context overlap
- **Unsupported claim handling**: Automatic flagging or removal of claims with low support rates
- **LLM-based verification**: Optional LLM judge for complex inference validation (with cost/latency trade-offs)

## License

MIT License
