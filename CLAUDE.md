# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

An enterprise-grade RAG (Retrieval-Augmented Generation) system implementation project. The goal is to build a system that can reproduce, detect, and mitigate failure modes including Prompt Injection, Data Leakage, Hallucination, Cost Explosion, and Rate Limiting.

## Tech Stack

- **Python 3.10+**
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2) - runs locally
- **Vector Store**: FAISS (faiss-cpu)
- **LLM**: Anthropic Claude API
- **Schema**: Pydantic v2
- **Text Splitting**: LangChain text splitters

## Directory Structure

```
src/
├── app.py              # CLI entry point
└── rag/
    ├── config.py       # Settings class (includes AuditSettings, GuardrailSettings, EvalSettings)
    ├── models.py       # Pydantic models
    ├── ingest.py       # Document ingestion pipeline
    ├── retrieve.py     # Vector search with RBAC
    ├── generate.py     # Answer generation with citations
    ├── rbac.py         # Role-Based Access Control with mandatory audit
    ├── audit.py        # Enterprise audit logging (hash chain, tamper detection)
    ├── guardrails.py   # Input/Output guardrails (injection/leakage detection)
    ├── prompts.py      # System/user prompt templates
    └── evals/          # Evaluation framework
        ├── models.py           # Eval data models (cases, results, traces)
        ├── metrics.py          # Metric calculations (MRR, NDCG, AUC, etc.)
        ├── retrieval_evals.py  # A. Retrieval evaluation
        ├── context_quality_evals.py  # B. Context quality evaluation
        ├── groundedness_evals.py     # C. Groundedness evaluation
        ├── safety_evals.py           # E. Safety evaluation
        ├── pipeline_evals.py         # F. Pipeline evaluation
        ├── runner.py           # EvalRunner orchestrator
        └── report.py           # Report generation (JSON/Markdown)

data/docs/              # Source documents (.md + .meta.json)
├── public/
├── internal/
└── confidential/

indexes/                # FAISS index + docstore.json
logs/                   # Audit logs (hash-chained JSON)
reports/evals/          # Generated evaluation reports
traces/                 # Execution traces for debugging
tests/                  # pytest tests
tests/fixtures/evals/   # Evaluation fixtures (cases.jsonl, *_labels.jsonl)
```

## Allowed Commands
- git status
- git diff
- git log
- git branch
- git show
- python *
- pytest *
- poetry *
- pip *

## Disallowed Commands
- sudo *
- shutdown *

## Mandatory Development Workflow Rules

### Branching Rules (MUST)

- If the current git branch is `main`, Claude MUST:
  1. Create a new branch following the naming conventions below
  2. Switch to that branch
  3. Only then start planning or implementation

- Claude MUST NEVER start planning or implementation work directly on `main`.


### Branch Naming Conventions

Claude MUST use the following branch prefixes and meanings:

- **feature/**  
  General feature development.  
  Includes:
  - Application logic
  - Test implementation
  - Infrastructure or CDK-related implementation

- **fix/**  
  Bug fixes intended to be released as hotfixes.

- **docs/**  
  Documentation updates only.  
  Includes:
  - CLAUDE.md
  - README.md
  - Any other documentation files

- **spec/**  
  Specification or proposal work ONLY.
  - MUST NOT include any implementation code
  - Used for specifications, design documents, or proposals

#### Restrictions by Branch Type

- **spec/** branches:
  - Claude MUST NOT write or modify any implementation code
  - Only documentation files may be changed

- **docs/** branches:
  - Claude MUST NOT modify application logic

### Planning Rules (Plan Mode)

- All plans created in Plan Mode MUST be persisted to a file.
- Plan files MUST be created under:
  ```
  $PROJECT_ROOT/.claude/plans/
  ```
- File naming convention:
  ```
  plan-<scope>.md
  ```
- Plans MUST be:
  - Step-by-step
  - Explicit
  - Implementation-ready

Claude MUST NOT rely solely on chat output for planning.

### Testing Rules (MUST)

- After implementation, Claude MUST:
  1. Create appropriate test code
  2. Place all tests under:
     ```
     tests/
     ```
  3. Execute the test suite

- Claude MUST repeat fixing and testing until **all tests pass**.
- Claude MUST NOT declare a task complete while any test is failing.

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Ingest documents
python -m src.app ingest

# Test search
python -m src.app search "query"

# Ask a question (RAG)
python -m src.app ask "question"
python -m src.app ask "question" --json  # JSON output
python -m src.app ask "question" -H      # Hierarchical retrieval

# Show system info
python -m src.app info

# Run evaluations
python -m src.app eval                           # Full suite
python -m src.app eval --suite smoke             # Smoke test (fast)
python -m src.app eval --perspective retrieval   # Specific perspective
python -m src.app eval --save-trace -v           # With traces and verbose

# Run tests
make test

# Run linter
make lint

# Format code
make format
```

## Document Metadata Format

Each document consists of a `.md` file paired with a `.meta.json` sidecar file:

```json
{
  "doc_id": "public-001",
  "classification": "public|internal|confidential",
  "allowed_roles": ["employee", "contractor"],
  "pii_flag": false,
  "source": "confluence"
}
```

## Implementation Roadmap

1. ✅ **Step 1**: Ingest + Retrieval (no RBAC)
2. ✅ **Step 2**: Generation (with citations)
3. ✅ **Step 3**: RBAC filter (role-based)
4. ✅ **Step 4**: Audit logging
5. ✅ **Step 5**: Guardrails (Injection / Leakage) - detect & mitigate
6. ✅ **Step 6**: Evals integration
7. ⬜ **Step 7**: Remaining failures (Hallucination / Cost / Rate Limiting)

## Coding Conventions

- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff
- **Type hints**: Required (Pydantic models and function signatures)
- **Docstrings**: Google style

## Testing

- Framework: pytest
- Test directory: `tests/`
- Fixtures: `tests/conftest.py`
- Run: `make test` or `pytest tests/ -v`

## Guardrails Architecture

### Input Guardrail (Injection Detection)

Composite scoring with 5 components (weights: pattern 0.25, structural 0.25, delimiter 0.15, anomaly 0.15, jailbreak_intent 0.20):

- Fixed threshold (not classification-based) for security
- Actions: ALLOW → WARN → BLOCK

### Output Guardrail (Leakage Detection)

Two-lane architecture:

1. **Sanitize Lane**: PII/Secret/Metadata detection
   - Sets `sanitize_needed` flag → caller calls `redact()` if True
   - High-confidence secrets (PEM, AWS key, JWT) → immediate BLOCK

2. **Content Lane**: Verbatim/substring overlap detection
   - Classification-based thresholds (stricter for confidential)
   - Actions: ALLOW → WARN → BLOCK

Key design principle: `check()` for inspection, `redact()` for transformation (separation of concerns).

## Evaluation Framework

Perspective-based evaluation for clear diagnosis (e.g., "retrieval good, groundedness bad" → focus on prompts).

### Perspectives

| Perspective | Description | Key Metrics |
|-------------|-------------|-------------|
| A. Retrieval | Document/chunk retrieval quality | NDCG@k, MRR, Recall@k, Precision@k |
| B. Context Quality | Retrieved context usefulness | Redundancy ratio, fact dispersion |
| C. Groundedness | Answer-context alignment | Claim support rate, citation validity |
| E. Safety | Guardrail effectiveness | AUC, TPR@FPR=1% |
| F. Pipeline | End-to-end integration | Outcome validation, latency |

### EvalSettings (config.py)

Configurable thresholds via environment variables:

```bash
# Algorithm parameters
EVALS__CLAIM_OVERLAP_THRESHOLD=0.3       # Claim-context matching
EVALS__INFERENCE_THRESHOLD_RATIO=0.7     # Relaxation for inference claims

# Success criteria
EVALS__MIN_CLAIM_SUPPORT_RATE=0.85       # Minimum claim support rate
EVALS__MIN_CITATION_VALIDITY_FORM=0.95   # Minimum citation validity

# Context quality
EVALS__REDUNDANCY_THRESHOLD=0.5          # N-gram overlap for redundancy
EVALS__TFIDF_SIMILARITY_THRESHOLD=0.8    # TF-IDF cosine similarity
```

### Design Principles

- **Heuristic-based**: No LLM calls for cost-free CI runs
- **Query-centric fixtures**: Shared `case_id` across perspectives
- **Trace storage**: Execution traces for debugging failed cases
- **Action-based detection**: Safety evals use `action != ALLOW` (consistent with guardrails)

## Notes

- Embedding model runs locally (no API cost)
- FAISS index is saved to `indexes/`
- Set `ANTHROPIC_API_KEY` in `.env` file (required for Generation step)
