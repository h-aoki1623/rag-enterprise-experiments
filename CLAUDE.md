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
    ├── config.py       # Settings class
    ├── models.py       # Pydantic models
    ├── ingest.py       # Document ingestion pipeline
    └── retrieve.py     # Vector search

data/docs/              # Source documents (.md + .meta.json)
├── public/
├── internal/
└── confidential/

indexes/                # FAISS index + docstore.json
tests/                  # pytest tests
```

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Ingest documents
python -m src.app ingest

# Test search
python -m src.app search "query"

# Show system info
python -m src.app info

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
  "tenant_id": "acme-corp",
  "classification": "public|internal|confidential",
  "allowed_roles": ["employee", "contractor"],
  "pii_flag": false,
  "source": "confluence"
}
```

## Implementation Roadmap

1. ✅ **Step 1**: Ingest + Retrieval (no RBAC)
2. ⬜ **Step 2**: Generation (with citations)
3. ⬜ **Step 3**: RBAC filter (tenant/role)
4. ⬜ **Step 4**: Audit logging
5. ⬜ **Step 5**: Failure modes (Injection / Leakage) - reproduce & mitigate
6. ⬜ **Step 6**: Evals integration
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

## Notes

- Embedding model runs locally (no API cost)
- FAISS index is saved to `indexes/`
- Set `ANTHROPIC_API_KEY` in `.env` file (required for Generation step)
