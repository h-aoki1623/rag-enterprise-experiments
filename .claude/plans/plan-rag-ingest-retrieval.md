# RAG System Implementation Plan: Step 1 - Ingest + Retrieval (No RBAC)

## Overview
Implement the foundation of the enterprise RAG system with document ingestion and vector retrieval capabilities. RBAC filtering will be added in Step 3.

## Directory Structure to Create

```
rag-enterprise-experiments/
├── src/
│   ├── __init__.py
│   ├── app.py                 # Main entry point (CLI for now)
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py          # Document chunking & embedding
│       ├── retrieve.py        # Vector search
│       ├── models.py          # Pydantic schemas
│       └── config.py          # Configuration settings
├── data/
│   └── docs/                  # Sample documents (3 classifications)
│       ├── public/            # .md files + .meta.json sidecars
│       ├── internal/
│       └── confidential/
├── indexes/                   # FAISS index storage
├── eval/
│   └── tests/
├── Makefile
├── pyproject.toml
├── .env.example
└── README.md
```

## Implementation Steps

### 1. Project Setup
- Create `pyproject.toml` with dependencies:
  - `anthropic` (Claude API)
  - `langchain`, `langchain-community` (text splitting)
  - `faiss-cpu` (vector store)
  - `sentence-transformers` (embeddings - local model)
  - `pydantic` (schemas)
  - `python-dotenv` (env management)
- Create `.env.example` with required environment variables

### 2. Pydantic Models (`src/rag/models.py`)
Define core data structures:
- `DocumentMetadata`: tenant_id, doc_id, classification, allowed_roles, pii_flag, source
- `Chunk`: chunk_id, text, doc_id, metadata (inherited from doc)
- `RetrievalResult`: chunks with relevance scores

### 3. Configuration (`src/rag/config.py`)
- Chunk size (500-800 tokens → ~2000 chars)
- Chunk overlap (100 chars)
- Embedding model name
- FAISS index path
- Default retrieval k

### 4. Ingestion Pipeline (`src/rag/ingest.py`)
Functions:
- `load_documents(docs_dir)` → List[Document with metadata]
- `chunk_documents(docs)` → List[Chunk] (inherits metadata)
- `embed_chunks(chunks)` → embeddings array
- `build_index(embeddings, chunks)` → saves FAISS index + docstore.json
- `ingest_all()` → orchestrates the pipeline

Key decisions:
- Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings (free, fast)
- Store metadata in separate `docstore.json` (FAISS doesn't store metadata well)
- Chunking with `RecursiveCharacterTextSplitter`

### 5. Retrieval Layer (`src/rag/retrieve.py`)
Functions:
- `load_index()` → FAISS index + docstore
- `retrieve(query, k=5)` → List[RetrievalResult]
- `embed_query(query)` → query embedding

No RBAC filtering yet - will be added in Step 3.

### 6. Sample Documents (`data/docs/`)
Create 3-5 sample documents per classification:
- `public/`: Company overview, product FAQ
- `internal/`: Employee handbook, IT policies
- `confidential/`: Executive salary info, M&A plans

**Metadata format**: JSON sidecar files alongside each document.
```
data/docs/public/
├── company-overview.md
├── company-overview.meta.json    # {"doc_id": "...", "classification": "public", ...}
├── product-faq.md
└── product-faq.meta.json
```

Each `.meta.json` contains:
```json
{
  "doc_id": "public-001",
  "tenant_id": "acme-corp",
  "classification": "public",
  "allowed_roles": ["employee", "contractor", "public"],
  "pii_flag": false,
  "source": "confluence"
}
```

### 7. CLI Entry Point (`src/app.py`)
Commands:
- `python -m src.app ingest` - Run ingestion pipeline
- `python -m src.app search "query"` - Test retrieval

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Create | Dependencies & project config |
| `.env.example` | Create | Environment template |
| `src/__init__.py` | Create | Package init |
| `src/rag/__init__.py` | Create | RAG module init |
| `src/rag/models.py` | Create | Pydantic schemas |
| `src/rag/config.py` | Create | Configuration |
| `src/rag/ingest.py` | Create | Ingestion pipeline |
| `src/rag/retrieve.py` | Create | Retrieval logic |
| `src/app.py` | Create | CLI entry point |
| `data/docs/public/*.md` | Create | Sample public docs |
| `data/docs/internal/*.md` | Create | Sample internal docs |
| `data/docs/confidential/*.md` | Create | Sample confidential docs |
| `Makefile` | Create | Common commands |

## Verification

After implementation, verify with:

```bash
# 1. Install dependencies
pip install -e .

# 2. Run ingestion
python -m src.app ingest
# Expected: Creates indexes/faiss.index and indexes/docstore.json

# 3. Test retrieval
python -m src.app search "What is the vacation policy?"
# Expected: Returns relevant chunks with scores from internal docs

python -m src.app search "What products does the company offer?"
# Expected: Returns relevant chunks from public docs
```

## Notes

- Embedding model runs locally (no API cost)
- FAISS index is stored on disk for persistence
- Metadata stored separately in JSON for flexibility
- This step focuses on correctness; RBAC and optimization come later
