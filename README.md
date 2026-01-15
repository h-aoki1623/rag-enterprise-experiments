# Enterprise RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to demonstrate failure mode handling, security controls, and audit capabilities.

## Features

- **Document Ingestion**: Load, chunk, and embed documents with metadata
- **Vector Search**: FAISS-based similarity search with relevance scoring
- **Multi-tenant Support**: Tenant isolation via metadata filtering
- **Classification Levels**: Public, Internal, and Confidential document handling
- **Audit Trail**: Complete traceability of retrieval and generation

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
python -m src.app search "What is the vacation policy?"
```

#### 3. View System Info

```bash
python -m src.app info
```

## Project Structure

```
rag-enterprise-experiments/
├── src/
│   ├── app.py                 # CLI entry point
│   └── rag/
│       ├── config.py          # Configuration settings
│       ├── models.py          # Pydantic data models
│       ├── ingest.py          # Document ingestion pipeline
│       └── retrieve.py        # Vector search layer
├── data/
│   └── docs/                  # Source documents
│       ├── public/            # Public documents
│       ├── internal/          # Internal documents
│       └── confidential/      # Confidential documents
├── indexes/                   # Generated FAISS index
├── tests/                     # Test suite
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
  "tenant_id": "acme-corp",
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
| `make info` | Show system information |
| `make test` | Run tests |
| `make lint` | Run linter |
| `make format` | Format code |
| `make clean` | Remove index files |

## Implementation Roadmap

- [x] **Step 1**: Ingest + Retrieval (no RBAC)
- [ ] **Step 2**: Generation (with citations)
- [ ] **Step 3**: RBAC filter (tenant/role)
- [ ] **Step 4**: Audit logging
- [ ] **Step 5**: Failure modes (Injection / Leakage)
- [ ] **Step 6**: Evals integration
- [ ] **Step 7**: Remaining failures (Hallucination / Cost / Rate Limiting)

## Architecture

### Ingestion Pipeline

```
Documents → Load → Chunk → Embed → FAISS Index
                              ↓
                         Docstore (JSON)
```

### Retrieval Flow

```
Query → Embed → FAISS Search → Top-K Chunks → Results
```

## Security Considerations

- **Classification-based access control** (planned)
- **Tenant isolation** via metadata filtering
- **PII flagging** for sensitive documents
- **Audit logging** for compliance (planned)

## License

MIT License
