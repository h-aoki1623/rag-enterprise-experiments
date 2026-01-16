# Hierarchical Chunking Implementation Plan

## Overview

Extend the current Flat Chunking (RecursiveCharacterTextSplitter) to Hierarchical Chunking to improve search precision and context quality.

**Selected Configuration:**
- 2-level hierarchy (parent and child chunks)
- Search results return both parent and matched child chunks
- Preserve existing Flat Chunking as an option (switchable via settings)

## Architecture

```
Document → Parent Chunks (section-level, ~3000 chars)
               ↓
           Child Chunks (paragraph-level, ~500 chars) ← Indexed in FAISS

Retrieval Flow:
Query → FAISS search (children) → Resolve parent chunks → Return both parent + children
```

## Implementation Steps

### Step 1: Extend Data Models (`src/rag/models.py`)

```python
# New models to add
class ChunkLevel(str, Enum):
    PARENT = "parent"
    CHILD = "child"

class HierarchicalChunk(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    metadata: DocumentMetadata
    level: ChunkLevel
    parent_id: Optional[str] = None      # Parent ID for child chunks
    children_ids: list[str] = []         # Child IDs for parent chunks
    section_header: Optional[str] = None # Section heading

class HierarchicalRetrievalResult(BaseModel):
    parent_chunk: HierarchicalChunk
    matched_children: list[HierarchicalChunk]
    child_scores: list[float]
    aggregate_score: float
    rank: int
```

### Step 2: Add Configuration (`src/rag/config.py`)

```python
# New settings to add
hierarchy_enabled: bool = True          # Enable hierarchical chunking
parent_chunk_size: int = 3000           # Parent chunk size in chars
child_chunk_size: int = 500             # Child chunk size in chars
child_chunk_overlap: int = 50           # Child chunk overlap in chars
```

### Step 3: Hierarchical Chunking Logic (New: `src/rag/hierarchy.py`)

**Key Functions:**

1. `extract_sections(content: str)` - Split content by Markdown headers (`##`)
2. `create_parent_chunks(document: Document)` - Create parent chunks from document
3. `create_child_chunks(parent: HierarchicalChunk)` - Create child chunks from parent
4. `chunk_document_hierarchical(document: Document)` - Main entry point

**Chunk Size Configuration:**
| Level | Size | Overlap | Purpose |
|-------|------|---------|---------|
| Parent | 3000 chars | 0 | Context provision |
| Child | 500 chars | 50 chars | Vector search |

### Step 4: Update Ingest Pipeline (`src/rag/ingest.py`)

**Functions to Modify:**
- `chunk_documents()` - Switch between Flat/Hierarchical based on settings
- `build_index()` - Support new docstore format

**New docstore.json Format (v2):**
```json
{
  "version": "2.0",
  "chunks": {
    "parents": [...],
    "children": [...]
  },
  "index_mapping": {"0": "chunk-id-000", ...},
  "metadata": {
    "total_parents": 15,
    "total_children": 45,
    "hierarchy_enabled": true
  }
}
```

### Step 5: Update Retrieval Logic (`src/rag/retrieve.py`)

**Functions to Add:**
- `load_hierarchical_index()` - Load hierarchical index
- `retrieve_hierarchical()` - Hierarchical search with parent resolution

**Retrieval Algorithm:**
1. Search child chunks in FAISS (over-fetch: k*3 results)
2. Group by parent_id
3. Calculate aggregate score for each parent (max of child scores)
4. Return parent chunk + matched child chunks

### Step 6: Update CLI (`src/app.py`)

**Modifications:**
- `ingest` command: Add `--flat` option for legacy mode
- `search` command: Format hierarchical results display

## Files to Modify

| File | Changes |
|------|---------|
| [models.py](src/rag/models.py) | Add `ChunkLevel`, `HierarchicalChunk`, `HierarchicalRetrievalResult` |
| [config.py](src/rag/config.py) | Add hierarchical chunking settings |
| [hierarchy.py](src/rag/hierarchy.py) | **New file** - Hierarchical chunking logic |
| [ingest.py](src/rag/ingest.py) | Modify `chunk_documents()`, `build_index()` |
| [retrieve.py](src/rag/retrieve.py) | Add `retrieve_hierarchical()` |
| [app.py](src/app.py) | Add CLI options, update result display |

## Backward Compatibility

- Docstore v1 format remains readable
- `hierarchy_enabled=False` uses legacy Flat Chunking
- `--flat` flag explicitly selects Flat Chunking during ingest

## Verification

1. **Index Rebuild Test:**
   ```bash
   python -m src.app ingest
   # Verify indexes/docstore.json is in v2 format
   ```

2. **Search Test:**
   ```bash
   python -m src.app search "working hours"
   # Verify both parent and child chunks are displayed
   ```

3. **Backward Compatibility Test:**
   ```bash
   python -m src.app ingest --flat
   python -m src.app search "query"
   # Verify legacy results are returned
   ```

4. **Unit Tests:**
   ```bash
   make test
   # Verify new tests in tests/test_hierarchy.py pass
   ```
