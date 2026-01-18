# RBAC Filter Implementation Plan

## Overview

Implement Role-Based Access Control (RBAC) filtering for the RAG enterprise system to enforce multi-tenant isolation and role-based document access. This is Step 3 in the roadmap.

## Design Decisions

### 1. Post-Retrieval Filtering Strategy
- Fetch k×3 candidates from FAISS (e.g., k=5 → fetch 15)
- Apply RBAC filters (tenant_id + allowed_roles) to results
- Return top-k filtered results
- If filtered results < k, expand search up to k×5

**Rationale**:
- **Compatibility**: Works with existing FAISS index without modification
- **Simplicity**: No need for index reconstruction or custom FAISS extensions
- **Flexibility**: Dynamic access control without re-indexing
- **Performance**: Over-fetching is acceptable (fetch 15-25 candidates, filter, return 5)

### 2. Filtering Rules
1. **Tenant Isolation**: `chunk.metadata.tenant_id == user_context.tenant_id`
2. **Role-Based Access**: `user_roles ∩ allowed_roles ≠ ∅` (at least one matching role)
3. **Default Behavior**: When `user_context=None`, allow PUBLIC documents only (secure by default)

### 3. Hierarchical Mode
- Filter at **parent level** only (children inherit parent metadata)
- Simpler logic, consistent with parent-as-context pattern

## Implementation Steps

### Step 1: Create RBAC Module
**File**: `src/rag/rbac.py` (NEW)

```python
from typing import Optional
from pydantic import BaseModel, Field
from .models import DocumentMetadata, RetrievalResult, HierarchicalRetrievalResult

class UserContext(BaseModel):
    """User context for RBAC filtering."""
    tenant_id: str
    user_roles: list[str] = Field(default_factory=list)
    user_id: Optional[str] = None

    @classmethod
    def public_access(cls, tenant_id: str = "public") -> "UserContext":
        return cls(tenant_id=tenant_id, user_roles=["public"])

def check_access(
    chunk_metadata: DocumentMetadata,
    user_context: Optional[UserContext]
) -> tuple[bool, Optional[str]]:
    """Check RBAC access. Returns (has_access, denial_reason)."""
    # Default to public-only
    if user_context is None:
        user_context = UserContext.public_access()

    # Rule 1: Tenant isolation
    if chunk_metadata.tenant_id != user_context.tenant_id:
        return False, f"tenant_mismatch:{chunk_metadata.tenant_id}"

    # Rule 2: Role-based access
    if not chunk_metadata.allowed_roles:
        return False, "no_allowed_roles"

    user_roles_set = set(user_context.user_roles)
    allowed_roles_set = set(chunk_metadata.allowed_roles)

    if not (user_roles_set & allowed_roles_set):
        return False, f"role_mismatch:required={allowed_roles_set}"

    return True, None

def filter_retrieval_results(
    results: list[RetrievalResult],
    user_context: Optional[UserContext],
) -> list[RetrievalResult]:
    """Filter flat retrieval results by RBAC rules."""
    filtered = []
    rank = 1
    for result in results:
        has_access, _ = check_access(result.chunk.metadata, user_context)
        if has_access:
            filtered_result = RetrievalResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
            )
            filtered.append(filtered_result)
            rank += 1
    return filtered

def filter_hierarchical_results(
    results: list[HierarchicalRetrievalResult],
    user_context: Optional[UserContext],
) -> list[HierarchicalRetrievalResult]:
    """Filter hierarchical results by parent RBAC rules."""
    filtered = []
    rank = 1
    for result in results:
        has_access, _ = check_access(result.parent_chunk.metadata, user_context)
        if has_access:
            filtered_result = HierarchicalRetrievalResult(
                parent_chunk=result.parent_chunk,
                matched_children=result.matched_children,
                child_scores=result.child_scores,
                aggregate_score=result.aggregate_score,
                rank=rank,
            )
            filtered.append(filtered_result)
            rank += 1
    return filtered
```

### Step 2: Update Retrieve Module
**File**: [src/rag/retrieve.py](src/rag/retrieve.py)

**Changes to `retrieve()` function**:
1. Add parameter: `user_context: UserContext | None = None`
2. Implement over-fetch: `fetch_k = min(k * 3, settings.max_top_k * 3)`
3. Search FAISS with `fetch_k` to get more candidates
4. Build unfiltered results list
5. Apply filtering: `filtered_results = filter_retrieval_results(unfiltered_results, user_context)`
6. If `len(filtered_results) < k` and haven't hit max limit:
   - Expand search to `k * 5`
   - Re-search and re-filter
7. Return top-k filtered results

**Changes to `retrieve_hierarchical()` function**:
1. Add parameter: `user_context: UserContext | None = None`
2. Search children from FAISS as usual
3. Group by parent_id and build unfiltered hierarchical results
4. Apply filtering: `filtered_results = filter_hierarchical_results(unfiltered_results, user_context)`
5. Return filtered hierarchical results (parents are filtered, children inherit access)

### Step 3: Update Generate Module
**File**: [src/rag/generate.py](src/rag/generate.py)

**Changes**:
1. Add `user_context` parameter to `generate_with_retrieval()` function
2. Pass `user_context` to `retrieve()` or `retrieve_hierarchical()` calls

### Step 4: Update CLI
**File**: [src/app.py](src/app.py)

**Add CLI arguments**:
```python
ask_parser.add_argument(
    "--tenant",
    type=str,
    help="Tenant ID for multi-tenancy (default: acme-corp)",
)
ask_parser.add_argument(
    "--roles",
    type=str,
    help="Comma-separated list of user roles (e.g., 'employee,contractor')",
)

# Same for search_parser
```

**Build UserContext in `cmd_ask()` and `cmd_search()`**:
```python
user_context = None
if args.tenant or args.roles:
    tenant_id = args.tenant or "acme-corp"
    roles = args.roles.split(",") if args.roles else ["public"]
    user_context = UserContext(tenant_id=tenant_id, user_roles=roles)
```

**Display user context** in output for transparency.

### Step 5: Create Unit Tests
**File**: `tests/test_rbac.py` (NEW)

**Test classes**:
1. `TestUserContext` - Test model creation and validation
2. `TestCheckAccess` - Test tenant isolation, role matching, default behavior
3. `TestFilterRetrievalResults` - Test post-filtering preserves order, updates ranks
4. `TestFilterHierarchicalResults` - Test parent-level filtering

**Key scenarios**:
- Tenant isolation (different tenants cannot access each other's docs)
- Role-based access (employee can access internal, public cannot)
- Default behavior (None context → public-only)
- Empty allowed_roles (secure default: deny access)
- Multiple roles (any match grants access)

### Step 6: Add Integration Tests
**File**: [tests/test_retrieve.py](tests/test_retrieve.py)

**New test class**: `TestRBACRetrieval`

**Scenarios**:
- Employee can access PUBLIC and INTERNAL
- Public role limited to PUBLIC docs only
- Executive can access CONFIDENTIAL
- Tenant isolation enforced
- No context defaults to public
- Over-fetch and filter strategy works
- Search expansion on insufficient results

**New test class**: `TestRBACHierarchicalRetrieval`
- Hierarchical filtering uses parent metadata
- Matched children preserved after parent filter

### Step 7: Add E2E Tests
**File**: `tests/test_e2e_rbac.py` (NEW)

**Real-world scenarios**:
- Public user asks about salaries → no confidential data returned
- Employee asks about vacation policy → gets internal policy
- Executive gets comprehensive results
- Multi-tenant isolation enforced

## Critical Files to Modify

1. **`src/rag/rbac.py`** (NEW) - Core RBAC logic
2. **`src/rag/retrieve.py`** - Add user_context parameter and filtering
3. **`src/rag/generate.py`** - Pass user_context through
4. **`src/app.py`** - CLI integration with --tenant and --roles
5. **`tests/test_rbac.py`** (NEW) - Unit tests
6. **`tests/test_retrieve.py`** - Integration tests
7. **`tests/test_e2e_rbac.py`** (NEW) - End-to-end tests

## Verification

### Manual Testing

```bash
# Public access (default)
python -m src.app ask "What products does the company offer?"

# Employee access
python -m src.app ask "What is the vacation policy?" --tenant acme-corp --roles employee

# Executive access
python -m src.app ask "What are the executive salaries?" --tenant acme-corp --roles executive

# Multiple roles
python -m src.app ask "company information" --tenant acme-corp --roles employee,contractor

# Hierarchical with RBAC
python -m src.app ask "company policies" --tenant acme-corp --roles employee -H

# Wrong tenant (should filter out results)
python -m src.app ask "company info" --tenant different-corp --roles employee
```

### Automated Tests

```bash
# Run all RBAC tests
pytest tests/test_rbac.py -v

# Run integration tests
pytest tests/test_retrieve.py::TestRBACRetrieval -v

# Run E2E tests
pytest tests/test_e2e_rbac.py -v

# Run full test suite
make test
```

## Backward Compatibility

- **Optional parameter**: `user_context=None` in all functions
- **Secure default**: None context → public-only access
- **Existing tests**: Update to pass explicit UserContext where needed
- **No breaking changes**: All existing API signatures remain valid

## Edge Cases

1. **All results filtered out**: Return empty list (valid scenario, no accessible documents)
2. **Tenant not in index**: Return empty results (no cross-tenant leakage)
3. **Empty allowed_roles**: Deny access (secure default)
4. **Over-fetch exhaustion**: Return < k results (acceptable)
5. **Performance with heavy filtering**: Expand search multiplier handles this (up to k×5)

## Success Criteria

1. ✅ All existing tests pass (with minimal updates)
2. ✅ New RBAC tests achieve >95% code coverage
3. ✅ CLI commands work with and without RBAC arguments
4. ✅ Tenant isolation: 0% cross-tenant leakage
5. ✅ Role-based access: 100% accuracy
6. ✅ Performance: <10% degradation for typical queries
