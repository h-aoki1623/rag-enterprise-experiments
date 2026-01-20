# Plan: Remove Multi-Tenancy Support (tenant_id)

## Overview

Remove the `tenant_id` multi-tenancy feature from the enterprise RAG system to simplify toward a single-organization RBAC design.

## Features to Retain

- `classification` (public/internal/confidential) access control
- `allowed_roles` role-based access control
- Audit logging functionality (only tenant-related fields removed)

## Features to Remove

- `tenant_id` field (models, metadata)
- Tenant isolation logic (RBAC Rule 1)
- CLI `--tenant` argument
- Tenant-related test cases

---

## Implementation Steps

### Step 1: Modify Pydantic Models

**File:** [src/rag/models.py](src/rag/models.py)

- [ ] Remove `tenant_id` field from `UserContext` (line 36)
- [ ] Remove `tenant_id` field from `DocumentMetadata` (line 49)

### Step 2: Modify RBAC Logic

**File:** [src/rag/rbac.py](src/rag/rbac.py)

- [ ] Remove `DenialReason.TENANT_MISMATCH`
- [ ] Remove tenant_mismatch handling from `_map_denial_reason` (lines 28-29)
- [ ] Remove Rule 1 (tenant isolation) from `check_access` (lines 66-68)

### Step 3: Modify Audit Logging

**File:** [src/rag/audit.py](src/rag/audit.py)

- [ ] Remove `authenticated_tenant_id` from `Actor` model (line 98)
- [ ] Remove `asserted_tenant_id` from `Actor` model (line 102)
- [ ] Remove tenant_id processing from `create_actor_from_user_context` (lines 511, 513)

### Step 4: Modify CLI

**File:** [src/app.py](src/app.py)

- [ ] Remove `--tenant` argument from `search` command
- [ ] Remove `--tenant` argument from `ask` command
- [ ] Remove tenant_id from UserContext creation
- [ ] Remove tenant display from output messages

### Step 5: Modify Tests

**Files:**
- [tests/conftest.py](tests/conftest.py)
- [tests/test_rbac.py](tests/test_rbac.py)
- [tests/test_ingest.py](tests/test_ingest.py)
- [tests/test_retrieve.py](tests/test_retrieve.py)
- [tests/test_generate.py](tests/test_generate.py)
- [tests/test_audit.py](tests/test_audit.py)

- [ ] Remove `tenant_id` from fixtures
- [ ] Remove tenant isolation tests (`test_tenant_isolation_*`, `test_cross_tenant_*`, etc.)
- [ ] Remove `tenant_id` references from remaining tests
- [ ] Update UserContext creation to use only `user_roles`

### Step 6: Modify Sample Data

**Files:** `data/docs/**/*.meta.json` (6 files)

- [ ] public/company-overview.meta.json
- [ ] public/product-faq.meta.json
- [ ] internal/employee-handbook.meta.json
- [ ] internal/it-security-policy.meta.json
- [ ] confidential/executive-compensation.meta.json
- [ ] confidential/acquisition-plans.meta.json

Remove `"tenant_id": "acme-corp"` line from each file.

### Step 7: Rebuild Index

- [ ] Delete existing indexes/
- [ ] Run `python -m src.app ingest` to re-index

### Step 8: Update Documentation

**File:** [CLAUDE.md](CLAUDE.md)

- [ ] Remove tenant_id from Document Metadata Format section
- [ ] Update related descriptions as necessary

---

## Verification Steps

1. **Run Tests**
   ```bash
   make test
   ```
   Verify all tests pass.

2. **Run Linter**
   ```bash
   make lint
   ```
   Verify no errors.

3. **Functional Verification**
   ```bash
   # Rebuild index
   python -m src.app ingest

   # Test search
   python -m src.app search "company overview"

   # Test RAG (should work without --tenant)
   python -m src.app ask "What is the company about?" --roles employee
   ```

---

## Impact Summary

| Category | File Count | Main Changes |
|----------|------------|--------------|
| Models | 1 | Remove 2 fields |
| Logic | 2 | Remove tenant isolation rule |
| CLI | 1 | Remove --tenant argument |
| Tests | 6 | Remove/modify tenant-related tests |
| Data | 6 | Remove tenant_id from metadata |
| Documentation | 1 | Update CLAUDE.md |

**Total:** 17 files to modify
