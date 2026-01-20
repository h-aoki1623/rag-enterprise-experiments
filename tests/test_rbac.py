"""Tests for RBAC (Role-Based Access Control) module."""

import pytest

from src.rag.audit import AuditLogger
from src.rag.config import AuditSettings
from src.rag.models import (
    Chunk,
    ChunkLevel,
    Classification,
    DocumentMetadata,
    HierarchicalChunk,
    HierarchicalRetrievalResult,
    RetrievalResult,
    UserContext,
)
from src.rag.rbac import (
    check_access,
    filter_hierarchical_results,
    filter_retrieval_results,
)


@pytest.fixture
def audit_logger():
    """Create an audit logger for testing."""
    settings = AuditSettings(enabled=True, handler_type="memory")
    logger = AuditLogger(settings)
    yield logger
    # Reset singleton for next test
    AuditLogger._instance = None


@pytest.fixture
def request_id():
    """Create a test request ID."""
    return "test-request-001"


class TestUserContext:
    """Test UserContext model."""

    def test_user_context_creation(self):
        """Test basic UserContext instantiation."""
        ctx = UserContext(tenant_id="test-tenant", user_roles=["employee", "contractor"])
        assert ctx.tenant_id == "test-tenant"
        assert ctx.user_roles == ["employee", "contractor"]
        assert ctx.user_id is None

    def test_user_context_with_user_id(self):
        """Test UserContext with user_id."""
        ctx = UserContext(
            tenant_id="test-tenant", user_roles=["employee"], user_id="user123"
        )
        assert ctx.user_id == "user123"

    def test_user_context_with_empty_roles(self):
        """Test UserContext with no roles (defaults to empty list)."""
        ctx = UserContext(tenant_id="test-tenant")
        assert ctx.tenant_id == "test-tenant"
        assert ctx.user_roles == []

    def test_user_context_validation(self):
        """Test Pydantic validation."""
        # tenant_id is required
        with pytest.raises(Exception):  # Pydantic validation error
            UserContext(user_roles=["employee"])


class TestCheckAccess:
    """Test check_access function."""

    def test_tenant_isolation_same_tenant(self, audit_logger, request_id):
        """Test access granted for same tenant."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="tenant-a",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="tenant-a", user_roles=["public"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is True
        assert reason is None

    def test_tenant_isolation_different_tenant(self, audit_logger, request_id):
        """Test that different tenants cannot access each other's docs."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="tenant-a",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="tenant-b", user_roles=["public"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is False
        assert "tenant_mismatch" in reason

    def test_role_based_access_granted(self, audit_logger, request_id):
        """Test access granted when user has required role."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee", "contractor"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is True
        assert reason is None

    def test_role_based_access_denied(self, audit_logger, request_id):
        """Test access denied when user lacks required role."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is False
        assert "role_mismatch" in reason

    def test_multiple_roles_any_match(self, audit_logger, request_id):
        """Test that any matching role grants access."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee", "contractor"],
            pii_flag=False,
            source="test",
        )
        # User has contractor role which matches
        user_context = UserContext(
            tenant_id="test-tenant", user_roles=["public", "contractor"]
        )

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is True
        assert reason is None

    def test_no_user_context_denies_all(self, audit_logger, request_id):
        """Test None user_context denies access to all documents (strict tenant isolation)."""
        # Even public documents are not accessible without user context
        metadata_public = DocumentMetadata(
            doc_id="doc1",
            tenant_id="any-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        has_access, reason = check_access(
            metadata_public, None, request_id, audit_logger
        )
        assert has_access is False
        assert reason == "no_user_context"

        # Internal documents are also not accessible
        metadata_internal = DocumentMetadata(
            doc_id="doc2",
            tenant_id="any-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )
        has_access, reason = check_access(
            metadata_internal, None, request_id, audit_logger
        )
        assert has_access is False
        assert reason == "no_user_context"

    def test_public_within_tenant(self, audit_logger, request_id):
        """Test that public documents are accessible within the same tenant."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        # User with any role in the same tenant can access public documents
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is True
        assert reason is None

    def test_no_allowed_roles_denies_access(self, audit_logger, request_id):
        """Test secure default for documents with empty allowed_roles."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=[],  # Empty
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])

        has_access, reason = check_access(
            metadata, user_context, request_id, audit_logger
        )
        assert has_access is False
        assert "no_allowed_roles" in reason

    def test_audit_logging_on_access_granted(self, audit_logger, request_id):
        """Test that audit log is created when access is granted."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])

        check_access(metadata, user_context, request_id, audit_logger)

        records = audit_logger.get_memory_records()
        assert len(records) == 1
        assert "access_granted" in records[0]
        assert "doc1" in records[0]

    def test_audit_logging_on_access_denied(self, audit_logger, request_id):
        """Test that audit log is created when access is denied."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="tenant-a",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        user_context = UserContext(tenant_id="tenant-b", user_roles=["public"])

        check_access(metadata, user_context, request_id, audit_logger)

        records = audit_logger.get_memory_records()
        assert len(records) == 1
        assert "access_denied" in records[0]
        assert "tenant_mismatch" in records[0]


class TestFilterRetrievalResults:
    """Test filter_retrieval_results function."""

    def test_filter_preserves_order(self, audit_logger, request_id):
        """Test that filtering maintains rank order."""
        metadata1 = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        metadata2 = DocumentMetadata(
            doc_id="doc2",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )

        results = [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk1", text="text1", doc_id="doc1", metadata=metadata1
                ),
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk2", text="text2", doc_id="doc2", metadata=metadata2
                ),
                score=0.8,
                rank=2,
            ),
        ]

        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])
        filtered = filter_retrieval_results(
            results, user_context, request_id, audit_logger
        )

        assert len(filtered) == 2
        assert filtered[0].chunk.chunk_id == "chunk1"
        assert filtered[1].chunk.chunk_id == "chunk2"

    def test_filter_updates_ranks(self, audit_logger, request_id):
        """Test that ranks are re-numbered after filtering."""
        metadata_public = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        metadata_internal = DocumentMetadata(
            doc_id="doc2",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )

        results = [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk1",
                    text="text1",
                    doc_id="doc1",
                    metadata=metadata_public,
                ),
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk2",
                    text="text2",
                    doc_id="doc2",
                    metadata=metadata_internal,
                ),
                score=0.8,
                rank=2,
            ),
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk3",
                    text="text3",
                    doc_id="doc1",
                    metadata=metadata_public,
                ),
                score=0.7,
                rank=3,
            ),
        ]

        # Public user can only see public docs
        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])
        filtered = filter_retrieval_results(
            results, user_context, request_id, audit_logger
        )

        assert len(filtered) == 2
        assert filtered[0].rank == 1
        assert filtered[0].chunk.chunk_id == "chunk1"
        assert filtered[1].rank == 2
        assert filtered[1].chunk.chunk_id == "chunk3"

    def test_filter_returns_only_allowed(self, audit_logger, request_id):
        """Test that only allowed documents pass through."""
        metadata_employee = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )
        metadata_exec = DocumentMetadata(
            doc_id="doc2",
            tenant_id="test-tenant",
            classification=Classification.CONFIDENTIAL,
            allowed_roles=["executive"],
            pii_flag=False,
            source="test",
        )

        results = [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk1",
                    text="text1",
                    doc_id="doc1",
                    metadata=metadata_employee,
                ),
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk2",
                    text="text2",
                    doc_id="doc2",
                    metadata=metadata_exec,
                ),
                score=0.8,
                rank=2,
            ),
        ]

        # Employee cannot see executive docs
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])
        filtered = filter_retrieval_results(
            results, user_context, request_id, audit_logger
        )

        assert len(filtered) == 1
        assert filtered[0].chunk.chunk_id == "chunk1"

    def test_filter_handles_empty_results(self, audit_logger, request_id):
        """Test filtering empty list."""
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])
        filtered = filter_retrieval_results([], user_context, request_id, audit_logger)
        assert len(filtered) == 0


class TestFilterHierarchicalResults:
    """Test filter_hierarchical_results function."""

    def test_filter_by_parent_metadata(self, audit_logger, request_id):
        """Test that filtering uses parent metadata."""
        parent_metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )

        parent = HierarchicalChunk(
            chunk_id="parent1",
            text="parent text",
            doc_id="doc1",
            metadata=parent_metadata,
            level=ChunkLevel.PARENT,
            section_header="Section 1",
        )

        child = HierarchicalChunk(
            chunk_id="child1",
            text="child text",
            doc_id="doc1",
            metadata=parent_metadata,
            level=ChunkLevel.CHILD,
            parent_id="parent1",
        )

        results = [
            HierarchicalRetrievalResult(
                parent_chunk=parent,
                matched_children=[child],
                child_scores=[0.9],
                aggregate_score=0.9,
                rank=1,
            )
        ]

        # Employee can access
        user_context = UserContext(tenant_id="test-tenant", user_roles=["employee"])
        filtered = filter_hierarchical_results(
            results, user_context, request_id, audit_logger
        )
        assert len(filtered) == 1

        # Reset audit logger for next check
        audit_logger._memory_handler.clear()

        # Public cannot access
        user_context_public = UserContext(tenant_id="test-tenant", user_roles=["public"])
        filtered_public = filter_hierarchical_results(
            results, user_context_public, request_id, audit_logger
        )
        assert len(filtered_public) == 0

    def test_filter_preserves_children(self, audit_logger, request_id):
        """Test that matched children are preserved after parent filter."""
        parent_metadata = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )

        parent = HierarchicalChunk(
            chunk_id="parent1",
            text="parent text",
            doc_id="doc1",
            metadata=parent_metadata,
            level=ChunkLevel.PARENT,
            section_header="Section 1",
        )

        children = [
            HierarchicalChunk(
                chunk_id=f"child{i}",
                text=f"child text {i}",
                doc_id="doc1",
                metadata=parent_metadata,
                level=ChunkLevel.CHILD,
                parent_id="parent1",
            )
            for i in range(3)
        ]

        results = [
            HierarchicalRetrievalResult(
                parent_chunk=parent,
                matched_children=children,
                child_scores=[0.9, 0.8, 0.7],
                aggregate_score=0.9,
                rank=1,
            )
        ]

        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])
        filtered = filter_hierarchical_results(
            results, user_context, request_id, audit_logger
        )

        assert len(filtered) == 1
        assert len(filtered[0].matched_children) == 3
        assert filtered[0].child_scores == [0.9, 0.8, 0.7]

    def test_filter_updates_hierarchical_ranks(self, audit_logger, request_id):
        """Test that hierarchical ranks are updated after filtering."""
        metadata_public = DocumentMetadata(
            doc_id="doc1",
            tenant_id="test-tenant",
            classification=Classification.PUBLIC,
            allowed_roles=["public"],
            pii_flag=False,
            source="test",
        )
        metadata_internal = DocumentMetadata(
            doc_id="doc2",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=False,
            source="test",
        )

        results = [
            HierarchicalRetrievalResult(
                parent_chunk=HierarchicalChunk(
                    chunk_id="parent1",
                    text="text",
                    doc_id="doc1",
                    metadata=metadata_public,
                    level=ChunkLevel.PARENT,
                ),
                matched_children=[],
                child_scores=[],
                aggregate_score=0.9,
                rank=1,
            ),
            HierarchicalRetrievalResult(
                parent_chunk=HierarchicalChunk(
                    chunk_id="parent2",
                    text="text",
                    doc_id="doc2",
                    metadata=metadata_internal,
                    level=ChunkLevel.PARENT,
                ),
                matched_children=[],
                child_scores=[],
                aggregate_score=0.8,
                rank=2,
            ),
            HierarchicalRetrievalResult(
                parent_chunk=HierarchicalChunk(
                    chunk_id="parent3",
                    text="text",
                    doc_id="doc1",
                    metadata=metadata_public,
                    level=ChunkLevel.PARENT,
                ),
                matched_children=[],
                child_scores=[],
                aggregate_score=0.7,
                rank=3,
            ),
        ]

        # Public user filters out internal doc
        user_context = UserContext(tenant_id="test-tenant", user_roles=["public"])
        filtered = filter_hierarchical_results(
            results, user_context, request_id, audit_logger
        )

        assert len(filtered) == 2
        assert filtered[0].rank == 1
        assert filtered[0].parent_chunk.chunk_id == "parent1"
        assert filtered[1].rank == 2
        assert filtered[1].parent_chunk.chunk_id == "parent3"
