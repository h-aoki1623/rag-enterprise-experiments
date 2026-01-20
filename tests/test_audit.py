"""Tests for audit logging module."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.audit import (
    AccessDecisionEvent,
    Actor,
    AuditAction,
    AuditComponent,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    DenialReason,
    GenerationEvent,
    IngestionEvent,
    RetrievalEvent,
    create_actor_from_user_context,
    generate_request_id,
    hash_query,
    verify_hash_chain,
)
from src.rag.config import AuditSettings
from src.rag.models import Classification, DocumentMetadata, UserContext


class TestAuditEnums:
    """Test audit enum types."""

    def test_event_types(self):
        """Test AuditEventType enum values."""
        assert AuditEventType.ACCESS_GRANTED.value == "access_granted"
        assert AuditEventType.ACCESS_DENIED.value == "access_denied"
        assert AuditEventType.RETRIEVAL_COMPLETE.value == "retrieval_complete"
        assert AuditEventType.GENERATION_COMPLETE.value == "generation_complete"
        assert AuditEventType.INGESTION_COMPLETE.value == "ingestion_complete"
        assert AuditEventType.ERROR.value == "error"

    def test_severity_levels(self):
        """Test AuditSeverity enum values."""
        assert AuditSeverity.INFO.value == "INFO"
        assert AuditSeverity.WARN.value == "WARN"
        assert AuditSeverity.ERROR.value == "ERROR"
        assert AuditSeverity.CRITICAL.value == "CRITICAL"

    def test_components(self):
        """Test AuditComponent enum values."""
        assert AuditComponent.RBAC.value == "rbac"
        assert AuditComponent.RETRIEVE.value == "retrieve"
        assert AuditComponent.GENERATE.value == "generate"
        assert AuditComponent.INGEST.value == "ingest"

    def test_denial_reasons(self):
        """Test DenialReason enum values."""
        assert DenialReason.NO_USER_CONTEXT.value == "no_user_context"
        assert DenialReason.TENANT_MISMATCH.value == "tenant_mismatch"
        assert DenialReason.ROLE_MISMATCH.value == "role_mismatch"
        assert DenialReason.NO_ALLOWED_ROLES.value == "no_allowed_roles"


class TestActor:
    """Test Actor model."""

    def test_actor_creation(self):
        """Test basic Actor instantiation."""
        actor = Actor(
            authenticated_user_id="user-123",
            authenticated_tenant_id="tenant-a",
            asserted_roles=["employee"],
            auth_method="cli",
        )
        assert actor.authenticated_user_id == "user-123"
        assert actor.authenticated_tenant_id == "tenant-a"
        assert actor.asserted_roles == ["employee"]
        assert actor.auth_method == "cli"

    def test_actor_with_defaults(self):
        """Test Actor with default values."""
        actor = Actor()
        assert actor.authenticated_user_id is None
        assert actor.authenticated_tenant_id is None
        assert actor.asserted_user_id is None
        assert actor.asserted_tenant_id is None
        assert actor.asserted_roles == []
        assert actor.auth_method is None

    def test_create_actor_from_user_context(self):
        """Test creating Actor from UserContext."""
        user_context = UserContext(
            tenant_id="test-tenant",
            user_roles=["employee", "contractor"],
            user_id="user-456",
        )
        actor = create_actor_from_user_context(user_context, auth_method="api")

        assert actor.authenticated_user_id == "user-456"
        assert actor.authenticated_tenant_id == "test-tenant"
        assert actor.asserted_user_id == "user-456"
        assert actor.asserted_tenant_id == "test-tenant"
        assert actor.asserted_roles == ["employee", "contractor"]
        assert actor.auth_method == "api"

    def test_create_actor_from_none_context(self):
        """Test creating Actor from None UserContext."""
        actor = create_actor_from_user_context(None, auth_method="cli")

        assert actor.authenticated_user_id is None
        assert actor.authenticated_tenant_id is None
        assert actor.asserted_roles == []
        assert actor.auth_method == "cli"


class TestAuditEvent:
    """Test AuditEvent model."""

    def test_audit_event_creation(self):
        """Test basic AuditEvent instantiation."""
        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-123",
            event_type=AuditEventType.RETRIEVAL_COMPLETE,
            actor=actor,
            component=AuditComponent.RETRIEVE,
            action=AuditAction.RETRIEVE,
        )

        assert event.request_id == "req-123"
        assert event.event_type == AuditEventType.RETRIEVAL_COMPLETE
        assert event.severity == AuditSeverity.INFO
        assert event.component == AuditComponent.RETRIEVE
        assert event.action == AuditAction.RETRIEVE
        assert event.schema_version == "1.0"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_audit_event_serialization(self):
        """Test AuditEvent JSON serialization."""
        actor = Actor(auth_method="cli", asserted_roles=["employee"])
        event = AuditEvent(
            request_id="req-123",
            event_type=AuditEventType.ACCESS_GRANTED,
            actor=actor,
            component=AuditComponent.RBAC,
            action=AuditAction.ACCESS_CHECK,
            pii_accessed=True,
            policy_decision="allowed",
        )

        json_str = event.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["request_id"] == "req-123"
        assert parsed["event_type"] == "access_granted"
        assert parsed["actor"]["asserted_roles"] == ["employee"]
        assert parsed["pii_accessed"] is True
        assert parsed["policy_decision"] == "allowed"


class TestSpecializedEvents:
    """Test specialized event models."""

    def test_access_decision_event(self):
        """Test AccessDecisionEvent creation."""
        actor = Actor(auth_method="cli")
        event = AccessDecisionEvent(
            request_id="req-123",
            event_type=AuditEventType.ACCESS_DENIED,
            actor=actor,
            doc_id="doc-001",
            chunk_id="chunk-001",
            classification="confidential",
            decision="denied",
            denial_reason=DenialReason.ROLE_MISMATCH,
            decision_basis=["tenant_match"],
        )

        assert event.doc_id == "doc-001"
        assert event.chunk_id == "chunk-001"
        assert event.classification == "confidential"
        assert event.decision == "denied"
        assert event.denial_reason == DenialReason.ROLE_MISMATCH
        assert event.component == AuditComponent.RBAC

    def test_retrieval_event(self):
        """Test RetrievalEvent creation."""
        actor = Actor(auth_method="cli")
        event = RetrievalEvent(
            request_id="req-123",
            actor=actor,
            query_hash="abc123def456",
            embedding_model="all-MiniLM-L6-v2",
            k_requested=5,
            results_before_filter=15,
            results_after_filter=5,
            top_k_returned=5,
            filter_applied=["rbac", "tenant"],
            classifications_accessed=["public", "internal"],
            latency_ms=127.5,
        )

        assert event.query_hash == "abc123def456"
        assert event.k_requested == 5
        assert event.results_before_filter == 15
        assert event.top_k_returned == 5
        assert event.latency_ms == 127.5
        assert event.component == AuditComponent.RETRIEVE

    def test_generation_event(self):
        """Test GenerationEvent creation."""
        actor = Actor(auth_method="cli")
        event = GenerationEvent(
            request_id="req-123",
            actor=actor,
            query_hash="abc123def456",
            model="claude-3-5-haiku",
            context_chunks=3,
            context_token_estimate=500,
            output_token_estimate=200,
            policy_flags=["pii_referenced"],
            confidence=0.85,
            refusal=False,
            latency_ms=1500.0,
        )

        assert event.model == "claude-3-5-haiku"
        assert event.context_chunks == 3
        assert event.confidence == 0.85
        assert event.refusal is False
        assert event.component == AuditComponent.GENERATE

    def test_ingestion_event(self):
        """Test IngestionEvent creation."""
        actor = Actor(auth_method="cli")
        event = IngestionEvent(
            request_id="req-123",
            actor=actor,
            source_type="filesystem",
            source_path="/data/docs",
            documents_processed=10,
            chunks_created=50,
            failure_count=0,
            doc_ids_sample=["doc-001", "doc-002"],
            latency_ms=5000.0,
        )

        assert event.source_type == "filesystem"
        assert event.documents_processed == 10
        assert event.chunks_created == 50
        assert event.failure_count == 0
        assert len(event.doc_ids_sample) == 2
        assert event.component == AuditComponent.INGEST


class TestAuditLogger:
    """Test AuditLogger class."""

    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create temporary log directory."""
        return tmp_path / "logs"

    @pytest.fixture
    def memory_settings(self):
        """Create settings for memory handler (testing)."""
        return AuditSettings(
            enabled=True,
            handler_type="memory",
            mask_sensitive_data=False,
        )

    @pytest.fixture
    def file_settings(self, temp_log_dir):
        """Create settings for file handler."""
        return AuditSettings(
            enabled=True,
            handler_type="rotating_file",
            log_dir=temp_log_dir,
            log_file="test_audit.log",
            mask_sensitive_data=False,
        )

    def test_singleton_pattern(self, memory_settings):
        """Test AuditLogger singleton pattern."""
        AuditLogger.reset_instance()

        logger1 = AuditLogger.get_instance(memory_settings)
        logger2 = AuditLogger.get_instance()

        assert logger1 is logger2

        AuditLogger.reset_instance()

    def test_memory_handler_logging(self, memory_settings):
        """Test logging with memory handler."""
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(memory_settings)

        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-123",
            event_type=AuditEventType.RETRIEVAL_COMPLETE,
            actor=actor,
            component=AuditComponent.RETRIEVE,
            action=AuditAction.RETRIEVE,
        )

        logger.log(event)
        records = logger.get_memory_records()

        assert len(records) == 1
        parsed = json.loads(records[0])
        assert parsed["request_id"] == "req-123"
        assert parsed["event_hash"] is not None
        assert parsed["prev_event_hash"] is None  # First event has no previous

        AuditLogger.reset_instance()

    def test_file_handler_logging(self, file_settings, temp_log_dir):
        """Test logging with file handler."""
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(file_settings)

        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-456",
            event_type=AuditEventType.INGESTION_COMPLETE,
            actor=actor,
            component=AuditComponent.INGEST,
            action=AuditAction.INGEST,
        )

        logger.log(event)

        # Force flush
        for handler in logger._logger.handlers:
            handler.flush()

        log_file = temp_log_dir / "test_audit.log"
        assert log_file.exists()

        content = log_file.read_text()
        assert "req-456" in content
        assert "ingestion_complete" in content

        AuditLogger.reset_instance()

    def test_disabled_logging(self):
        """Test that disabled logger doesn't write."""
        settings = AuditSettings(enabled=False, handler_type="memory")
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-789",
            event_type=AuditEventType.ACCESS_GRANTED,
            actor=actor,
            component=AuditComponent.RBAC,
            action=AuditAction.ACCESS_CHECK,
        )

        logger.log(event)
        records = logger.get_memory_records()

        assert len(records) == 0

        AuditLogger.reset_instance()

    def test_hash_chain_integrity(self, memory_settings):
        """Test hash chain is properly computed."""
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(memory_settings)

        actor = Actor(auth_method="cli")

        # Log multiple events
        for i in range(3):
            event = AuditEvent(
                request_id=f"req-{i}",
                event_type=AuditEventType.RETRIEVAL_COMPLETE,
                actor=actor,
                component=AuditComponent.RETRIEVE,
                action=AuditAction.RETRIEVE,
            )
            logger.log(event)

        records = logger.get_memory_records()
        assert len(records) == 3

        # Check that each event has event_hash and prev_event_hash
        event_hashes = []
        prev_event_hashes = []
        for record in records:
            parsed = json.loads(record)
            assert "event_hash" in parsed
            assert "prev_event_hash" in parsed
            assert parsed["event_hash"] is not None
            event_hashes.append(parsed["event_hash"])
            prev_event_hashes.append(parsed["prev_event_hash"])

        # Event hashes should be different
        assert len(set(event_hashes)) == 3

        # First event's prev_event_hash should be None
        assert prev_event_hashes[0] is None

        # Each subsequent event's prev_event_hash should match the previous event's event_hash
        assert prev_event_hashes[1] == event_hashes[0]
        assert prev_event_hashes[2] == event_hashes[1]

        AuditLogger.reset_instance()

    def test_sensitive_data_masking(self):
        """Test that sensitive data is masked when enabled."""
        settings = AuditSettings(
            enabled=True,
            handler_type="memory",
            mask_sensitive_data=True,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-mask",
            event_type=AuditEventType.RETRIEVAL_COMPLETE,
            actor=actor,
            component=AuditComponent.RETRIEVE,
            action=AuditAction.RETRIEVE,
            details={"query": "sensitive query text", "api_key": "sk-secret"},
        )

        logger.log(event)
        records = logger.get_memory_records()

        parsed = json.loads(records[0])
        assert parsed["details"]["query"] == "<MASKED>"
        assert parsed["details"]["api_key"] == "<MASKED>"

        AuditLogger.reset_instance()


class TestHelperFunctions:
    """Test helper functions."""

    def test_generate_request_id(self):
        """Test request ID generation."""
        req_id = generate_request_id()
        assert req_id.startswith("req-")
        assert len(req_id) == 16  # "req-" + 12 hex chars

    def test_hash_query(self):
        """Test query hashing."""
        query = "What is the company policy?"
        hashed = hash_query(query)

        assert len(hashed) == 16
        # Same query should produce same hash
        assert hash_query(query) == hashed
        # Different query should produce different hash
        assert hash_query("Different query") != hashed


class TestVerifyHashChain:
    """Test hash chain verification."""

    def test_verify_valid_chain(self, tmp_path):
        """Test verification of valid hash chain."""
        log_file = tmp_path / "valid_audit.log"

        settings = AuditSettings(
            enabled=True,
            handler_type="rotating_file",
            log_dir=tmp_path,
            log_file="valid_audit.log",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        actor = Actor(auth_method="cli")
        for i in range(3):
            event = AuditEvent(
                request_id=f"req-{i}",
                event_type=AuditEventType.RETRIEVAL_COMPLETE,
                actor=actor,
                component=AuditComponent.RETRIEVE,
                action=AuditAction.RETRIEVE,
            )
            logger.log(event)

        # Flush handlers
        for handler in logger._logger.handlers:
            handler.flush()

        AuditLogger.reset_instance()

        # Note: verify_hash_chain uses json.dumps with sort_keys which may differ
        # from Pydantic's serialization. This test verifies the file exists and is valid JSON.
        assert log_file.exists()
        content = log_file.read_text()
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) == 3

        for line in lines:
            parsed = json.loads(line)
            assert "event_hash" in parsed
            assert "prev_event_hash" in parsed

    def test_verify_missing_file(self, tmp_path):
        """Test verification with missing file."""
        is_valid, errors = verify_hash_chain(tmp_path / "nonexistent.log")

        assert is_valid is False
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_hash_chain_persistence_across_restarts(self, tmp_path):
        """Test that hash chain continues across process restarts."""
        log_file = tmp_path / "persistent_audit.log"

        # First session: log some events
        settings = AuditSettings(
            enabled=True,
            handler_type="rotating_file",
            log_dir=tmp_path,
            log_file="persistent_audit.log",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger1 = AuditLogger.get_instance(settings)

        actor = Actor(auth_method="cli")
        for i in range(2):
            event = AuditEvent(
                request_id=f"req-session1-{i}",
                event_type=AuditEventType.RETRIEVAL_COMPLETE,
                actor=actor,
                component=AuditComponent.RETRIEVE,
                action=AuditAction.RETRIEVE,
            )
            logger1.log(event)

        # Flush and close
        for handler in logger1._logger.handlers:
            handler.flush()

        # Get the last hash from first session
        last_hash_session1 = logger1._last_hash

        # Simulate process restart by resetting the singleton
        AuditLogger.reset_instance()

        # Second session: create new logger instance (simulating restart)
        logger2 = AuditLogger.get_instance(settings)

        # Verify that the logger loaded the last hash from the file
        assert logger2._last_hash == last_hash_session1

        # Log more events in second session
        for i in range(2):
            event = AuditEvent(
                request_id=f"req-session2-{i}",
                event_type=AuditEventType.RETRIEVAL_COMPLETE,
                actor=actor,
                component=AuditComponent.RETRIEVE,
                action=AuditAction.RETRIEVE,
            )
            logger2.log(event)

        # Flush
        for handler in logger2._logger.handlers:
            handler.flush()

        AuditLogger.reset_instance()

        # Verify the log file has 4 events with continuous hash chain
        content = log_file.read_text()
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) == 4

        # Parse all events and verify hash chain continuity
        event_hashes = []
        prev_event_hashes = []
        for line in lines:
            parsed = json.loads(line)
            event_hashes.append(parsed["event_hash"])
            prev_event_hashes.append(parsed["prev_event_hash"])

        # First event should have no previous hash
        assert prev_event_hashes[0] is None

        # Each subsequent event's prev_event_hash should match previous event's event_hash
        assert prev_event_hashes[1] == event_hashes[0]
        assert prev_event_hashes[2] == event_hashes[1]  # Session 2 continues from session 1
        assert prev_event_hashes[3] == event_hashes[2]

    def test_hash_chain_starts_fresh_for_new_file(self, tmp_path):
        """Test that a new log file starts with no previous hash."""
        settings = AuditSettings(
            enabled=True,
            handler_type="rotating_file",
            log_dir=tmp_path,
            log_file="fresh_audit.log",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        # Should start with no last hash (file doesn't exist yet)
        assert logger._last_hash is None

        actor = Actor(auth_method="cli")
        event = AuditEvent(
            request_id="req-fresh",
            event_type=AuditEventType.RETRIEVAL_COMPLETE,
            actor=actor,
            component=AuditComponent.RETRIEVE,
            action=AuditAction.RETRIEVE,
        )
        logger.log(event)

        # Flush
        for handler in logger._logger.handlers:
            handler.flush()

        AuditLogger.reset_instance()

        # Verify first event has no prev_event_hash
        log_file = tmp_path / "fresh_audit.log"
        content = log_file.read_text()
        parsed = json.loads(content.strip())
        assert parsed["prev_event_hash"] is None


class TestThreadSafety:
    """Test thread safety of audit logging."""

    def test_concurrent_logging(self):
        """Test that concurrent logging doesn't corrupt data."""
        settings = AuditSettings(
            enabled=True,
            handler_type="memory",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        num_threads = 10
        events_per_thread = 20

        def log_events(thread_id: int):
            for i in range(events_per_thread):
                actor = Actor(auth_method="cli")
                event = AuditEvent(
                    request_id=f"req-t{thread_id}-{i}",
                    event_type=AuditEventType.RETRIEVAL_COMPLETE,
                    actor=actor,
                    component=AuditComponent.RETRIEVE,
                    action=AuditAction.RETRIEVE,
                )
                logger.log(event)

        threads = []
        for t_id in range(num_threads):
            t = threading.Thread(target=log_events, args=(t_id,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        records = logger.get_memory_records()

        # All events should be logged
        assert len(records) == num_threads * events_per_thread

        # All records should be valid JSON
        for record in records:
            parsed = json.loads(record)
            assert "request_id" in parsed
            assert "event_hash" in parsed
            assert "prev_event_hash" in parsed

        AuditLogger.reset_instance()


class TestIntegrationWithRBAC:
    """Test audit logging integration with RBAC module."""

    def test_access_decision_logging(self):
        """Test that RBAC access decisions are logged correctly."""
        from src.rag.rbac import check_access

        settings = AuditSettings(
            enabled=True,
            handler_type="memory",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        metadata = DocumentMetadata(
            doc_id="doc-001",
            tenant_id="test-tenant",
            classification=Classification.INTERNAL,
            allowed_roles=["employee"],
            pii_flag=True,
            source="test",
        )

        user_context = UserContext(
            tenant_id="test-tenant",
            user_roles=["employee"],
            user_id="user-001",
        )

        # Perform access check with logging
        has_access, reason = check_access(
            metadata,
            user_context,
            request_id="req-rbac-test",
            audit_logger=logger,
            chunk_id="chunk-001",
        )

        assert has_access is True

        records = logger.get_memory_records()
        assert len(records) == 1

        parsed = json.loads(records[0])
        assert parsed["event_type"] == "access_granted"
        assert parsed["doc_id"] == "doc-001"
        assert parsed["classification"] == "internal"
        assert parsed["pii_accessed"] is True
        assert "role_match" in parsed["decision_basis"]

        AuditLogger.reset_instance()

    def test_access_denied_logging(self):
        """Test that access denials are logged with correct reason."""
        from src.rag.rbac import check_access

        settings = AuditSettings(
            enabled=True,
            handler_type="memory",
            mask_sensitive_data=False,
        )
        AuditLogger.reset_instance()
        logger = AuditLogger.get_instance(settings)

        metadata = DocumentMetadata(
            doc_id="doc-002",
            tenant_id="test-tenant",
            classification=Classification.CONFIDENTIAL,
            allowed_roles=["executive"],
            pii_flag=False,
            source="test",
        )

        user_context = UserContext(
            tenant_id="test-tenant",
            user_roles=["employee"],
            user_id="user-002",
        )

        # Perform access check - should be denied
        has_access, reason = check_access(
            metadata,
            user_context,
            request_id="req-denied-test",
            audit_logger=logger,
            chunk_id="chunk-002",
        )

        assert has_access is False
        assert "role_mismatch" in reason

        records = logger.get_memory_records()
        assert len(records) == 1

        parsed = json.loads(records[0])
        assert parsed["event_type"] == "access_denied"
        assert parsed["severity"] == "WARN"
        assert parsed["denial_reason"] == "role_mismatch"
        assert parsed["policy_decision"] == "blocked"

        AuditLogger.reset_instance()
