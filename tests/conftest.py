"""Pytest fixtures for RAG system tests."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.rag.config import GuardrailSettings
from src.rag.models import Classification, DocumentMetadata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_docs_dir(temp_dir):
    """Create sample documents in a temporary directory."""
    docs_dir = temp_dir / "docs"

    # Create classification directories
    (docs_dir / "public").mkdir(parents=True)
    (docs_dir / "internal").mkdir(parents=True)
    (docs_dir / "confidential").mkdir(parents=True)

    # Public document
    public_doc = docs_dir / "public" / "test-public.md"
    public_doc.write_text(
        "# Public Document\n\n"
        "This is a public document about our company products.\n\n"
        "## Products\n\n"
        "We offer CloudSync and SecureVault."
    )
    public_meta = docs_dir / "public" / "test-public.meta.json"
    public_meta.write_text(json.dumps({
        "doc_id": "test-public-001",
        "classification": "public",
        "allowed_roles": ["employee", "public"],
        "pii_flag": False,
        "source": "test"
    }))

    # Internal document
    internal_doc = docs_dir / "internal" / "test-internal.md"
    internal_doc.write_text(
        "# Internal Policy\n\n"
        "This is an internal document about employee policies.\n\n"
        "## Vacation Policy\n\n"
        "Employees receive 15 days of paid vacation per year.\n\n"
        "## Sick Leave\n\n"
        "Employees receive 10 days of sick leave per year."
    )
    internal_meta = docs_dir / "internal" / "test-internal.meta.json"
    internal_meta.write_text(json.dumps({
        "doc_id": "test-internal-001",
        "classification": "internal",
        "allowed_roles": ["employee"],
        "pii_flag": False,
        "source": "test"
    }))

    # Confidential document
    confidential_doc = docs_dir / "confidential" / "test-confidential.md"
    confidential_doc.write_text(
        "# Confidential Report\n\n"
        "This document contains sensitive executive information.\n\n"
        "## Executive Salaries\n\n"
        "CEO salary: $500,000\n"
        "CTO salary: $400,000"
    )
    confidential_meta = docs_dir / "confidential" / "test-confidential.meta.json"
    confidential_meta.write_text(json.dumps({
        "doc_id": "test-confidential-001",
        "classification": "confidential",
        "allowed_roles": ["executive"],
        "pii_flag": True,
        "source": "test"
    }))

    yield docs_dir


@pytest.fixture
def sample_index_dir(temp_dir):
    """Create a temporary directory for index files."""
    index_dir = temp_dir / "indexes"
    index_dir.mkdir(parents=True)
    yield index_dir


@pytest.fixture
def sample_metadata():
    """Create sample document metadata."""
    return DocumentMetadata(
        doc_id="test-doc-001",
        classification=Classification.INTERNAL,
        allowed_roles=["employee"],
        pii_flag=False,
        source="test"
    )


@pytest.fixture
def long_document_text():
    """Create a long document for chunking tests."""
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"## Section {i + 1}\n\n"
            f"This is paragraph {i + 1} of the document. "
            f"It contains important information about topic {i + 1}. "
            f"The content here is meant to test chunking behavior. "
            f"We need enough text to ensure multiple chunks are created.\n"
        )
    return "\n".join(paragraphs)


@pytest.fixture
def guardrail_settings():
    """Create GuardrailSettings for testing.

    Uses default settings which are suitable for most tests.
    Individual tests can modify the returned object if needed.
    """
    return GuardrailSettings(
        input_guardrail_enabled=True,
        output_guardrail_enabled=True,
        log_guardrail_events=False,  # Disable logging in tests
        max_query_length=2000,
        ngram_size=5,
        # Input guardrail thresholds (individual action thresholds)
        injection_allow_threshold=0.25,
        injection_warn_threshold=0.40,
        injection_block_threshold=0.50,
        # Output guardrail thresholds (classification-based, individual action thresholds)
        leakage_thresholds={
            "public": {"allow": 0.40, "warn": 0.64, "block": 0.80},
            "internal": {"allow": 0.30, "warn": 0.48, "block": 0.60},
            "confidential": {"allow": 0.20, "warn": 0.32, "block": 0.40},
        },
        default_leakage_allow_threshold=0.30,
        default_leakage_warn_threshold=0.48,
        default_leakage_block_threshold=0.60,
    )
