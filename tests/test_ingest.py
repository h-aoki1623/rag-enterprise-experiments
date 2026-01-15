"""Tests for document ingestion pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.ingest import (
    build_index,
    chunk_documents,
    embed_chunks,
    ingest_all,
    load_documents,
)
from src.rag.models import Chunk, Classification, Document, DocumentMetadata


class TestLoadDocuments:
    """Tests for load_documents function."""

    def test_load_documents_returns_correct_count(self, sample_docs_dir):
        """Test that all documents with metadata are loaded."""
        documents = load_documents(sample_docs_dir)
        assert len(documents) == 3

    def test_load_documents_parses_metadata(self, sample_docs_dir):
        """Test that document metadata is correctly parsed."""
        documents = load_documents(sample_docs_dir)

        # Find the public document
        public_docs = [d for d in documents if d.metadata.classification == Classification.PUBLIC]
        assert len(public_docs) == 1
        assert public_docs[0].metadata.doc_id == "test-public-001"
        assert public_docs[0].metadata.tenant_id == "test-tenant"
        assert "public" in public_docs[0].metadata.allowed_roles

    def test_load_documents_reads_content(self, sample_docs_dir):
        """Test that document content is correctly loaded."""
        documents = load_documents(sample_docs_dir)

        # Find the internal document
        internal_docs = [d for d in documents if d.metadata.classification == Classification.INTERNAL]
        assert len(internal_docs) == 1
        assert "Vacation Policy" in internal_docs[0].content
        assert "15 days" in internal_docs[0].content

    def test_load_documents_sets_file_path(self, sample_docs_dir):
        """Test that file path is stored in document."""
        documents = load_documents(sample_docs_dir)
        for doc in documents:
            assert doc.file_path is not None
            assert doc.file_path.endswith(".md")

    def test_load_documents_skips_without_metadata(self, sample_docs_dir):
        """Test that documents without metadata files are skipped."""
        # Create a document without metadata
        orphan_doc = sample_docs_dir / "public" / "orphan.md"
        orphan_doc.write_text("# Orphan Document\n\nNo metadata file.")

        documents = load_documents(sample_docs_dir)
        # Should still be 3 (original documents only)
        assert len(documents) == 3
        assert not any("orphan" in (d.file_path or "") for d in documents)

    def test_load_documents_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        documents = load_documents(empty_dir)
        assert len(documents) == 0

    def test_load_documents_handles_pii_flag(self, sample_docs_dir):
        """Test that PII flag is correctly parsed."""
        documents = load_documents(sample_docs_dir)

        confidential_docs = [
            d for d in documents if d.metadata.classification == Classification.CONFIDENTIAL
        ]
        assert len(confidential_docs) == 1
        assert confidential_docs[0].metadata.pii_flag is True

        public_docs = [d for d in documents if d.metadata.classification == Classification.PUBLIC]
        assert public_docs[0].metadata.pii_flag is False


class TestChunkDocuments:
    """Tests for chunk_documents function."""

    def test_chunk_documents_creates_chunks(self, sample_docs_dir):
        """Test that documents are split into chunks."""
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_inherits_metadata(self, sample_docs_dir):
        """Test that chunks inherit document metadata."""
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        for chunk in chunks:
            # Each chunk should have valid metadata
            assert chunk.metadata is not None
            assert chunk.metadata.doc_id is not None
            assert chunk.metadata.tenant_id == "test-tenant"
            assert chunk.doc_id == chunk.metadata.doc_id

    def test_chunk_id_format(self, sample_docs_dir):
        """Test that chunk IDs follow expected format."""
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        for chunk in chunks:
            # Chunk ID should be: doc_id-chunk-XXX
            assert chunk.chunk_id.startswith(chunk.doc_id)
            assert "-chunk-" in chunk.chunk_id

    def test_chunk_contains_text(self, sample_docs_dir):
        """Test that chunks contain non-empty text."""
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_long_document_creates_multiple_chunks(self, sample_metadata, long_document_text):
        """Test that long documents are split into multiple chunks."""
        document = Document(
            content=long_document_text,
            metadata=sample_metadata,
            file_path="/test/long-doc.md"
        )

        chunks = chunk_documents([document])

        # Long document should create multiple chunks
        assert len(chunks) > 1

    def test_chunk_text_not_too_long(self, sample_metadata, long_document_text):
        """Test that chunks respect size limits."""
        from src.rag.config import settings

        document = Document(
            content=long_document_text,
            metadata=sample_metadata,
            file_path="/test/long-doc.md"
        )

        chunks = chunk_documents([document])

        # Allow some tolerance for chunk size (overlap can cause slight overflow)
        max_size = settings.chunk_size + settings.chunk_overlap
        for chunk in chunks:
            assert len(chunk.text) <= max_size * 1.5  # 50% tolerance


class TestEmbedChunks:
    """Tests for embed_chunks function."""

    def test_embed_chunks_returns_numpy_array(self, sample_docs_dir):
        """Test that embeddings are returned as numpy array."""
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        embeddings = embed_chunks(chunks)

        assert isinstance(embeddings, np.ndarray)

    def test_embed_chunks_correct_shape(self, sample_docs_dir):
        """Test that embeddings have correct shape."""
        from src.rag.config import settings

        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        embeddings = embed_chunks(chunks)

        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == settings.embedding_dimension

    def test_embed_chunks_with_provided_model(self, sample_docs_dir):
        """Test embedding with pre-loaded model."""
        from sentence_transformers import SentenceTransformer
        from src.rag.config import settings

        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)

        model = SentenceTransformer(settings.embedding_model)
        embeddings = embed_chunks(chunks, model=model)

        assert embeddings.shape[0] == len(chunks)

    def test_embed_chunks_different_texts_different_embeddings(self, sample_metadata):
        """Test that different texts produce different embeddings."""
        chunks = [
            Chunk(
                chunk_id="test-1",
                text="The cat sat on the mat.",
                doc_id="test",
                metadata=sample_metadata
            ),
            Chunk(
                chunk_id="test-2",
                text="Quantum physics is complex.",
                doc_id="test",
                metadata=sample_metadata
            ),
        ]

        embeddings = embed_chunks(chunks)

        # Different texts should have different embeddings
        similarity = np.dot(embeddings[0], embeddings[1])
        assert similarity < 0.9  # Should not be too similar


class TestBuildIndex:
    """Tests for build_index function."""

    def test_build_index_creates_files(self, sample_docs_dir, sample_index_dir):
        """Test that index files are created."""
        from src.rag.config import settings

        # Temporarily override settings
        original_index_dir = settings.index_dir
        settings.index_dir = sample_index_dir

        try:
            documents = load_documents(sample_docs_dir)
            chunks = chunk_documents(documents)
            embeddings = embed_chunks(chunks)

            build_index(embeddings, chunks)

            assert settings.faiss_index_path.exists()
            assert settings.docstore_path.exists()
        finally:
            settings.index_dir = original_index_dir

    def test_build_index_docstore_content(self, sample_docs_dir, sample_index_dir):
        """Test that docstore contains correct data."""
        from src.rag.config import settings

        original_index_dir = settings.index_dir
        settings.index_dir = sample_index_dir

        try:
            documents = load_documents(sample_docs_dir)
            chunks = chunk_documents(documents)
            embeddings = embed_chunks(chunks)

            build_index(embeddings, chunks)

            with open(settings.docstore_path) as f:
                docstore = json.load(f)

            assert "chunks" in docstore
            assert "metadata" in docstore
            assert len(docstore["chunks"]) == len(chunks)
            assert docstore["metadata"]["total_chunks"] == len(chunks)
        finally:
            settings.index_dir = original_index_dir


class TestIngestAll:
    """Tests for ingest_all function."""

    def test_ingest_all_returns_stats(self, sample_docs_dir, sample_index_dir):
        """Test that ingest_all returns statistics."""
        from src.rag.config import settings

        original_index_dir = settings.index_dir
        settings.index_dir = sample_index_dir

        try:
            stats = ingest_all(sample_docs_dir)

            assert "documents" in stats
            assert "chunks" in stats
            assert stats["documents"] == 3
            assert stats["chunks"] > 0
        finally:
            settings.index_dir = original_index_dir

    def test_ingest_all_empty_directory(self, temp_dir, sample_index_dir):
        """Test ingest_all with empty directory."""
        from src.rag.config import settings

        original_index_dir = settings.index_dir
        settings.index_dir = sample_index_dir

        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        try:
            stats = ingest_all(empty_dir)

            assert stats["documents"] == 0
            assert stats["chunks"] == 0
        finally:
            settings.index_dir = original_index_dir
