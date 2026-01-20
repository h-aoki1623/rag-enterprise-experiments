"""Tests for retrieval layer."""


import numpy as np
import pytest

from src.rag.ingest import build_index, chunk_documents, embed_chunks, load_documents
from src.rag.models import Chunk, Classification, RetrievalResult, UserContext
from src.rag.retrieve import (
    embed_query,
    get_model,
    load_index,
    retrieve,
    retrieve_with_debug,
)


# Test user contexts for different access levels
EMPLOYEE_CONTEXT = UserContext(
    tenant_id="test-tenant",
    user_roles=["employee"],
    user_id="test-employee",
)

EXECUTIVE_CONTEXT = UserContext(
    tenant_id="test-tenant",
    user_roles=["executive", "employee"],
    user_id="test-executive",
)


@pytest.fixture
def indexed_data(sample_docs_dir, sample_index_dir):
    """Create a complete index from sample documents."""
    import src.rag.retrieve as retrieve_module
    from src.rag.config import settings

    original_index_dir = settings.index_dir
    settings.index_dir = sample_index_dir

    # Clear module-level cache
    retrieve_module._index = None
    retrieve_module._chunks = None
    retrieve_module._model = None

    try:
        documents = load_documents(sample_docs_dir)
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        build_index(embeddings, chunks)

        yield {
            "documents": documents,
            "chunks": chunks,
            "embeddings": embeddings,
            "index_dir": sample_index_dir,
        }
    finally:
        settings.index_dir = original_index_dir
        # Clear cache again
        retrieve_module._index = None
        retrieve_module._chunks = None
        retrieve_module._model = None


class TestLoadIndex:
    """Tests for load_index function."""

    def test_load_index_returns_tuple(self, indexed_data):
        """Test that load_index returns index and chunks."""
        index, chunks = load_index()

        assert index is not None
        assert chunks is not None
        assert isinstance(chunks, list)

    def test_load_index_correct_vector_count(self, indexed_data):
        """Test that index has correct number of vectors."""
        index, chunks = load_index()

        assert index.ntotal == len(chunks)
        assert index.ntotal == len(indexed_data["chunks"])

    def test_load_index_chunks_have_metadata(self, indexed_data):
        """Test that loaded chunks have metadata."""
        index, chunks = load_index()

        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata is not None
            assert chunk.doc_id is not None

    def test_load_index_caching(self, indexed_data):
        """Test that index is cached after first load."""
        index1, chunks1 = load_index()
        index2, chunks2 = load_index()

        # Should return the same objects
        assert index1 is index2
        assert chunks1 is chunks2

    def test_load_index_force_reload(self, indexed_data):
        """Test that force_reload bypasses cache."""
        import src.rag.retrieve as retrieve_module

        index1, chunks1 = load_index()
        retrieve_module._index = None  # Clear cache

        index2, chunks2 = load_index(force_reload=True)

        # Should be different objects after force reload
        assert index1 is not index2

    def test_load_index_file_not_found(self, sample_index_dir):
        """Test error when index files don't exist."""
        import src.rag.retrieve as retrieve_module
        from src.rag.config import settings

        original_index_dir = settings.index_dir
        settings.index_dir = sample_index_dir
        retrieve_module._index = None
        retrieve_module._chunks = None

        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                load_index(force_reload=True)

            assert "Run ingestion first" in str(exc_info.value)
        finally:
            settings.index_dir = original_index_dir


class TestEmbedQuery:
    """Tests for embed_query function."""

    def test_embed_query_returns_numpy_array(self):
        """Test that query embedding is numpy array."""
        embedding = embed_query("What is the vacation policy?")

        assert isinstance(embedding, np.ndarray)

    def test_embed_query_correct_shape(self):
        """Test that query embedding has correct shape."""
        from src.rag.config import settings

        embedding = embed_query("What is the vacation policy?")

        assert embedding.shape == (1, settings.embedding_dimension)

    def test_embed_query_normalized(self):
        """Test that query embedding is normalized for cosine similarity."""
        embedding = embed_query("What is the vacation policy?")

        # L2 norm should be approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_embed_query_different_queries_different_embeddings(self):
        """Test that different queries produce different embeddings."""
        emb1 = embed_query("What is the vacation policy?")
        emb2 = embed_query("How do quantum computers work?")

        similarity = np.dot(emb1[0], emb2[0])
        assert similarity < 0.9  # Should not be too similar


class TestRetrieve:
    """Tests for retrieve function."""

    def test_retrieve_returns_list(self, indexed_data):
        """Test that retrieve returns a list."""
        results = retrieve("vacation policy", user_context=EMPLOYEE_CONTEXT)

        assert isinstance(results, list)

    def test_retrieve_returns_retrieval_results(self, indexed_data):
        """Test that results are RetrievalResult objects."""
        results = retrieve("vacation policy", user_context=EMPLOYEE_CONTEXT)

        for result in results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.chunk, Chunk)
            assert isinstance(result.score, float)
            assert isinstance(result.rank, int)

    def test_retrieve_respects_k_parameter(self, indexed_data):
        """Test that k parameter limits results."""
        results_k2 = retrieve("vacation policy", k=2, user_context=EMPLOYEE_CONTEXT)
        results_k5 = retrieve("vacation policy", k=5, user_context=EMPLOYEE_CONTEXT)

        assert len(results_k2) <= 2
        assert len(results_k5) <= 5

    def test_retrieve_results_ranked(self, indexed_data):
        """Test that results are ranked by score."""
        results = retrieve("vacation policy", k=5, user_context=EMPLOYEE_CONTEXT)

        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Ranks should be sequential
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_retrieve_max_k_limit(self, indexed_data):
        """Test that max_top_k is enforced."""
        from src.rag.config import settings

        # Request more than max
        results = retrieve("vacation policy", k=100, user_context=EMPLOYEE_CONTEXT)

        assert len(results) <= settings.max_top_k

    def test_retrieve_default_k(self, indexed_data):
        """Test that default k is used when not specified."""
        from src.rag.config import settings

        results = retrieve("vacation policy", user_context=EMPLOYEE_CONTEXT)

        assert len(results) <= settings.default_top_k

    def test_retrieve_relevant_results(self, indexed_data):
        """Test that relevant documents are retrieved."""
        results = retrieve("vacation policy", user_context=EMPLOYEE_CONTEXT)

        # Should find the internal document about vacation
        found_vacation_doc = any(
            "vacation" in r.chunk.text.lower() for r in results
        )
        assert found_vacation_doc

    def test_retrieve_min_score_filter(self, indexed_data):
        """Test that min_score filters low-scoring results."""
        results_no_filter = retrieve("vacation policy", user_context=EMPLOYEE_CONTEXT)
        results_with_filter = retrieve(
            "vacation policy", min_score=0.5, user_context=EMPLOYEE_CONTEXT
        )

        # Filtered results should all have score >= min_score
        for result in results_with_filter:
            assert result.score >= 0.5

        # Filtered results should be subset
        assert len(results_with_filter) <= len(results_no_filter)

    def test_retrieve_includes_metadata(self, indexed_data):
        """Test that retrieved chunks include metadata."""
        results = retrieve("company products", user_context=EMPLOYEE_CONTEXT)

        for result in results:
            assert result.chunk.metadata is not None
            assert result.chunk.metadata.classification in [
                Classification.PUBLIC,
                Classification.INTERNAL,
                Classification.CONFIDENTIAL,
            ]

    def test_retrieve_no_user_context_returns_empty(self, indexed_data):
        """Test that retrieve with no user_context returns empty results (RBAC)."""
        results = retrieve("vacation policy", user_context=None)

        # Without user context, all access should be denied
        assert len(results) == 0


class TestRetrieveWithDebug:
    """Tests for retrieve_with_debug function."""

    def test_retrieve_with_debug_returns_dict(self, indexed_data):
        """Test that debug retrieval returns dictionary."""
        results = retrieve_with_debug("vacation policy", user_context=EMPLOYEE_CONTEXT)

        assert isinstance(results, dict)

    def test_retrieve_with_debug_contains_query(self, indexed_data):
        """Test that debug results contain query."""
        query = "vacation policy"
        results = retrieve_with_debug(query, user_context=EMPLOYEE_CONTEXT)

        assert results["query"] == query

    def test_retrieve_with_debug_contains_k(self, indexed_data):
        """Test that debug results contain k value."""
        results = retrieve_with_debug("vacation policy", k=3, user_context=EMPLOYEE_CONTEXT)

        assert results["k"] == 3

    def test_retrieve_with_debug_contains_results(self, indexed_data):
        """Test that debug results contain formatted results."""
        results = retrieve_with_debug("vacation policy", user_context=EMPLOYEE_CONTEXT)

        assert "results" in results
        assert "num_results" in results
        assert results["num_results"] == len(results["results"])

    def test_retrieve_with_debug_result_format(self, indexed_data):
        """Test that debug results have expected format."""
        results = retrieve_with_debug("vacation policy", user_context=EMPLOYEE_CONTEXT)

        for r in results["results"]:
            assert "rank" in r
            assert "score" in r
            assert "chunk_id" in r
            assert "doc_id" in r
            assert "classification" in r
            assert "text_preview" in r

    def test_retrieve_with_debug_text_preview_truncated(self, indexed_data):
        """Test that long text is truncated in preview."""
        results = retrieve_with_debug("vacation policy", user_context=EMPLOYEE_CONTEXT)

        for r in results["results"]:
            # Preview should end with ... if truncated, or be complete
            if len(r["text_preview"]) > 200:
                assert r["text_preview"].endswith("...")


class TestGetModel:
    """Tests for get_model function."""

    def test_get_model_returns_model(self):
        """Test that get_model returns a model."""
        from sentence_transformers import SentenceTransformer

        model = get_model()

        assert isinstance(model, SentenceTransformer)

    def test_get_model_caching(self):
        """Test that model is cached."""
        model1 = get_model()
        model2 = get_model()

        assert model1 is model2


class TestRetrievalScenarios:
    """Integration tests for retrieval scenarios."""

    def test_public_document_retrieval(self, indexed_data):
        """Test retrieval of public documents."""
        # Employee can see public and internal documents
        results = retrieve("company products CloudSync", user_context=EMPLOYEE_CONTEXT)

        # Should find public document
        public_results = [
            r for r in results
            if r.chunk.metadata.classification == Classification.PUBLIC
        ]
        assert len(public_results) > 0

    def test_internal_document_retrieval(self, indexed_data):
        """Test retrieval of internal documents."""
        # Employee can see internal documents
        results = retrieve("sick leave policy", user_context=EMPLOYEE_CONTEXT)

        # Should find internal document
        internal_results = [
            r for r in results
            if r.chunk.metadata.classification == Classification.INTERNAL
        ]
        assert len(internal_results) > 0

    def test_confidential_document_retrieval(self, indexed_data):
        """Test retrieval of confidential documents."""
        # Executive can see confidential documents
        results = retrieve("executive salary CEO compensation", user_context=EXECUTIVE_CONTEXT)

        # Should find confidential document
        confidential_results = [
            r for r in results
            if r.chunk.metadata.classification == Classification.CONFIDENTIAL
        ]
        assert len(confidential_results) > 0

    def test_cross_classification_retrieval(self, indexed_data):
        """Test that retrieval works across classifications."""
        # Executive can see all classifications
        results = retrieve("document information", k=10, user_context=EXECUTIVE_CONTEXT)

        # Should potentially find documents from multiple classifications
        classifications = set(r.chunk.metadata.classification for r in results)
        # At minimum should find something
        assert len(classifications) >= 1

    def test_employee_cannot_see_confidential(self, indexed_data):
        """Test that employee cannot retrieve confidential documents."""
        # Employee should not see confidential documents
        results = retrieve("executive salary CEO compensation", user_context=EMPLOYEE_CONTEXT)

        # Should not find confidential document
        confidential_results = [
            r for r in results
            if r.chunk.metadata.classification == Classification.CONFIDENTIAL
        ]
        assert len(confidential_results) == 0
