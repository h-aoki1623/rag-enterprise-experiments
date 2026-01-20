"""Tests for the generation layer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.rag.generate import _compute_policy_flags, _parse_llm_response, generate
from src.rag.models import (
    Chunk,
    ChunkLevel,
    Citation,
    Classification,
    DocumentMetadata,
    GenerationResult,
    HierarchicalChunk,
    HierarchicalRetrievalResult,
    PolicyFlag,
    RetrievalResult,
)
from src.rag.prompts import SYSTEM_PROMPT, build_context_section, build_user_prompt


class TestPrompts:
    """Tests for prompt building functions."""

    def test_system_prompt_contains_injection_prevention(self):
        """System prompt should contain prompt injection prevention rules."""
        assert "reference information" in SYSTEM_PROMPT.lower()
        assert "not" in SYSTEM_PROMPT.lower()
        assert "instructions" in SYSTEM_PROMPT.lower()

    def test_system_prompt_requires_citations(self):
        """System prompt should require citations."""
        assert "citations" in SYSTEM_PROMPT.lower()
        assert "doc_id" in SYSTEM_PROMPT.lower()
        assert "chunk_id" in SYSTEM_PROMPT.lower()

    def test_build_context_section_empty(self):
        """Empty contexts should return no documents message."""
        result = build_context_section([])
        assert "no relevant documents" in result.lower()

    def test_build_context_section_flat(self, sample_metadata):
        """Build context from flat retrieval results."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Test content about vacation policy.",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        context = build_context_section(results)

        assert "doc-001" in context
        assert "chunk-001" in context
        assert "Test content about vacation policy" in context
        assert "0.85" in context

    def test_build_context_section_hierarchical(self, sample_metadata):
        """Build context from hierarchical retrieval results."""
        parent = HierarchicalChunk(
            chunk_id="parent-001",
            text="Parent content about policies.",
            doc_id="doc-001",
            metadata=sample_metadata,
            level=ChunkLevel.PARENT,
            section_header="Company Policies",
        )
        child = HierarchicalChunk(
            chunk_id="child-001",
            text="Child content about vacation.",
            doc_id="doc-001",
            metadata=sample_metadata,
            level=ChunkLevel.CHILD,
            parent_id="parent-001",
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

        context = build_context_section(results)

        assert "doc-001" in context
        assert "parent-001" in context
        assert "Company Policies" in context
        assert "Parent content about policies" in context

    def test_build_user_prompt(self, sample_metadata):
        """Build complete user prompt."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Employees get 15 days vacation.",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        prompt = build_user_prompt("How many vacation days?", results)

        assert "Reference Documents" in prompt
        assert "Question" in prompt
        assert "How many vacation days?" in prompt
        assert "Employees get 15 days vacation" in prompt


class TestPolicyFlags:
    """Tests for policy flag computation."""

    def test_empty_context_returns_no_context_flag(self):
        """Empty context should return NO_CONTEXT flag."""
        flags = _compute_policy_flags([])
        assert PolicyFlag.NO_CONTEXT in flags

    def test_pii_flag_detection(self):
        """PII metadata should trigger PII_REFERENCED flag."""
        metadata = DocumentMetadata(
            doc_id="test-doc",
            classification=Classification.INTERNAL,
            pii_flag=True,
        )
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Test content",
            doc_id="test-doc",
            metadata=metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        flags = _compute_policy_flags(results)

        assert PolicyFlag.PII_REFERENCED in flags

    def test_confidential_flag_detection(self):
        """Confidential classification should trigger CONFIDENTIAL flag."""
        metadata = DocumentMetadata(
            doc_id="test-doc",
            classification=Classification.CONFIDENTIAL,
            pii_flag=False,
        )
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Test content",
            doc_id="test-doc",
            metadata=metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        flags = _compute_policy_flags(results)

        assert PolicyFlag.CONFIDENTIAL in flags

    def test_public_doc_no_special_flags(self):
        """Public doc without PII should have no special flags."""
        metadata = DocumentMetadata(
            doc_id="test-doc",
            classification=Classification.PUBLIC,
            pii_flag=False,
        )
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Test content",
            doc_id="test-doc",
            metadata=metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        flags = _compute_policy_flags(results)

        assert len(flags) == 0


class TestParseLLMResponse:
    """Tests for LLM response parsing."""

    def test_parse_valid_json(self):
        """Valid JSON should parse correctly."""
        response = json.dumps({
            "answer": "Test answer",
            "citations": [{"doc_id": "doc-001", "chunk_id": "chunk-001", "text_snippet": "test"}],
            "confidence": 0.85,
        })

        result = _parse_llm_response(response)

        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.85
        assert len(result["citations"]) == 1

    def test_parse_json_with_markdown_wrapper(self):
        """JSON wrapped in markdown code blocks should parse."""
        response = """```json
{
    "answer": "Test answer",
    "citations": [],
    "confidence": 0.5
}
```"""

        result = _parse_llm_response(response)

        assert result["answer"] == "Test answer"

    def test_parse_invalid_json_raises(self):
        """Invalid JSON should raise ValueError."""
        response = "This is not valid JSON"

        with pytest.raises(ValueError) as exc_info:
            _parse_llm_response(response)

        assert "Failed to parse" in str(exc_info.value)


class TestGenerate:
    """Tests for the generate function."""

    def test_generate_without_api_key_raises(self, sample_metadata):
        """Generate without API key should raise ValueError."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Test content",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        with patch("src.rag.generate.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""

            with pytest.raises(ValueError) as exc_info:
                generate("Test query", results)

            assert "API key" in str(exc_info.value)

    def test_generate_with_mock_client(self, sample_metadata):
        """Generate with mocked Anthropic client."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Employees get 15 days vacation.",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps({
                    "answer": "Employees receive 15 days of vacation.",
                    "citations": [
                        {
                            "doc_id": "doc-001",
                            "chunk_id": "chunk-001",
                            "text_snippet": "15 days vacation",
                        }
                    ],
                    "confidence": 0.95,
                })
            )
        ]
        mock_client.messages.create.return_value = mock_response

        result = generate("How many vacation days?", results, client=mock_client)

        assert isinstance(result, GenerationResult)
        assert "15 days" in result.answer
        assert result.confidence == 0.95
        assert len(result.citations) == 1
        assert result.citations[0].doc_id == "doc-001"

    def test_generate_adds_uncertain_flag_on_low_confidence(self, sample_metadata):
        """Low confidence should add UNCERTAIN flag."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Some content.",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps({
                    "answer": "I'm not sure.",
                    "citations": [],
                    "confidence": 0.2,
                })
            )
        ]
        mock_client.messages.create.return_value = mock_response

        result = generate("Unknown question", results, client=mock_client)

        assert PolicyFlag.UNCERTAIN in result.policy_flags

    def test_generate_adds_no_context_flag_when_no_citations(self, sample_metadata):
        """No citations should add NO_CONTEXT flag."""
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Some content.",
            doc_id="doc-001",
            metadata=sample_metadata,
        )
        results = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps({
                    "answer": "I cannot find relevant information.",
                    "citations": [],
                    "confidence": 0.0,
                })
            )
        ]
        mock_client.messages.create.return_value = mock_response

        result = generate("Unknown question", results, client=mock_client)

        assert PolicyFlag.NO_CONTEXT in result.policy_flags


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_valid_generation_result(self):
        """Valid GenerationResult should be created correctly."""
        result = GenerationResult(
            answer="Test answer",
            citations=[
                Citation(doc_id="doc-001", chunk_id="chunk-001", text_snippet="test")
            ],
            confidence=0.85,
            policy_flags=[PolicyFlag.CONFIDENTIAL],
            raw_context_used=True,
        )

        assert result.answer == "Test answer"
        assert result.confidence == 0.85
        assert len(result.citations) == 1

    def test_confidence_clamped_to_range(self):
        """Confidence should be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            GenerationResult(
                answer="Test",
                citations=[],
                confidence=1.5,  # Invalid: > 1.0
                policy_flags=[],
            )

        with pytest.raises(ValueError):
            GenerationResult(
                answer="Test",
                citations=[],
                confidence=-0.5,  # Invalid: < 0.0
                policy_flags=[],
            )
