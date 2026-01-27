"""Data models for evaluation framework.

This module defines all Pydantic models used across the evaluation system,
including eval cases, results, and execution traces.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalPerspective(str, Enum):
    """Evaluation perspectives for the RAG system."""

    RETRIEVAL = "retrieval"
    CONTEXT_QUALITY = "context_quality"
    GROUNDEDNESS = "groundedness"
    SAFETY = "safety"
    PIPELINE = "pipeline"


class QueryType(str, Enum):
    """Query types for context quality evaluation."""

    FAQ = "faq"  # Expect low diversity (focused answer)
    RESEARCH = "research"  # Allow high diversity (multi-faceted info)
    COMPARISON = "comparison"  # Expect medium diversity


class ClaimType(str, Enum):
    """Claim types for groundedness evaluation."""

    ASSERTION = "assertion"  # Definitive statements (is, must, will) - strict verification
    INFERENCE = "inference"  # Hedged statements (may, might, could) - relaxed verification
    GENERAL = "general"  # General statements (generally, typically) - skip verification


class PipelineOutcome(str, Enum):
    """Expected outcomes for pipeline evaluation."""

    SUCCESS = "success"  # citations >= 1, no GUARDRAIL_BLOCKED
    BLOCKED = "blocked"  # GUARDRAIL_BLOCKED in policy_flags
    NO_RESULTS = "no_results"  # retrieved_count == 0 or NO_CONTEXT flag
    UNCERTAIN = "uncertain"  # UNCERTAIN flag or confidence < 0.5


# =============================================================================
# Eval Case Models
# =============================================================================


class BaseEvalCase(BaseModel):
    """Base model for all evaluation cases."""

    case_id: str = Field(..., description="Unique case identifier (shared across perspectives)")
    query: str = Field(..., description="The query text")
    user_roles: list[str] = Field(default_factory=list, description="User roles for RBAC")
    query_type: QueryType = Field(default=QueryType.FAQ, description="Query type classification")


class RetrievalEvalCase(BaseModel):
    """Evaluation case for retrieval perspective (references case_id from base)."""

    case_id: str = Field(..., description="Reference to base case")
    relevant_docs: list[str] = Field(..., description="List of relevant doc_ids")
    relevant_chunks: list[str] = Field(
        default_factory=list, description="List of relevant chunk_ids (optional)"
    )
    relevance_grades: dict[str, int] = Field(
        default_factory=dict,
        description="Graded relevance: doc_id -> grade (3=direct, 2=strong, 1=peripheral, 0=irrelevant)",
    )
    chunk_relevance_grades: dict[str, int] = Field(
        default_factory=dict,
        description="Chunk-level graded relevance: chunk_id -> grade (3=direct, 2=strong, 1=peripheral)",
    )


class GoldFact(BaseModel):
    """A gold fact with aliases for context quality evaluation."""

    fact: str = Field(..., description="The canonical fact text")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative expressions of the same fact"
    )


class ContextQualityEvalCase(BaseModel):
    """Evaluation case for context quality perspective."""

    case_id: str = Field(..., description="Reference to base case")
    gold_facts: list[GoldFact] = Field(
        default_factory=list, description="Expected facts with aliases"
    )
    expected_chunks: list[str] = Field(
        default_factory=list, description="Chunk IDs where facts should be found"
    )


class GroundednessEvalCase(BaseModel):
    """Evaluation case for groundedness perspective."""

    case_id: str = Field(..., description="Reference to base case")
    expected_claims: list[str] = Field(
        default_factory=list, description="Key claims that should appear in answer"
    )
    expected_citations: list[str] = Field(
        default_factory=list, description="Expected doc_ids to be cited"
    )
    forbidden_claims: list[str] = Field(
        default_factory=list, description="Claims that should NOT appear (hallucination indicators)"
    )


class SafetyEvalCase(BaseModel):
    """Evaluation case for safety perspective (injection/leakage)."""

    case_id: str = Field(..., description="Unique case identifier")
    text: str = Field(..., description="Query or output text to evaluate")
    is_attack: bool = Field(..., description="Whether this is an attack case")
    category: str = Field(default="unknown", description="Attack/case category")
    expected_detection: bool = Field(
        default=True, description="Whether detection is expected for attacks"
    )


class PipelineEvalCase(BaseModel):
    """Evaluation case for pipeline (E2E) perspective."""

    case_id: str = Field(..., description="Reference to base case")
    expected_outcome: PipelineOutcome = Field(..., description="Expected outcome")
    required_flags: list[str] = Field(
        default_factory=list, description="Policy flags that must be present"
    )
    forbidden_flags: list[str] = Field(
        default_factory=list, description="Policy flags that must NOT be present"
    )
    min_citations: int = Field(default=0, description="Minimum number of citations required")
    latency_budget_ms: dict[str, float] = Field(
        default_factory=dict, description="Latency budget by percentile (e.g., {'p95': 3000})"
    )


# =============================================================================
# Result Models
# =============================================================================


class EvalResult(BaseModel):
    """Result from evaluating a single case."""

    case_id: str = Field(..., description="The evaluated case ID")
    perspective: EvalPerspective = Field(..., description="Evaluation perspective")
    success: bool = Field(..., description="Whether the case passed")
    metric_values: dict[str, float] = Field(
        default_factory=dict, description="Computed metric values"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details for debugging"
    )
    latency_ms: Optional[float] = Field(default=None, description="Evaluation latency in ms")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")


class EvalRunSummary(BaseModel):
    """Summary of an evaluation run for a perspective."""

    run_id: str = Field(..., description="Unique run identifier")
    perspective: EvalPerspective = Field(..., description="Evaluation perspective")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Run timestamp")
    total_cases: int = Field(..., description="Total number of cases evaluated")
    passed_cases: int = Field(..., description="Number of cases that passed")
    failed_cases: int = Field(..., description="Number of cases that failed")
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict, description="Aggregated metric values"
    )
    metric_distributions: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metric distributions (e.g., percentiles)"
    )
    category_breakdown: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Metrics broken down by category"
    )
    top_failures: list[str] = Field(
        default_factory=list, description="Case IDs of worst failures (up to 10)"
    )
    duration_seconds: float = Field(..., description="Total evaluation duration")
    config: dict[str, Any] = Field(default_factory=dict, description="Evaluation configuration")


# =============================================================================
# Trace Models
# =============================================================================


class EvalTrace(BaseModel):
    """Execution trace for debugging failed cases."""

    case_id: str = Field(..., description="The evaluated case ID")
    run_id: str = Field(..., description="The evaluation run ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Trace timestamp")

    # Retrieval artifacts
    retrieved_doc_ids: list[str] = Field(
        default_factory=list, description="Retrieved document IDs"
    )
    retrieved_chunk_ids: list[str] = Field(default_factory=list, description="Retrieved chunk IDs")
    retrieval_scores: list[float] = Field(default_factory=list, description="Retrieval scores")

    # Generation artifacts
    final_prompt: Optional[str] = Field(default=None, description="Final prompt sent to LLM")
    generated_answer: Optional[str] = Field(default=None, description="Generated answer")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="Citations")

    # Guardrail artifacts
    input_guardrail_result: Optional[dict[str, Any]] = Field(
        default=None, description="Input guardrail result"
    )
    output_guardrail_result: Optional[dict[str, Any]] = Field(
        default=None, description="Output guardrail result"
    )

    # Timing
    stage_latencies: dict[str, float] = Field(
        default_factory=dict, description="Latency by stage (retrieve, generate, guardrail)"
    )
