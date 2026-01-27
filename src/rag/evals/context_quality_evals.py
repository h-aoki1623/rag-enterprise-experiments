"""B. Context Quality Evals - Retrieved context usefulness evaluation.

Metrics:
- Redundancy Ratio: Duplicate/near-duplicate chunks in top-k (target < 0.2)
- Fact Dispersion: How many chunks contain the gold fact (lower is better)
- Unique Token Ratio: Distinct tokens / total tokens (target > 0.7)

Supports query_type-based diversity targets:
- faq: expect low diversity (focused answer)
- research: allow high diversity (multi-faceted info)
- comparison: expect medium diversity
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from src.rag.config import EvalSettings, settings
from src.rag.evals.metrics import (
    calculate_percentiles,
    fact_dispersion,
    redundancy_ratio,
    redundancy_ratio_tfidf,
    unique_token_ratio,
)
from src.rag.evals.models import (
    BaseEvalCase,
    ContextQualityEvalCase,
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    EvalTrace,
    QueryType,
)
from src.rag.models import UserContext


class ContextQualityEvaluator:
    """Evaluator for context quality metrics."""

    # Target thresholds by query type
    REDUNDANCY_TARGETS = {
        QueryType.FAQ: 0.15,  # FAQ should have minimal redundancy
        QueryType.RESEARCH: 0.25,  # Research may have some topic overlap
        QueryType.COMPARISON: 0.20,  # Comparison has moderate redundancy
    }

    UNIQUE_TOKEN_TARGETS = {
        QueryType.FAQ: 0.75,  # FAQ should be focused
        QueryType.RESEARCH: 0.60,  # Research may repeat key terms
        QueryType.COMPARISON: 0.65,
    }

    def __init__(
        self,
        default_user_context: Optional[UserContext] = None,
        use_tfidf: bool = True,
        ngram_size: int = 5,
        k: int = 5,
        eval_settings: Optional[EvalSettings] = None,
    ):
        """Initialize context quality evaluator.

        Args:
            default_user_context: Default user context for retrieval
            use_tfidf: Whether to use TF-IDF for redundancy detection
            ngram_size: N-gram size for overlap detection
            k: Number of chunks to retrieve
            eval_settings: Evaluation settings (uses global settings if None)
        """
        self.default_user_context = default_user_context or UserContext(
            user_roles=["employee", "executive"]
        )
        self.use_tfidf = use_tfidf
        self.ngram_size = ngram_size
        self.k = k

        # Load settings from config
        self.eval_settings = eval_settings or settings.evals
        self.redundancy_threshold = self.eval_settings.redundancy_threshold
        self.tfidf_similarity_threshold = self.eval_settings.tfidf_similarity_threshold

    def evaluate_case(
        self,
        base_case: BaseEvalCase,
        context_case: ContextQualityEvalCase,
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalResult, Optional[EvalTrace]]:
        """Evaluate context quality for a single case.

        Args:
            base_case: Base case with query and user info
            context_case: Context quality ground truth
            user_context: Override user context

        Returns:
            Tuple of (EvalResult, EvalTrace if error or failure)
        """
        from src.rag.retrieve import retrieve

        start_time = time.perf_counter()
        user_ctx = user_context or UserContext(user_roles=base_case.user_roles)
        if not user_ctx.user_roles:
            user_ctx = self.default_user_context

        trace = EvalTrace(
            case_id=base_case.case_id,
            run_id="",
        )

        try:
            # Retrieve chunks
            results = retrieve(base_case.query, k=self.k, user_context=user_ctx)
            chunk_texts = [r.chunk.text for r in results]
            chunk_ids = [r.chunk.chunk_id for r in results]

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update trace
            trace.retrieved_doc_ids = [r.chunk.doc_id for r in results]
            trace.retrieved_chunk_ids = chunk_ids
            trace.retrieval_scores = [r.score for r in results]
            trace.stage_latencies = {"retrieve": latency_ms}

            if not chunk_texts:
                return (
                    EvalResult(
                        case_id=base_case.case_id,
                        perspective=EvalPerspective.CONTEXT_QUALITY,
                        success=False,
                        metric_values={},
                        details={"error": "No chunks retrieved"},
                        latency_ms=latency_ms,
                    ),
                    trace,
                )

            metrics = {}

            # Redundancy metrics
            if self.use_tfidf:
                metrics["redundancy_ratio_tfidf"] = redundancy_ratio_tfidf(
                    chunk_texts, threshold=self.tfidf_similarity_threshold
                )
            metrics["redundancy_ratio_ngram"] = redundancy_ratio(
                chunk_texts, threshold=self.redundancy_threshold, n=self.ngram_size
            )
            # Use the higher of the two as the primary redundancy metric
            metrics["redundancy_ratio"] = max(
                metrics.get("redundancy_ratio_tfidf", 0),
                metrics["redundancy_ratio_ngram"],
            )

            # Unique token ratio
            metrics["unique_token_ratio"] = unique_token_ratio(chunk_texts)

            # Fact dispersion for each gold fact
            total_dispersion = 0
            facts_found = 0
            expected_chunks_found = 0

            for gold_fact in context_case.gold_facts:
                dispersion = fact_dispersion(
                    chunk_texts, gold_fact.fact, gold_fact.aliases
                )
                total_dispersion += dispersion
                if dispersion > 0:
                    facts_found += 1

            if context_case.gold_facts:
                metrics["avg_fact_dispersion"] = (
                    total_dispersion / len(context_case.gold_facts)
                )
                metrics["facts_found_ratio"] = facts_found / len(context_case.gold_facts)
            else:
                metrics["avg_fact_dispersion"] = 0
                metrics["facts_found_ratio"] = 1.0  # No facts to find

            # Check if expected chunks were retrieved
            if context_case.expected_chunks:
                found = sum(1 for ec in context_case.expected_chunks if ec in chunk_ids)
                metrics["expected_chunks_found"] = found / len(context_case.expected_chunks)
            else:
                metrics["expected_chunks_found"] = 1.0

            # Get targets based on query type
            query_type = base_case.query_type
            redundancy_target = self.REDUNDANCY_TARGETS.get(query_type, 0.20)
            unique_token_target = self.UNIQUE_TOKEN_TARGETS.get(query_type, 0.70)

            # Success criteria
            success = (
                metrics["redundancy_ratio"] < redundancy_target
                and metrics["unique_token_ratio"] > unique_token_target
                and metrics["facts_found_ratio"] >= 0.5  # At least half the facts found
            )

            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.CONTEXT_QUALITY,
                    success=success,
                    metric_values=metrics,
                    details={
                        "query_type": query_type.value,
                        "redundancy_target": redundancy_target,
                        "unique_token_target": unique_token_target,
                        "chunk_count": len(chunk_texts),
                    },
                    latency_ms=latency_ms,
                ),
                trace if not success else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.CONTEXT_QUALITY,
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                ),
                trace,
            )

    def evaluate_dataset(
        self,
        cases: list[tuple[BaseEvalCase, ContextQualityEvalCase]],
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalRunSummary, list[EvalTrace]]:
        """Evaluate all context quality cases.

        Args:
            cases: List of (base_case, context_case) tuples
            user_context: Override user context for all cases

        Returns:
            Tuple of (EvalRunSummary, list of traces for failed cases)
        """
        run_id = f"context-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        results = []
        traces = []

        for base_case, context_case in cases:
            result, trace = self.evaluate_case(base_case, context_case, user_context)
            results.append(result)
            if trace:
                trace.run_id = run_id
                traces.append(trace)

        duration = time.perf_counter() - start_time
        passed = sum(1 for r in results if r.success)

        # Aggregate metrics
        aggregate = {}
        metric_names = [
            "redundancy_ratio",
            "redundancy_ratio_tfidf",
            "redundancy_ratio_ngram",
            "unique_token_ratio",
            "avg_fact_dispersion",
            "facts_found_ratio",
            "expected_chunks_found",
        ]

        for metric_name in metric_names:
            values = [
                r.metric_values.get(metric_name)
                for r in results
                if r.metric_values.get(metric_name) is not None
            ]
            if values:
                aggregate[f"mean_{metric_name}"] = sum(values) / len(values)

        # Latency distribution
        latencies = [r.latency_ms for r in results if r.latency_ms]
        latency_dist = calculate_percentiles(latencies)

        # Category breakdown by query type
        category_breakdown = {}
        for base_case, _ in cases:
            qt = base_case.query_type.value
            if qt not in category_breakdown:
                category_breakdown[qt] = {"total": 0, "passed": 0}
            category_breakdown[qt]["total"] += 1

        for result in results:
            # Find the corresponding base case to get query_type
            qt = result.details.get("query_type", "faq")
            if qt in category_breakdown and result.success:
                category_breakdown[qt]["passed"] += 1

        # Top failures
        failed_results = [
            (r, r.metric_values.get("redundancy_ratio", 1.0))
            for r in results
            if not r.success
        ]
        failed_results.sort(key=lambda x: -x[1])  # Sort by redundancy descending (worst first)
        top_failures = [r[0].case_id for r in failed_results[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.CONTEXT_QUALITY,
                timestamp=datetime.utcnow(),
                total_cases=len(cases),
                passed_cases=passed,
                failed_cases=len(cases) - passed,
                aggregate_metrics=aggregate,
                metric_distributions={"latency_ms": latency_dist},
                category_breakdown=category_breakdown,
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "k": self.k,
                    "use_tfidf": self.use_tfidf,
                    "ngram_size": self.ngram_size,
                },
            ),
            traces,
        )
