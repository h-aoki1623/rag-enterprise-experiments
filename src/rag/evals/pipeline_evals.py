"""F. Pipeline Evals - End-to-end integration evaluation.

Evaluates the full RAG pipeline from query to answer with:
- Outcome validation: success/blocked/no_results/uncertain
- Policy flag matching: required/forbidden flags
- Latency measurement: per-stage breakdown (retrieve, generate, guardrail)
- RBAC filtering impact measurement

Outcome definitions:
- success: citations >= 1, no GUARDRAIL_BLOCKED flag
- blocked: GUARDRAIL_BLOCKED in policy_flags
- no_results: retrieved_count == 0 or NO_CONTEXT flag
- uncertain: UNCERTAIN flag present (set by generate.py when confidence < threshold)
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from src.rag.evals.metrics import calculate_percentiles
from src.rag.evals.models import (
    BaseEvalCase,
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    EvalTrace,
    PipelineEvalCase,
    PipelineOutcome,
)
from src.rag.models import PolicyFlag, UserContext


class PipelineEvaluator:
    """Evaluator for end-to-end pipeline metrics."""

    def __init__(
        self,
        k: int = 5,
        use_hierarchical: bool = False,
    ):
        """Initialize pipeline evaluator.

        Args:
            k: Number of chunks to retrieve
            use_hierarchical: Whether to use hierarchical retrieval
        """
        self.k = k
        self.use_hierarchical = use_hierarchical

    def _determine_outcome(
        self,
        policy_flags: list[PolicyFlag],
        citation_count: int,
        confidence: float,
        retrieved_count: int,
    ) -> PipelineOutcome:
        """Determine the actual outcome based on results.

        Args:
            policy_flags: Policy flags from generation
            citation_count: Number of citations
            confidence: Confidence score
            retrieved_count: Number of retrieved chunks

        Returns:
            PipelineOutcome enum value
        """
        if PolicyFlag.GUARDRAIL_BLOCKED in policy_flags:
            return PipelineOutcome.BLOCKED

        if retrieved_count == 0 or PolicyFlag.NO_CONTEXT in policy_flags:
            return PipelineOutcome.NO_RESULTS

        if PolicyFlag.UNCERTAIN in policy_flags:
            return PipelineOutcome.UNCERTAIN

        if citation_count >= 1:
            return PipelineOutcome.SUCCESS

        return PipelineOutcome.UNCERTAIN

    def evaluate_case(
        self,
        base_case: BaseEvalCase,
        pipeline_case: PipelineEvalCase,
    ) -> tuple[EvalResult, Optional[EvalTrace]]:
        """Evaluate a single pipeline case.

        Args:
            base_case: Base case with query and user info
            pipeline_case: Pipeline-specific ground truth

        Returns:
            Tuple of (EvalResult, EvalTrace if error or failure)
        """
        from src.rag.generate import generate
        from src.rag.retrieve import retrieve

        total_start = time.perf_counter()
        stage_latencies = {}

        user_context = UserContext(user_roles=base_case.user_roles)

        trace = EvalTrace(
            case_id=base_case.case_id,
            run_id="",
        )

        try:
            # Stage 1: Retrieval (for metrics, generation does this internally too)
            retrieve_start = time.perf_counter()
            retrieval_results = retrieve(
                base_case.query,
                k=self.k,
                user_context=user_context,
            )
            stage_latencies["retrieve"] = (time.perf_counter() - retrieve_start) * 1000

            trace.retrieved_doc_ids = [r.chunk.doc_id for r in retrieval_results]
            trace.retrieved_chunk_ids = [r.chunk.chunk_id for r in retrieval_results]
            trace.retrieval_scores = [r.score for r in retrieval_results]

            # Stage 2: Generation (includes guardrails internally)
            generate_start = time.perf_counter()
            gen_result = generate(
                query=base_case.query,
                k=self.k,
                use_hierarchical=self.use_hierarchical,
                user_context=user_context,
            )
            stage_latencies["generate"] = (time.perf_counter() - generate_start) * 1000

            trace.generated_answer = gen_result.answer
            trace.citations = [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text_snippet": c.text_snippet,
                }
                for c in gen_result.citations
            ]

            total_latency = (time.perf_counter() - total_start) * 1000
            stage_latencies["total"] = total_latency
            trace.stage_latencies = stage_latencies

            # Determine actual outcome
            actual_outcome = self._determine_outcome(
                gen_result.policy_flags,
                len(gen_result.citations),
                gen_result.confidence,
                len(retrieval_results),
            )

            # Check outcome match
            outcome_match = actual_outcome == pipeline_case.expected_outcome

            # Check required flags
            actual_flag_values = {f.value for f in gen_result.policy_flags}
            required_flags_present = all(
                flag in actual_flag_values for flag in pipeline_case.required_flags
            )

            # Check forbidden flags
            forbidden_flags_absent = all(
                flag not in actual_flag_values for flag in pipeline_case.forbidden_flags
            )

            # Check citation count
            citations_ok = len(gen_result.citations) >= pipeline_case.min_citations

            # Check latency budget
            latency_ok = True
            latency_violations = []
            for percentile, budget in pipeline_case.latency_budget_ms.items():
                # For single case, we just check against the budget
                if total_latency > budget:
                    latency_ok = False
                    latency_violations.append(f"{percentile}: {total_latency:.0f}ms > {budget}ms")

            # Overall success
            success = (
                outcome_match
                and required_flags_present
                and forbidden_flags_absent
                and citations_ok
                and latency_ok
            )

            metrics = {
                "outcome_match": 1.0 if outcome_match else 0.0,
                "required_flags_present": 1.0 if required_flags_present else 0.0,
                "forbidden_flags_absent": 1.0 if forbidden_flags_absent else 0.0,
                "citations_ok": 1.0 if citations_ok else 0.0,
                "latency_ok": 1.0 if latency_ok else 0.0,
                "confidence": gen_result.confidence,
                "citation_count": float(len(gen_result.citations)),
                "retrieved_count": float(len(retrieval_results)),
                "total_latency_ms": total_latency,
                "retrieve_latency_ms": stage_latencies.get("retrieve", 0),
                "generate_latency_ms": stage_latencies.get("generate", 0),
            }

            details = {
                "expected_outcome": pipeline_case.expected_outcome.value,
                "actual_outcome": actual_outcome.value,
                "expected_flags": pipeline_case.required_flags,
                "forbidden_flags": pipeline_case.forbidden_flags,
                "actual_flags": list(actual_flag_values),
                "answer_length": len(gen_result.answer),
                "latency_violations": latency_violations,
                "stage_latencies": stage_latencies,
            }

            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.PIPELINE,
                    success=success,
                    metric_values=metrics,
                    details=details,
                    latency_ms=total_latency,
                ),
                trace if not success else None,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - total_start) * 1000
            trace.stage_latencies = stage_latencies
            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.PIPELINE,
                    success=False,
                    error=str(e),
                    latency_ms=total_latency,
                ),
                trace,
            )

    def evaluate_dataset(
        self,
        cases: list[tuple[BaseEvalCase, PipelineEvalCase]],
    ) -> tuple[EvalRunSummary, list[EvalTrace]]:
        """Evaluate all pipeline cases.

        Args:
            cases: List of (base_case, pipeline_case) tuples

        Returns:
            Tuple of (EvalRunSummary, list of traces for failed cases)
        """
        run_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        results = []
        traces = []

        for base_case, pipeline_case in cases:
            result, trace = self.evaluate_case(base_case, pipeline_case)
            results.append(result)
            if trace:
                trace.run_id = run_id
                traces.append(trace)

        duration = time.perf_counter() - start_time
        passed = sum(1 for r in results if r.success)

        # Aggregate metrics
        aggregate = {}
        metric_names = [
            "outcome_match",
            "required_flags_present",
            "forbidden_flags_absent",
            "citations_ok",
            "latency_ok",
            "confidence",
            "citation_count",
            "retrieved_count",
            "total_latency_ms",
            "retrieve_latency_ms",
            "generate_latency_ms",
        ]

        for metric_name in metric_names:
            values = [
                r.metric_values.get(metric_name)
                for r in results
                if r.metric_values.get(metric_name) is not None
            ]
            if values:
                aggregate[f"mean_{metric_name}"] = sum(values) / len(values)

        # Latency distributions
        total_latencies = [r.metric_values.get("total_latency_ms", 0) for r in results]
        retrieve_latencies = [r.metric_values.get("retrieve_latency_ms", 0) for r in results]
        generate_latencies = [r.metric_values.get("generate_latency_ms", 0) for r in results]

        metric_distributions = {
            "total_latency_ms": calculate_percentiles(total_latencies),
            "retrieve_latency_ms": calculate_percentiles(retrieve_latencies),
            "generate_latency_ms": calculate_percentiles(generate_latencies),
        }

        # Outcome breakdown
        outcome_counts: dict[str, int] = {}
        for result in results:
            outcome = result.details.get("actual_outcome", "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        category_breakdown = {
            "outcomes": {
                "counts": outcome_counts,
                "pass_rate": passed / len(results) if results else 0,
            },
        }

        # Top failures
        failed_results = [
            (r, r.metric_values.get("outcome_match", 0))
            for r in results
            if not r.success
        ]
        failed_results.sort(key=lambda x: x[1])
        top_failures = [r[0].case_id for r in failed_results[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.PIPELINE,
                timestamp=datetime.utcnow(),
                total_cases=len(cases),
                passed_cases=passed,
                failed_cases=len(cases) - passed,
                aggregate_metrics=aggregate,
                metric_distributions=metric_distributions,
                category_breakdown=category_breakdown,
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "k": self.k,
                    "use_hierarchical": self.use_hierarchical,
                },
            ),
            traces,
        )
