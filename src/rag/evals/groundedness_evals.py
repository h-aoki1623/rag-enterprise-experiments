"""C. Groundedness Evals - Answer-context alignment evaluation.

Metrics:
- Claim Support Rate: % of answer claims supported by context (target > 0.9)
- Unsupported Claims: Count of claims with no context basis (target = 0)
- Citation Validity (form): cited doc_id exists in retrieved results (target > 0.95)
- Citation Validity (content): claim evidence exists near citation (target > 0.85)
- Numeric Fabrication: numbers not found in context (target = 0)

Supports claim type classification to reduce false positives:
- assertion: strict verification
- inference: relaxed verification
- general: skip verification
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from src.rag.config import EvalSettings, settings
from src.rag.evals.metrics import (
    calculate_percentiles,
    citation_validity_content,
    citation_validity_form,
    claim_context_overlap,
    extract_claims,
    numeric_fabrication_count,
)
from src.rag.evals.models import (
    BaseEvalCase,
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    EvalTrace,
    GroundednessEvalCase,
)
from src.rag.models import UserContext


class GroundednessEvaluator:
    """Evaluator for groundedness (answer-context alignment) metrics."""

    def __init__(
        self,
        default_user_context: Optional[UserContext] = None,
        k: int = 5,
        use_hierarchical: bool = False,
        eval_settings: Optional[EvalSettings] = None,
    ):
        """Initialize groundedness evaluator.

        Args:
            default_user_context: Default user context for retrieval
            k: Number of chunks to retrieve
            use_hierarchical: Whether to use hierarchical retrieval
            eval_settings: Evaluation settings (uses global settings if None)
        """
        self.default_user_context = default_user_context or UserContext(
            user_roles=["employee", "executive"]
        )
        self.k = k
        self.use_hierarchical = use_hierarchical

        # Load settings from config
        self.eval_settings = eval_settings or settings.evals
        self.claim_overlap_threshold = self.eval_settings.claim_overlap_threshold
        self.inference_threshold_ratio = self.eval_settings.inference_threshold_ratio

        # Success criteria thresholds
        self.min_claim_support_rate = self.eval_settings.min_claim_support_rate
        self.min_citation_validity_form = self.eval_settings.min_citation_validity_form

    def evaluate_case(
        self,
        base_case: BaseEvalCase,
        groundedness_case: GroundednessEvalCase,
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalResult, Optional[EvalTrace]]:
        """Evaluate groundedness for a single case.

        Args:
            base_case: Base case with query and user info
            groundedness_case: Groundedness ground truth
            user_context: Override user context

        Returns:
            Tuple of (EvalResult, EvalTrace if error or failure)
        """
        from src.rag.generate import generate

        start_time = time.perf_counter()
        user_ctx = user_context or UserContext(user_roles=base_case.user_roles)
        if not user_ctx.user_roles:
            user_ctx = self.default_user_context

        trace = EvalTrace(
            case_id=base_case.case_id,
            run_id="",
        )

        try:
            # Generate answer with citations
            gen_result = generate(
                query=base_case.query,
                k=self.k,
                use_hierarchical=self.use_hierarchical,
                user_context=user_ctx,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update trace
            trace.generated_answer = gen_result.answer
            trace.citations = [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text_snippet": c.text_snippet,
                }
                for c in gen_result.citations
            ]
            trace.stage_latencies = {"generate": latency_ms}

            metrics = {}
            details = {}

            # Get context from citations for validation
            cited_doc_ids = [c.doc_id for c in gen_result.citations]
            context_text = " ".join(c.text_snippet for c in gen_result.citations)

            # 1. Extract and classify claims
            claims = extract_claims(gen_result.answer)
            details["total_claims"] = len(claims)

            # Separate by type
            assertions = [(c, t) for c, t in claims if t == "assertion"]
            inferences = [(c, t) for c, t in claims if t == "inference"]
            generals = [(c, t) for c, t in claims if t == "general"]

            details["assertion_count"] = len(assertions)
            details["inference_count"] = len(inferences)
            details["general_count"] = len(generals)

            # 2. Check claim support for assertions (strict)
            supported_assertions = 0
            unsupported_claims = []

            for claim_text, _ in assertions:
                if claim_context_overlap(claim_text, context_text, self.claim_overlap_threshold):
                    supported_assertions += 1
                else:
                    unsupported_claims.append(claim_text[:100])  # Truncate for logging

            # Check inferences with relaxed threshold
            supported_inferences = 0
            for claim_text, _ in inferences:
                if claim_context_overlap(claim_text, context_text, self.claim_overlap_threshold * self.inference_threshold_ratio):
                    supported_inferences += 1

            # Calculate claim support rate (assertions + inferences, exclude generals)
            verifiable_claims = len(assertions) + len(inferences)
            supported_claims = supported_assertions + supported_inferences

            if verifiable_claims > 0:
                metrics["claim_support_rate"] = supported_claims / verifiable_claims
            else:
                metrics["claim_support_rate"] = 1.0  # No claims to verify

            metrics["unsupported_claim_count"] = len(unsupported_claims)
            details["unsupported_claims"] = unsupported_claims[:5]  # Top 5 for debugging

            # 3. Check for forbidden claims (hallucination indicators)
            forbidden_found = []
            for forbidden in groundedness_case.forbidden_claims:
                if forbidden.lower() in gen_result.answer.lower():
                    forbidden_found.append(forbidden)

            metrics["forbidden_claims_found"] = len(forbidden_found)
            details["forbidden_claims_found"] = forbidden_found

            # 4. Citation validity - form check
            # Need to get retrieved doc IDs (from generation result if available)
            from src.rag.retrieve import retrieve

            retrieval_results = retrieve(base_case.query, k=self.k, user_context=user_ctx)
            retrieved_doc_ids = [r.chunk.doc_id for r in retrieval_results]
            retrieved_chunks = {r.chunk.chunk_id: r.chunk.text for r in retrieval_results}

            trace.retrieved_doc_ids = retrieved_doc_ids
            trace.retrieved_chunk_ids = list(retrieved_chunks.keys())

            metrics["citation_validity_form"] = citation_validity_form(
                cited_doc_ids, retrieved_doc_ids
            )

            # 5. Citation validity - content check
            citations_for_content = [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text_snippet": c.text_snippet,
                }
                for c in gen_result.citations
            ]
            metrics["citation_validity_content"] = citation_validity_content(
                citations_for_content, retrieved_chunks, gen_result.answer
            )

            # 6. Numeric fabrication check
            metrics["numeric_fabrication_count"] = numeric_fabrication_count(
                gen_result.answer, context_text
            )

            # 7. Check expected claims are present
            expected_found = 0
            for expected in groundedness_case.expected_claims:
                if expected.lower() in gen_result.answer.lower():
                    expected_found += 1

            if groundedness_case.expected_claims:
                metrics["expected_claims_found"] = (
                    expected_found / len(groundedness_case.expected_claims)
                )
            else:
                metrics["expected_claims_found"] = 1.0

            # 8. Check expected citations
            expected_citations_found = 0
            for expected_doc in groundedness_case.expected_citations:
                if expected_doc in cited_doc_ids:
                    expected_citations_found += 1

            if groundedness_case.expected_citations:
                metrics["expected_citations_found"] = (
                    expected_citations_found / len(groundedness_case.expected_citations)
                )
            else:
                metrics["expected_citations_found"] = 1.0

            # Success criteria
            # Note: numeric_fabrication_count and forbidden_claims_found must be 0
            # (these are absolute requirements, not configurable)
            success = (
                metrics["claim_support_rate"] >= self.min_claim_support_rate
                and metrics["citation_validity_form"] >= self.min_citation_validity_form
                and metrics["numeric_fabrication_count"] == 0
                and metrics["forbidden_claims_found"] == 0
            )

            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.GROUNDEDNESS,
                    success=success,
                    metric_values=metrics,
                    details=details,
                    latency_ms=latency_ms,
                ),
                trace if not success else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.GROUNDEDNESS,
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                ),
                trace,
            )

    def evaluate_dataset(
        self,
        cases: list[tuple[BaseEvalCase, GroundednessEvalCase]],
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalRunSummary, list[EvalTrace]]:
        """Evaluate all groundedness cases.

        Args:
            cases: List of (base_case, groundedness_case) tuples
            user_context: Override user context for all cases

        Returns:
            Tuple of (EvalRunSummary, list of traces for failed cases)
        """
        run_id = f"groundedness-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        results = []
        traces = []

        for base_case, groundedness_case in cases:
            result, trace = self.evaluate_case(base_case, groundedness_case, user_context)
            results.append(result)
            if trace:
                trace.run_id = run_id
                traces.append(trace)

        duration = time.perf_counter() - start_time
        passed = sum(1 for r in results if r.success)

        # Aggregate metrics
        aggregate = {}
        metric_names = [
            "claim_support_rate",
            "unsupported_claim_count",
            "citation_validity_form",
            "citation_validity_content",
            "numeric_fabrication_count",
            "forbidden_claims_found",
            "expected_claims_found",
            "expected_citations_found",
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

        # Top failures (sorted by claim support rate ascending)
        failed_results = [
            (r, r.metric_values.get("claim_support_rate", 0))
            for r in results
            if not r.success
        ]
        failed_results.sort(key=lambda x: x[1])
        top_failures = [r[0].case_id for r in failed_results[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.GROUNDEDNESS,
                timestamp=datetime.utcnow(),
                total_cases=len(cases),
                passed_cases=passed,
                failed_cases=len(cases) - passed,
                aggregate_metrics=aggregate,
                metric_distributions={"latency_ms": latency_dist},
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "k": self.k,
                    "use_hierarchical": self.use_hierarchical,
                    "claim_overlap_threshold": self.claim_overlap_threshold,
                },
            ),
            traces,
        )
