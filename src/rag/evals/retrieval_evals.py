"""A. Retrieval Evals - Document/chunk retrieval quality evaluation.

Metrics:
- NDCG@k (primary): Graded relevance with position discount
- Recall@k: Percentage of relevant docs in top-k
- Precision@k: Percentage of top-k that are relevant
- MRR: Mean Reciprocal Rank

Supports both flat and hierarchical retrieval modes with two-stage evaluation
for hierarchical: parent-level Recall@k, then child-level Recall@k.
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from src.rag.evals.metrics import (
    calculate_percentiles,
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.rag.evals.models import (
    BaseEvalCase,
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    EvalTrace,
    RetrievalEvalCase,
)
from src.rag.models import UserContext


class RetrievalEvaluator:
    """Evaluator for retrieval quality metrics."""

    def __init__(
        self,
        k_values: Optional[list[int]] = None,
        default_user_context: Optional[UserContext] = None,
    ):
        """Initialize retrieval evaluator.

        Args:
            k_values: Values of k for @k metrics (default: [1, 3, 5, 10])
            default_user_context: Default user context for retrieval
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.default_user_context = default_user_context or UserContext(
            user_roles=["employee", "executive"]
        )

    def evaluate_case(
        self,
        base_case: BaseEvalCase,
        retrieval_case: RetrievalEvalCase,
        use_hierarchical: bool = False,
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalResult, Optional[EvalTrace]]:
        """Evaluate a single retrieval case.

        Args:
            base_case: Base case with query and user info
            retrieval_case: Retrieval-specific ground truth
            use_hierarchical: Whether to use hierarchical retrieval
            user_context: Override user context

        Returns:
            Tuple of (EvalResult, EvalTrace if error or failure)
        """
        from src.rag.retrieve import retrieve, retrieve_hierarchical

        start_time = time.perf_counter()
        user_ctx = user_context or UserContext(user_roles=base_case.user_roles)
        if not user_ctx.user_roles:
            user_ctx = self.default_user_context

        trace = EvalTrace(
            case_id=base_case.case_id,
            run_id="",  # Will be set by runner
        )

        try:
            # Perform retrieval
            max_k = max(self.k_values)

            if use_hierarchical:
                results = retrieve_hierarchical(
                    base_case.query,
                    k=max_k * 3,
                    return_parents=max_k,
                    user_context=user_ctx,
                )
                # Extract doc IDs from hierarchical results
                retrieved_doc_ids = [r.parent_chunk.doc_id for r in results]
                retrieved_chunk_ids = [r.parent_chunk.chunk_id for r in results]
                retrieval_scores = [r.aggregate_score for r in results]

                # For hierarchical, also collect child chunk IDs
                child_chunk_ids = []
                for r in results:
                    child_chunk_ids.extend([c.chunk_id for c in r.matched_children])
                trace.details = {"child_chunk_ids": child_chunk_ids}
            else:
                results = retrieve(
                    base_case.query,
                    k=max_k,
                    user_context=user_ctx,
                )
                retrieved_doc_ids = [r.chunk.doc_id for r in results]
                retrieved_chunk_ids = [r.chunk.chunk_id for r in results]
                retrieval_scores = [r.score for r in results]

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update trace
            trace.retrieved_doc_ids = retrieved_doc_ids
            trace.retrieved_chunk_ids = retrieved_chunk_ids
            trace.retrieval_scores = retrieval_scores
            trace.stage_latencies = {"retrieve": latency_ms}

            # Calculate metrics
            relevant_docs = set(retrieval_case.relevant_docs)
            relevant_chunks = (
                set(retrieval_case.relevant_chunks) if retrieval_case.relevant_chunks else None
            )
            doc_relevance_grades = retrieval_case.relevance_grades
            chunk_relevance_grades = retrieval_case.chunk_relevance_grades

            metrics = {}

            # Chunk-level evaluation takes priority if chunk ground truth is available
            if relevant_chunks and chunk_relevance_grades:
                # Chunk-level MRR (primary)
                metrics["mrr"] = mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunks)

                # Chunk-level metrics at different k values
                for k in self.k_values:
                    metrics[f"recall@{k}"] = recall_at_k(
                        retrieved_chunk_ids, relevant_chunks, k
                    )
                    metrics[f"precision@{k}"] = precision_at_k(
                        retrieved_chunk_ids, relevant_chunks, k
                    )
                    metrics[f"f1@{k}"] = f1_at_k(retrieved_chunk_ids, relevant_chunks, k)

                    # Chunk-level NDCG
                    metrics[f"ndcg@{k}"] = ndcg_at_k(
                        retrieved_chunk_ids, chunk_relevance_grades, k
                    )

                    # Also calculate doc-level metrics for comparison (prefixed)
                    if relevant_docs:
                        metrics[f"doc_recall@{k}"] = recall_at_k(
                            retrieved_doc_ids, relevant_docs, k
                        )

                # Success criteria: at least one relevant chunk in top-5
                success = any(
                    chunk_id in relevant_chunks for chunk_id in retrieved_chunk_ids[:5]
                )
            else:
                # Fall back to doc-level evaluation
                metrics["mrr"] = mean_reciprocal_rank(retrieved_doc_ids, relevant_docs)

                for k in self.k_values:
                    metrics[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, relevant_docs, k)
                    metrics[f"precision@{k}"] = precision_at_k(
                        retrieved_doc_ids, relevant_docs, k
                    )
                    metrics[f"f1@{k}"] = f1_at_k(retrieved_doc_ids, relevant_docs, k)

                    if doc_relevance_grades:
                        metrics[f"ndcg@{k}"] = ndcg_at_k(
                            retrieved_doc_ids, doc_relevance_grades, k
                        )

                # Success criteria: at least one relevant doc in top-5
                success = any(doc_id in relevant_docs for doc_id in retrieved_doc_ids[:5])

            # Hierarchical two-stage metrics
            if use_hierarchical and "child_chunk_ids" in trace.details:
                child_ids = trace.details["child_chunk_ids"]
                if relevant_chunks:
                    for k in self.k_values:
                        metrics[f"child_recall@{k}"] = recall_at_k(
                            child_ids, relevant_chunks, k
                        )

            # Build details based on evaluation mode
            details = {
                "retrieved_count": len(results),
                "retrieved_doc_ids": retrieved_doc_ids[:10],
                "retrieved_chunk_ids": retrieved_chunk_ids[:10],
            }
            if relevant_chunks and chunk_relevance_grades:
                details["eval_mode"] = "chunk"
                details["relevant_chunks_found"] = [
                    c for c in retrieved_chunk_ids if c in relevant_chunks
                ]
            else:
                details["eval_mode"] = "doc"
                details["relevant_docs_found"] = [
                    d for d in retrieved_doc_ids if d in relevant_docs
                ]

            return (
                EvalResult(
                    case_id=base_case.case_id,
                    perspective=EvalPerspective.RETRIEVAL,
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
                    perspective=EvalPerspective.RETRIEVAL,
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                ),
                trace,
            )

    def evaluate_dataset(
        self,
        cases: list[tuple[BaseEvalCase, RetrievalEvalCase]],
        use_hierarchical: bool = False,
        user_context: Optional[UserContext] = None,
    ) -> tuple[EvalRunSummary, list[EvalTrace]]:
        """Evaluate all retrieval cases and produce summary.

        Args:
            cases: List of (base_case, retrieval_case) tuples
            use_hierarchical: Whether to use hierarchical retrieval
            user_context: Override user context for all cases

        Returns:
            Tuple of (EvalRunSummary, list of traces for failed cases)
        """
        run_id = f"retrieval-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        results = []
        traces = []

        for base_case, retrieval_case in cases:
            result, trace = self.evaluate_case(
                base_case, retrieval_case, use_hierarchical, user_context
            )
            results.append(result)
            if trace:
                trace.run_id = run_id
                traces.append(trace)

        duration = time.perf_counter() - start_time
        passed = sum(1 for r in results if r.success)

        # Aggregate metrics
        aggregate = {}
        all_latencies = []

        # Collect all metric names from first result
        if results and results[0].metric_values:
            metric_names = list(results[0].metric_values.keys())
            for metric_name in metric_names:
                values = [
                    r.metric_values.get(metric_name)
                    for r in results
                    if r.metric_values.get(metric_name) is not None
                ]
                if values:
                    aggregate[f"mean_{metric_name}"] = sum(values) / len(values)

        for r in results:
            if r.latency_ms:
                all_latencies.append(r.latency_ms)

        latency_dist = calculate_percentiles(all_latencies)

        # Top failures
        failed_results = [(r, r.metric_values.get("mrr", 0)) for r in results if not r.success]
        failed_results.sort(key=lambda x: x[1])  # Sort by MRR ascending (worst first)
        top_failures = [r[0].case_id for r in failed_results[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.RETRIEVAL,
                timestamp=datetime.utcnow(),
                total_cases=len(cases),
                passed_cases=passed,
                failed_cases=len(cases) - passed,
                aggregate_metrics=aggregate,
                metric_distributions={"latency_ms": latency_dist},
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "k_values": self.k_values,
                    "use_hierarchical": use_hierarchical,
                },
            ),
            traces,
        )
