"""Evaluation runner - orchestrates evaluation across perspectives.

Supports:
- Selective perspective execution
- Suite modes (smoke vs full)
- Trace storage for failed cases
- Fixture loading from tests/fixtures/evals/
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.rag.evals.models import (
    BaseEvalCase,
    ContextQualityEvalCase,
    EvalPerspective,
    EvalRunSummary,
    EvalTrace,
    GoldFact,
    GroundednessEvalCase,
    PipelineEvalCase,
    PipelineOutcome,
    QueryType,
    RetrievalEvalCase,
)


class EvalRunner:
    """Orchestrates evaluation runs across perspectives."""

    # Default fixture paths
    FIXTURES_DIR = Path("tests/fixtures/evals")

    def __init__(
        self,
        fixtures_dir: Optional[Path] = None,
        traces_dir: Optional[Path] = None,
    ):
        """Initialize evaluation runner.

        Args:
            fixtures_dir: Directory containing evaluation fixtures
            traces_dir: Directory to save traces
        """
        self.fixtures_dir = fixtures_dir or self.FIXTURES_DIR
        self.traces_dir = traces_dir or Path("traces")

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load JSONL file."""
        if not path.exists():
            return []

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _load_base_cases(self, suite: str = "full") -> dict[str, BaseEvalCase]:
        """Load base cases from cases.jsonl.

        Args:
            suite: "smoke" for minimal cases, "full" for all

        Returns:
            Dict mapping case_id to BaseEvalCase
        """
        cases_path = self.fixtures_dir / "cases.jsonl"
        raw_cases = self._load_jsonl(cases_path)

        cases = {}
        for i, raw in enumerate(raw_cases):
            # For smoke suite, only take first 5 cases
            if suite == "smoke" and i >= 5:
                break

            case = BaseEvalCase(
                case_id=raw.get("case_id", f"case-{i}"),
                query=raw["query"],
                user_roles=raw.get("user_roles", ["employee"]),
                query_type=QueryType(raw.get("query_type", "faq")),
            )
            cases[case.case_id] = case

        return cases

    def _load_retrieval_labels(self) -> dict[str, RetrievalEvalCase]:
        """Load retrieval labels."""
        labels_path = self.fixtures_dir / "retrieval_labels.jsonl"
        raw_labels = self._load_jsonl(labels_path)

        labels = {}
        for raw in raw_labels:
            case = RetrievalEvalCase(
                case_id=raw["case_id"],
                relevant_docs=raw.get("relevant_docs", []),
                relevant_chunks=raw.get("relevant_chunks", []),
                relevance_grades=raw.get("relevance_grades", {}),
            )
            labels[case.case_id] = case

        return labels

    def _load_context_labels(self) -> dict[str, ContextQualityEvalCase]:
        """Load context quality labels."""
        labels_path = self.fixtures_dir / "context_labels.jsonl"
        raw_labels = self._load_jsonl(labels_path)

        labels = {}
        for raw in raw_labels:
            gold_facts = []
            for gf in raw.get("gold_facts", []):
                gold_facts.append(
                    GoldFact(
                        fact=gf["fact"],
                        aliases=gf.get("aliases", []),
                    )
                )

            case = ContextQualityEvalCase(
                case_id=raw["case_id"],
                gold_facts=gold_facts,
                expected_chunks=raw.get("expected_chunks", []),
            )
            labels[case.case_id] = case

        return labels

    def _load_groundedness_labels(self) -> dict[str, GroundednessEvalCase]:
        """Load groundedness labels."""
        labels_path = self.fixtures_dir / "groundedness_labels.jsonl"
        raw_labels = self._load_jsonl(labels_path)

        labels = {}
        for raw in raw_labels:
            case = GroundednessEvalCase(
                case_id=raw["case_id"],
                expected_claims=raw.get("expected_claims", []),
                expected_citations=raw.get("expected_citations", []),
                forbidden_claims=raw.get("forbidden_claims", []),
            )
            labels[case.case_id] = case

        return labels

    def _load_pipeline_labels(self) -> dict[str, PipelineEvalCase]:
        """Load pipeline labels."""
        labels_path = self.fixtures_dir / "pipeline_labels.jsonl"
        raw_labels = self._load_jsonl(labels_path)

        labels = {}
        for raw in raw_labels:
            case = PipelineEvalCase(
                case_id=raw["case_id"],
                expected_outcome=PipelineOutcome(raw.get("expected_outcome", "success")),
                required_flags=raw.get("required_flags", []),
                forbidden_flags=raw.get("forbidden_flags", []),
                min_citations=raw.get("min_citations", 0),
                latency_budget_ms=raw.get("latency_budget_ms", {}),
            )
            labels[case.case_id] = case

        return labels

    def _save_traces(self, run_id: str, traces: list[EvalTrace]) -> Path:
        """Save traces to file.

        Args:
            run_id: Run identifier
            traces: List of traces to save

        Returns:
            Path to saved file
        """
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.traces_dir / f"{run_id}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(trace.model_dump_json() + "\n")

        return output_path

    def run(
        self,
        perspectives: Optional[list[str]] = None,
        suite: str = "full",
        save_trace: bool = False,
        verbose: bool = False,
    ) -> list[EvalRunSummary]:
        """Run evaluations.

        Args:
            perspectives: List of perspectives to run (default: all)
            suite: "smoke" for minimal cases, "full" for all
            save_trace: Whether to save traces for failed cases
            verbose: Whether to print progress

        Returns:
            List of EvalRunSummary for each perspective
        """
        from src.rag.evals.context_quality_evals import ContextQualityEvaluator
        from src.rag.evals.groundedness_evals import GroundednessEvaluator
        from src.rag.evals.pipeline_evals import PipelineEvaluator
        from src.rag.evals.retrieval_evals import RetrievalEvaluator
        from src.rag.evals.safety_evals import SafetyEvaluator

        # Determine which perspectives to run
        all_perspectives = ["retrieval", "context", "groundedness", "safety", "pipeline"]
        if perspectives is None or "all" in perspectives:
            perspectives = all_perspectives
        else:
            # Validate perspectives
            perspectives = [p for p in perspectives if p in all_perspectives]

        summaries = []
        all_traces = []
        master_run_id = f"eval-{uuid.uuid4().hex[:8]}"

        if verbose:
            print(f"Starting evaluation run: {master_run_id}")
            print(f"Perspectives: {perspectives}")
            print(f"Suite: {suite}")
            print()

        # Load base cases
        base_cases = self._load_base_cases(suite)

        if verbose:
            print(f"Loaded {len(base_cases)} base cases")

        # Run each perspective
        if "retrieval" in perspectives:
            if verbose:
                print("Running retrieval evaluation...")

            retrieval_labels = self._load_retrieval_labels()
            cases = [
                (base_cases[case_id], label)
                for case_id, label in retrieval_labels.items()
                if case_id in base_cases
            ]

            if cases:
                evaluator = RetrievalEvaluator()
                summary, traces = evaluator.evaluate_dataset(cases)
                summaries.append(summary)
                all_traces.extend(traces)

                if verbose:
                    print(f"  Passed: {summary.passed_cases}/{summary.total_cases}")
            elif verbose:
                print("  No retrieval cases found")

        if "context" in perspectives:
            if verbose:
                print("Running context quality evaluation...")

            context_labels = self._load_context_labels()
            cases = [
                (base_cases[case_id], label)
                for case_id, label in context_labels.items()
                if case_id in base_cases
            ]

            if cases:
                evaluator = ContextQualityEvaluator()
                summary, traces = evaluator.evaluate_dataset(cases)
                summaries.append(summary)
                all_traces.extend(traces)

                if verbose:
                    print(f"  Passed: {summary.passed_cases}/{summary.total_cases}")
            elif verbose:
                print("  No context quality cases found")

        if "groundedness" in perspectives:
            if verbose:
                print("Running groundedness evaluation...")

            groundedness_labels = self._load_groundedness_labels()
            cases = [
                (base_cases[case_id], label)
                for case_id, label in groundedness_labels.items()
                if case_id in base_cases
            ]

            if cases:
                evaluator = GroundednessEvaluator()
                summary, traces = evaluator.evaluate_dataset(cases)
                summaries.append(summary)
                all_traces.extend(traces)

                if verbose:
                    print(f"  Passed: {summary.passed_cases}/{summary.total_cases}")
            elif verbose:
                print("  No groundedness cases found")

        if "safety" in perspectives:
            if verbose:
                print("Running safety evaluation...")

            evaluator = SafetyEvaluator()
            safety_summaries, _ = evaluator.evaluate_all()
            summaries.extend(safety_summaries)

            if verbose:
                for s in safety_summaries:
                    print(f"  {s.config.get('eval_type', 'safety')}: {s.passed_cases}/{s.total_cases}")

        if "pipeline" in perspectives:
            if verbose:
                print("Running pipeline evaluation...")

            pipeline_labels = self._load_pipeline_labels()
            cases = [
                (base_cases[case_id], label)
                for case_id, label in pipeline_labels.items()
                if case_id in base_cases
            ]

            if cases:
                evaluator = PipelineEvaluator()
                summary, traces = evaluator.evaluate_dataset(cases)
                summaries.append(summary)
                all_traces.extend(traces)

                if verbose:
                    print(f"  Passed: {summary.passed_cases}/{summary.total_cases}")
            elif verbose:
                print("  No pipeline cases found")

        # Save traces if requested
        if save_trace and all_traces:
            trace_path = self._save_traces(master_run_id, all_traces)
            if verbose:
                print(f"\nTraces saved to: {trace_path}")

        if verbose:
            print("\nEvaluation complete!")
            total_passed = sum(s.passed_cases for s in summaries)
            total_cases = sum(s.total_cases for s in summaries)
            print(f"Overall: {total_passed}/{total_cases} passed")

        return summaries
