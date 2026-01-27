"""E. Safety Evals - Guardrail effectiveness evaluation.

Metrics (operational threshold based):
- AUC: Area under ROC curve (target > 0.85)
- TPR@FPR=1%: Detection rate at 1% FPR (enterprise standard, target > 0.7)
- TPR@FPR=5%: Detection rate at 5% FPR (target > 0.85)
- Precision@threshold: Precision at operational threshold
- Recall@threshold: Recall at operational threshold

Evaluates both:
- Input guardrail (injection detection)
- Output guardrail (leakage detection)

Reuses existing fixtures from tests/fixtures/benign/ and tests/fixtures/attacks/
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.rag.evals.metrics import (
    auc_from_points,
    calculate_percentiles,
    precision_recall_at_threshold,
    roc_curve_points,
    tpr_at_fpr,
)
from src.rag.evals.models import (
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    SafetyEvalCase,
)
from src.rag.guardrails import GuardrailAction, InputGuardrail, OutputGuardrail
from src.rag.models import Classification


class SafetyEvaluator:
    """Evaluator for guardrail effectiveness metrics."""

    # Default fixture paths
    DEFAULT_BENIGN_QUERIES = Path("tests/fixtures/benign/queries.json")
    DEFAULT_ATTACK_QUERIES = Path("tests/fixtures/attacks/injection_queries.json")
    DEFAULT_LEAKAGE_OUTPUTS = Path("tests/fixtures/attacks/leakage_outputs.json")

    def __init__(
        self,
        guardrail_settings: Optional["GuardrailSettings"] = None,
        benign_queries_path: Optional[Path] = None,
        attack_queries_path: Optional[Path] = None,
        leakage_outputs_path: Optional[Path] = None,
    ):
        """Initialize safety evaluator.

        Args:
            guardrail_settings: Settings for guardrails
            benign_queries_path: Path to benign queries fixture
            attack_queries_path: Path to attack queries fixture
            leakage_outputs_path: Path to leakage outputs fixture
        """
        from src.rag.config import settings

        self.guardrail_settings = guardrail_settings or settings.guardrails
        self.input_guardrail = InputGuardrail(self.guardrail_settings)
        self.output_guardrail = OutputGuardrail(self.guardrail_settings)

        self.benign_queries_path = benign_queries_path or self.DEFAULT_BENIGN_QUERIES
        self.attack_queries_path = attack_queries_path or self.DEFAULT_ATTACK_QUERIES
        self.leakage_outputs_path = leakage_outputs_path or self.DEFAULT_LEAKAGE_OUTPUTS

    def _load_json_fixture(self, path: Path) -> dict:
        """Load JSON fixture file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_input_guardrail(
        self,
        benign_cases: Optional[list[SafetyEvalCase]] = None,
        attack_cases: Optional[list[SafetyEvalCase]] = None,
    ) -> tuple[EvalRunSummary, list[EvalResult]]:
        """Evaluate input guardrail (injection detection).

        Args:
            benign_cases: Benign query cases (loads from fixture if None)
            attack_cases: Attack query cases (loads from fixture if None)

        Returns:
            Tuple of (EvalRunSummary, list of individual results)
        """
        run_id = f"safety-input-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Load cases from fixtures if not provided
        if benign_cases is None:
            if self.benign_queries_path.exists():
                data = self._load_json_fixture(self.benign_queries_path)
                benign_cases = [
                    SafetyEvalCase(
                        case_id=q.get("id", f"benign-{i}"),
                        text=q["text"],
                        is_attack=False,
                        category=q.get("category", "benign"),
                        expected_detection=False,
                    )
                    for i, q in enumerate(data.get("queries", []))
                ]
            else:
                benign_cases = []

        if attack_cases is None:
            if self.attack_queries_path.exists():
                data = self._load_json_fixture(self.attack_queries_path)
                attack_cases = [
                    SafetyEvalCase(
                        case_id=q.get("id", f"attack-{i}"),
                        text=q["text"],
                        is_attack=True,
                        category=q.get("category", "unknown"),
                        expected_detection=q.get("expected_detection", True),
                    )
                    for i, q in enumerate(data.get("queries", []))
                ]
            else:
                attack_cases = []

        # Evaluate all cases
        all_scores = []
        all_labels = []  # True = attack, False = benign
        results = []
        category_stats: dict[str, dict[str, int]] = {}

        # Get operational thresholds for detection criteria
        warn_thresh = self.guardrail_settings.injection_warn_threshold
        block_thresh = self.guardrail_settings.injection_block_threshold

        # Process benign cases
        for case in benign_cases:
            result = self.input_guardrail.check(case.text)
            all_scores.append(result.threat_score)
            all_labels.append(False)

            success = result.action != GuardrailAction.BLOCK
            results.append(
                EvalResult(
                    case_id=case.case_id,
                    perspective=EvalPerspective.SAFETY,
                    success=success,
                    metric_values={
                        "threat_score": result.threat_score,
                        "blocked": 1.0 if result.action == GuardrailAction.BLOCK else 0.0,
                    },
                    details={
                        "action": result.action.value,
                        "category": case.category,
                        "is_attack": False,
                    },
                )
            )

        # Process attack cases
        for case in attack_cases:
            if not case.expected_detection:
                continue

            result = self.input_guardrail.check(case.text)
            all_scores.append(result.threat_score)
            all_labels.append(True)

            # For attacks, success = detected (action is WARN or BLOCK)
            detected = result.action != GuardrailAction.ALLOW
            blocked = result.action == GuardrailAction.BLOCK
            success = detected

            # Update category stats
            if case.category not in category_stats:
                category_stats[case.category] = {"total": 0, "detected": 0, "blocked": 0}
            category_stats[case.category]["total"] += 1
            if detected:
                category_stats[case.category]["detected"] += 1
            if blocked:
                category_stats[case.category]["blocked"] += 1

            results.append(
                EvalResult(
                    case_id=case.case_id,
                    perspective=EvalPerspective.SAFETY,
                    success=success,
                    metric_values={
                        "threat_score": result.threat_score,
                        "blocked": 1.0 if result.action == GuardrailAction.BLOCK else 0.0,
                    },
                    details={
                        "action": result.action.value,
                        "category": case.category,
                        "is_attack": True,
                        "score_breakdown": result.score_breakdown,
                    },
                )
            )

        duration = time.perf_counter() - start_time

        # Calculate ROC curve and metrics
        roc_points = roc_curve_points(all_scores, all_labels)
        auc_score = auc_from_points(roc_points)

        # TPR at specific FPR levels
        tpr_at_fpr_1 = tpr_at_fpr(roc_points, 0.01)
        tpr_at_fpr_5 = tpr_at_fpr(roc_points, 0.05)

        # Metrics at operational thresholds (thresholds defined earlier)
        precision_block, recall_block = precision_recall_at_threshold(
            all_scores, all_labels, block_thresh
        )
        precision_warn, recall_warn = precision_recall_at_threshold(
            all_scores, all_labels, warn_thresh
        )

        # FPR and block rates
        benign_blocked = sum(
            1 for r in results
            if not r.details.get("is_attack") and r.metric_values.get("blocked", 0) > 0
        )
        benign_total = len(benign_cases)
        benign_block_rate = benign_blocked / benign_total if benign_total > 0 else 0

        attack_total = sum(1 for c in attack_cases if c.expected_detection)
        attack_blocked = sum(
            1 for r in results
            if r.details.get("is_attack") and r.metric_values.get("blocked", 0) > 0
        )
        attack_block_rate = attack_blocked / attack_total if attack_total > 0 else 0

        # Detection rate (action != ALLOW, i.e., WARN or BLOCK)
        attack_detected = sum(
            1 for r in results
            if r.details.get("is_attack") and r.success  # success = detected for attacks
        )
        detection_rate = attack_detected / attack_total if attack_total > 0 else 0

        passed = sum(1 for r in results if r.success)

        # Category breakdown with rates
        category_breakdown = {
            cat: {
                "total": stats["total"],
                "detection_rate": stats["detected"] / stats["total"] if stats["total"] > 0 else 0,
                "block_rate": stats["blocked"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for cat, stats in category_stats.items()
        }

        # Top failures (attacks not detected)
        failed_attacks = [
            (r, r.metric_values.get("threat_score", 0))
            for r in results
            if r.details.get("is_attack") and not r.success
        ]
        failed_attacks.sort(key=lambda x: x[1])
        top_failures = [r[0].case_id for r in failed_attacks[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.SAFETY,
                timestamp=datetime.utcnow(),
                total_cases=len(results),
                passed_cases=passed,
                failed_cases=len(results) - passed,
                aggregate_metrics={
                    "auc": auc_score,
                    "tpr_at_fpr_1pct": tpr_at_fpr_1,
                    "tpr_at_fpr_5pct": tpr_at_fpr_5,
                    "precision_at_block": precision_block,
                    "recall_at_block": recall_block,
                    "precision_at_warn": precision_warn,
                    "recall_at_warn": recall_warn,
                    "benign_block_rate": benign_block_rate,
                    "attack_block_rate": attack_block_rate,
                    "detection_rate": detection_rate,
                },
                metric_distributions={
                    "threat_scores": calculate_percentiles(all_scores),
                },
                category_breakdown=category_breakdown,
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "eval_type": "input_guardrail",
                    "block_threshold": block_thresh,
                    "warn_threshold": warn_thresh,
                    "benign_count": len(benign_cases),
                    "attack_count": attack_total,
                },
            ),
            results,
        )

    def evaluate_output_guardrail(
        self,
        leakage_cases: Optional[list[dict]] = None,
    ) -> tuple[EvalRunSummary, list[EvalResult]]:
        """Evaluate output guardrail (leakage detection).

        Args:
            leakage_cases: Leakage cases (loads from fixture if None)

        Returns:
            Tuple of (EvalRunSummary, list of individual results)
        """
        run_id = f"safety-output-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Load cases from fixture if not provided
        if leakage_cases is None:
            if self.leakage_outputs_path.exists():
                data = self._load_json_fixture(self.leakage_outputs_path)
                leakage_cases = data.get("cases", [])
            else:
                leakage_cases = []

        results = []
        type_stats: dict[str, dict[str, int]] = {}

        for i, case in enumerate(leakage_cases):
            case_id = case.get("id", f"leakage-{i}")
            output = case["output"]
            contexts = case.get("contexts", [])
            metadata = case.get("metadata", [])
            expected_detection = case.get("expected_detection", True)
            leak_type = case.get("leak_type", "unknown")

            classification_str = (
                metadata[0].get("classification", "internal") if metadata else "internal"
            )
            classification = Classification(classification_str)

            result = self.output_guardrail.check(output, contexts, metadata, classification)

            # Track type stats
            if leak_type not in type_stats:
                type_stats[leak_type] = {"total": 0, "detected": 0, "blocked": 0}
            type_stats[leak_type]["total"] += 1

            # Detection based on action (WARN or BLOCK = detected)
            detected = result.action != GuardrailAction.ALLOW
            blocked = result.action == GuardrailAction.BLOCK

            if detected:
                type_stats[leak_type]["detected"] += 1
            if blocked:
                type_stats[leak_type]["blocked"] += 1

            # Success depends on whether detection was expected
            if expected_detection:
                success = detected
            else:
                # For non-leaky cases, success = not blocked
                success = not blocked

            # Get sanitize_needed from details (set by OutputGuardrail)
            sanitize_needed = result.details.get("sanitize_needed", False)

            results.append(
                EvalResult(
                    case_id=case_id,
                    perspective=EvalPerspective.SAFETY,
                    success=success,
                    metric_values={
                        "threat_score": result.threat_score,
                        "blocked": 1.0 if blocked else 0.0,
                        "sanitize_needed": 1.0 if sanitize_needed else 0.0,
                    },
                    details={
                        "action": result.action.value,
                        "leak_type": leak_type,
                        "expected_detection": expected_detection,
                        "classification": classification_str,
                    },
                )
            )

        duration = time.perf_counter() - start_time

        # Calculate aggregate metrics
        positive_cases = [c for c in leakage_cases if c.get("expected_detection", True)]
        negative_cases = [c for c in leakage_cases if not c.get("expected_detection", True)]

        # Detection based on success (action != ALLOW for expected_detection cases)
        detected_count = sum(
            1 for r in results
            if r.details.get("expected_detection") and r.success
        )
        detection_rate = detected_count / len(positive_cases) if positive_cases else 0

        false_positives = sum(
            1 for r in results
            if not r.details.get("expected_detection") and r.metric_values.get("blocked", 0) > 0
        )
        fpr = false_positives / len(negative_cases) if negative_cases else 0

        passed = sum(1 for r in results if r.success)

        # Category breakdown
        category_breakdown = {
            leak_type: {
                "total": stats["total"],
                "detection_rate": stats["detected"] / stats["total"] if stats["total"] > 0 else 0,
                "block_rate": stats["blocked"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for leak_type, stats in type_stats.items()
        }

        # Top failures
        failed_results = [
            (r, r.metric_values.get("threat_score", 0))
            for r in results
            if not r.success
        ]
        failed_results.sort(key=lambda x: x[1])
        top_failures = [r[0].case_id for r in failed_results[:10]]

        return (
            EvalRunSummary(
                run_id=run_id,
                perspective=EvalPerspective.SAFETY,
                timestamp=datetime.utcnow(),
                total_cases=len(results),
                passed_cases=passed,
                failed_cases=len(results) - passed,
                aggregate_metrics={
                    "detection_rate": detection_rate,
                    "false_positive_rate": fpr,
                },
                metric_distributions={},
                category_breakdown=category_breakdown,
                top_failures=top_failures,
                duration_seconds=duration,
                config={
                    "eval_type": "output_guardrail",
                    "leakage_thresholds": self.guardrail_settings.leakage_thresholds,
                    "positive_cases": len(positive_cases),
                    "negative_cases": len(negative_cases),
                },
            ),
            results,
        )

    def evaluate_all(
        self,
    ) -> tuple[list[EvalRunSummary], list[EvalResult]]:
        """Run all safety evaluations.

        Returns:
            Tuple of (list of summaries, list of all results)
        """
        summaries = []
        all_results = []

        # Input guardrail
        input_summary, input_results = self.evaluate_input_guardrail()
        summaries.append(input_summary)
        all_results.extend(input_results)

        # Output guardrail
        output_summary, output_results = self.evaluate_output_guardrail()
        summaries.append(output_summary)
        all_results.extend(output_results)

        return summaries, all_results
