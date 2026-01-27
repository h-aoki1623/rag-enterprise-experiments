"""Report generation for evaluation results.

Generates:
- JSON reports with full metric details
- Markdown reports with human-readable summaries
- Diagnostic guidance based on perspective results
- Regression detection when baseline provided
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.rag.evals.models import EvalPerspective, EvalRunSummary


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    # Diagnostic messages based on pass/fail patterns
    DIAGNOSTICS = {
        ("retrieval_fail", "groundedness_pass"): (
            "Retrieval is finding wrong documents but answer generation is working. "
            "→ Investigate embedding model or chunking strategy."
        ),
        ("retrieval_pass", "groundedness_fail"): (
            "Retrieval is good but answers aren't grounded in context. "
            "→ Investigate prompts or citation extraction."
        ),
        ("context_fail", "groundedness_fail"): (
            "Context quality and groundedness both failing. "
            "→ Likely a chunking issue causing fragmented or redundant context."
        ),
        ("safety_fail",): (
            "Safety checks failing. "
            "→ Review guardrail thresholds and attack patterns."
        ),
        ("pipeline_fail",): (
            "Pipeline integration failing. "
            "→ Check latency budgets and policy flag logic."
        ),
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or Path("reports/evals")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _determine_diagnosis(self, summaries: list[EvalRunSummary]) -> list[str]:
        """Determine diagnostic messages based on results.

        Args:
            summaries: List of evaluation summaries

        Returns:
            List of diagnostic messages
        """
        # Build pass/fail status for each perspective
        status = {}
        for summary in summaries:
            perspective = summary.perspective.value
            pass_rate = summary.passed_cases / summary.total_cases if summary.total_cases > 0 else 0
            status[perspective] = "pass" if pass_rate >= 0.8 else "fail"

        diagnostics = []

        # Check patterns
        if status.get("retrieval") == "fail" and status.get("groundedness") == "pass":
            diagnostics.append(self.DIAGNOSTICS[("retrieval_fail", "groundedness_pass")])

        if status.get("retrieval") == "pass" and status.get("groundedness") == "fail":
            diagnostics.append(self.DIAGNOSTICS[("retrieval_pass", "groundedness_fail")])

        if status.get("context_quality") == "fail" and status.get("groundedness") == "fail":
            diagnostics.append(self.DIAGNOSTICS[("context_fail", "groundedness_fail")])

        if status.get("safety") == "fail":
            diagnostics.append(self.DIAGNOSTICS[("safety_fail",)])

        if status.get("pipeline") == "fail":
            diagnostics.append(self.DIAGNOSTICS[("pipeline_fail",)])

        if not diagnostics:
            if all(v == "pass" for v in status.values()):
                diagnostics.append("All perspectives passing. System is performing well.")
            else:
                diagnostics.append("Mixed results. Review individual perspective failures.")

        return diagnostics

    def _compute_regression(
        self,
        current: list[EvalRunSummary],
        baseline: list[EvalRunSummary],
    ) -> dict[str, dict[str, float]]:
        """Compute regression/improvement from baseline.

        Args:
            current: Current evaluation summaries
            baseline: Baseline evaluation summaries

        Returns:
            Dict mapping perspective to metric deltas
        """
        baseline_by_perspective = {s.perspective.value: s for s in baseline}
        regression = {}

        for summary in current:
            perspective = summary.perspective.value
            if perspective not in baseline_by_perspective:
                continue

            baseline_summary = baseline_by_perspective[perspective]
            deltas = {}

            # Compare aggregate metrics
            for metric, value in summary.aggregate_metrics.items():
                baseline_value = baseline_summary.aggregate_metrics.get(metric, 0)
                delta = value - baseline_value
                if abs(delta) > 0.01:  # Only report significant changes
                    deltas[metric] = delta

            # Compare pass rate
            current_rate = summary.passed_cases / summary.total_cases if summary.total_cases > 0 else 0
            baseline_rate = (
                baseline_summary.passed_cases / baseline_summary.total_cases
                if baseline_summary.total_cases > 0
                else 0
            )
            rate_delta = current_rate - baseline_rate
            if abs(rate_delta) > 0.01:
                deltas["pass_rate"] = rate_delta

            if deltas:
                regression[perspective] = deltas

        return regression

    def generate_json_report(
        self,
        summaries: list[EvalRunSummary],
        report_name: Optional[str] = None,
        baseline: Optional[list[EvalRunSummary]] = None,
    ) -> Path:
        """Generate JSON report from summaries.

        Args:
            summaries: List of evaluation summaries
            report_name: Optional report name
            baseline: Optional baseline summaries for regression

        Returns:
            Path to generated report
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"eval_report_{timestamp}"

        # Compute overall stats
        total_cases = sum(s.total_cases for s in summaries)
        total_passed = sum(s.passed_cases for s in summaries)
        total_duration = sum(s.duration_seconds for s in summaries)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_name": report_name,
            "overall": {
                "total_cases": total_cases,
                "total_passed": total_passed,
                "pass_rate": total_passed / total_cases if total_cases > 0 else 0,
                "total_duration_seconds": total_duration,
            },
            "diagnostics": self._determine_diagnosis(summaries),
            "summaries": [s.model_dump() for s in summaries],
        }

        # Add regression if baseline provided
        if baseline:
            report["regression"] = self._compute_regression(summaries, baseline)

        output_path = self.output_dir / f"{report_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def generate_markdown_report(
        self,
        summaries: list[EvalRunSummary],
        report_name: Optional[str] = None,
        baseline: Optional[list[EvalRunSummary]] = None,
    ) -> Path:
        """Generate Markdown report from summaries.

        Args:
            summaries: List of evaluation summaries
            report_name: Optional report name
            baseline: Optional baseline summaries for regression

        Returns:
            Path to generated report
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"eval_report_{timestamp}"

        lines = [
            "# Evaluation Report",
            "",
            f"**Generated:** {datetime.utcnow().isoformat()}",
            "",
        ]

        # Overall Summary
        total_cases = sum(s.total_cases for s in summaries)
        total_passed = sum(s.passed_cases for s in summaries)
        pass_rate = total_passed / total_cases * 100 if total_cases > 0 else 0

        lines.extend([
            "## Overall Summary",
            "",
            f"- **Total Cases:** {total_cases}",
            f"- **Passed:** {total_passed}",
            f"- **Pass Rate:** {pass_rate:.1f}%",
            "",
        ])

        # Perspective Summary Table
        lines.extend([
            "## Perspective Summary",
            "",
            "| Perspective | Total | Passed | Failed | Pass Rate | Duration |",
            "|-------------|-------|--------|--------|-----------|----------|",
        ])

        for s in summaries:
            rate = s.passed_cases / s.total_cases * 100 if s.total_cases else 0
            lines.append(
                f"| {s.perspective.value} | {s.total_cases} | {s.passed_cases} | "
                f"{s.failed_cases} | {rate:.1f}% | {s.duration_seconds:.2f}s |"
            )

        lines.append("")

        # Diagnostics
        diagnostics = self._determine_diagnosis(summaries)
        lines.extend([
            "## Diagnostics",
            "",
        ])
        for diag in diagnostics:
            lines.append(f"- {diag}")
        lines.append("")

        # Regression (if baseline provided)
        if baseline:
            regression = self._compute_regression(summaries, baseline)
            if regression:
                lines.extend([
                    "## Regression Analysis",
                    "",
                ])
                for perspective, deltas in regression.items():
                    lines.append(f"### {perspective}")
                    for metric, delta in deltas.items():
                        sign = "+" if delta > 0 else ""
                        emoji = "🟢" if delta > 0 else "🔴"
                        lines.append(f"- {metric}: {sign}{delta:.4f} {emoji}")
                    lines.append("")

        # Detailed Metrics
        lines.extend([
            "## Detailed Metrics",
            "",
        ])

        for s in summaries:
            lines.extend([
                f"### {s.perspective.value.replace('_', ' ').title()}",
                "",
                f"- **Run ID:** `{s.run_id}`",
                f"- **Duration:** {s.duration_seconds:.2f}s",
                "",
            ])

            if s.aggregate_metrics:
                lines.append("**Aggregate Metrics:**")
                lines.append("")
                for metric, value in sorted(s.aggregate_metrics.items()):
                    if isinstance(value, float):
                        lines.append(f"- {metric}: {value:.4f}")
                    else:
                        lines.append(f"- {metric}: {value}")
                lines.append("")

            if s.top_failures:
                lines.append("**Top Failures:**")
                lines.append("")
                for case_id in s.top_failures[:5]:
                    lines.append(f"- `{case_id}`")
                lines.append("")

            if s.category_breakdown:
                lines.append("**Category Breakdown:**")
                lines.append("")
                for cat, stats in s.category_breakdown.items():
                    lines.append(f"- {cat}: {stats}")
                lines.append("")

        output_path = self.output_dir / f"{report_name}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def load_baseline(self, baseline_path: Path) -> list[EvalRunSummary]:
        """Load baseline report for regression comparison.

        Args:
            baseline_path: Path to baseline JSON report

        Returns:
            List of EvalRunSummary from baseline
        """
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summaries = []
        for raw in data.get("summaries", []):
            summary = EvalRunSummary(
                run_id=raw["run_id"],
                perspective=EvalPerspective(raw["perspective"]),
                timestamp=datetime.fromisoformat(raw["timestamp"]),
                total_cases=raw["total_cases"],
                passed_cases=raw["passed_cases"],
                failed_cases=raw["failed_cases"],
                aggregate_metrics=raw.get("aggregate_metrics", {}),
                metric_distributions=raw.get("metric_distributions", {}),
                category_breakdown=raw.get("category_breakdown", {}),
                top_failures=raw.get("top_failures", []),
                duration_seconds=raw["duration_seconds"],
                config=raw.get("config", {}),
            )
            summaries.append(summary)

        return summaries
