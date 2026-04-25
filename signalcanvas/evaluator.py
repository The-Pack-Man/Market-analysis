"""Groundedness evaluator for SignalCanvas answers."""

from __future__ import annotations

from typing import Any

from signalcanvas.models import ChartSpec, EvaluatorResult


def evaluate_groundedness(
    question: str,
    sql_used: str | None,
    result_summary: dict[str, Any],
    chart_spec: ChartSpec | None,
    narrative: str,
) -> EvaluatorResult:
    """Run lightweight rule-based groundedness checks."""
    issues: list[str] = []
    score = 1.0
    lowered = narrative.lower()

    if not sql_used:
        issues.append("No SQL was recorded for this answer.")
        score -= 0.2

    if chart_spec and chart_spec.y.lower().startswith("rolling_vol") and "volatil" not in lowered:
        issues.append("The chart uses volatility data but the narrative does not explain it.")
        score -= 0.1

    if "because" in lowered or "caused by" in lowered:
        issues.append("Narrative may overclaim causation relative to the retrieved data.")
        score -= 0.15

    if "possible external context" in lowered and "source:" not in lowered:
        issues.append("External context was mentioned without source attribution.")
        score -= 0.15

    if result_summary.get("row_count", 0) == 0:
        issues.append("The answer is based on an empty result set.")
        score -= 0.35

    if "forecast" in question.lower() and "uncertain" not in lowered and "estimate" not in lowered:
        issues.append("Forecast language should explicitly mention uncertainty.")
        score -= 0.2

    grounded = score >= 0.6
    return EvaluatorResult(score=max(0.0, round(score, 2)), issues=issues, grounded=grounded)
