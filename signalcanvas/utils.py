"""Utility helpers for SignalCanvas."""

from __future__ import annotations

import re
import uuid
from datetime import date


SYMBOL_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
COMMON_NON_SYMBOL_WORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BEST",
    "BY",
    "DAY",
    "DAYS",
    "FOR",
    "FROM",
    "HOW",
    "IN",
    "IS",
    "IT",
    "MONTH",
    "NEXT",
    "OF",
    "ON",
    "OR",
    "SHOW",
    "STOCK",
    "STOCKS",
    "THE",
    "THIS",
    "TO",
    "TWO",
    "VS",
    "WHAT",
    "WHICH",
    "YEAR",
}


def make_id() -> str:
    """Return a random identifier for logged analyses."""
    return str(uuid.uuid4())


def parse_symbols(
    question: str,
    fallback: list[str] | None = None,
    valid_symbols: list[str] | set[str] | None = None,
) -> list[str]:
    """Extract likely stock symbols from a natural-language question."""
    reserved = {"YTD", "SQL", "OHLCV"}
    valid_lookup = {symbol.upper() for symbol in valid_symbols} if valid_symbols else None
    symbols = []
    for match in SYMBOL_PATTERN.finditer(question.upper()):
        candidate = match.group(0)
        if candidate in reserved or candidate in COMMON_NON_SYMBOL_WORDS:
            continue
        if valid_lookup is not None and candidate not in valid_lookup:
            continue
        symbols.append(candidate)
    ordered: list[str] = []
    for symbol in symbols:
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered or (fallback or [])


def detect_analysis_type(question: str) -> str:
    """Classify the request into a small v1 taxonomy."""
    lowered = question.lower()
    if "forecast" in lowered or "predict" in lowered or "next " in lowered or "better buy" in lowered:
        return "forecast"
    if (
        "compare" in lowered
        or "vs" in lowered
        or "versus" in lowered
        or "better" in lowered
        or "outperform" in lowered
    ):
        return "compare"
    if (
        "volatile" in lowered
        or "anomaly" in lowered
        or "changed" in lowered
        or "unusual" in lowered
        or "spike" in lowered
        or "drop" in lowered
        or "explain" in lowered
        or "why" in lowered
        or "driving" in lowered
    ):
        return "anomaly"
    return "visualize"


def detect_question_focus(question: str) -> str:
    """Infer the main metric the user cares about."""
    lowered = question.lower()
    if "better buy" in lowered or ("buy" in lowered and "better" in lowered):
        return "forward_performance"
    if "why" in lowered or "driving" in lowered:
        return "explanation"
    if "spike" in lowered:
        return "spike"
    if "drop" in lowered:
        return "drop"
    if "trend" in lowered or "explain" in lowered:
        return "trend"
    if "volatil" in lowered or "risky" in lowered:
        return "volatility"
    if "drawdown" in lowered:
        return "drawdown"
    if "perform" in lowered or "return" in lowered or "better" in lowered or "outperform" in lowered:
        return "performance"
    if "forecast" in lowered or "predict" in lowered:
        return "forecast"
    return "price"


def infer_date_range(question: str, default_start: str, default_end: str) -> tuple[str, str]:
    """Infer a rough date range from the prompt."""
    lowered = question.lower()
    try:
        reference_end = date.fromisoformat(default_end)
    except ValueError:
        reference_end = date.today()
    if "ytd" in lowered or "year to date" in lowered:
        return date(reference_end.year, 1, 1).isoformat(), reference_end.isoformat()
    if "30 day" in lowered or "30 days" in lowered or "last month" in lowered or "this month" in lowered or "month" in lowered:
        return (reference_end.fromordinal(reference_end.toordinal() - 30)).isoformat(), reference_end.isoformat()
    if "10 day" in lowered or "10 days" in lowered or "past 10 days" in lowered:
        return (reference_end.fromordinal(reference_end.toordinal() - 10)).isoformat(), reference_end.isoformat()
    if "last week" in lowered or "recently" in lowered:
        return (reference_end.fromordinal(reference_end.toordinal() - 14)).isoformat(), reference_end.isoformat()
    if "10 trading days" in lowered:
        return default_start, default_end
    return default_start, default_end
