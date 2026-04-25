"""Tests for chart spec validation and rendering."""

from __future__ import annotations

import pandas as pd
import pytest

from signalcanvas.charts import render_chart, validate_chart_spec
from signalcanvas.models import ChartSpec


def test_validate_chart_spec_accepts_line_chart() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "close": [100.0, 101.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    spec = ChartSpec(chart_type="line", title="Test", x="trade_date", y="close", color="symbol")
    validated = validate_chart_spec(spec, frame)
    assert validated.chart_type == "line"


def test_validate_chart_spec_rejects_missing_column() -> None:
    frame = pd.DataFrame({"trade_date": ["2026-01-01"], "close": [100.0]})
    spec = {"chart_type": "line", "title": "Bad", "x": "trade_date", "y": "volume", "filters": {}, "annotations": [], "y2": None}
    with pytest.raises(ValueError):
        validate_chart_spec(spec, frame)


def test_render_chart_returns_plotly_figure() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "close": [100.0, 101.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    spec = ChartSpec(chart_type="line", title="Chart", x="trade_date", y="close", color="symbol")
    figure = render_chart(frame, spec)
    assert len(figure.data) == 1
