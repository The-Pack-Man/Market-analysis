"""Tool registry for SignalCanvas."""

from __future__ import annotations

from typing import Any

import pandas as pd

from signalcanvas.charts import validate_chart_spec
from signalcanvas.db import DatabaseManager
from signalcanvas.forecasts import baseline_forecast
from signalcanvas.models import (
    BuildChartSpecInput,
    ChartSpec,
    ForecastInput,
    ForecastResult,
    GetPriceHistoryInput,
    RunSQLInput,
    SQLResult,
)


def get_schema(db: DatabaseManager) -> dict[str, dict[str, list[str]]]:
    """Return database schema information."""
    return db.list_schema()


def run_sql(db: DatabaseManager, query: str) -> SQLResult:
    """Execute a safe read-only SQL query."""
    payload = RunSQLInput(query=query)
    frame = db.run_read_only_query(payload.query)
    rows = frame.to_dict(orient="records")
    return SQLResult(rows=rows, columns=frame.columns.tolist(), row_count=len(frame), sample_rows=rows[:5])


def get_price_history(db: DatabaseManager, symbols: list[str], start_date: str, end_date: str) -> SQLResult:
    """Fetch OHLCV history for a list of symbols."""
    payload = GetPriceHistoryInput(symbols=symbols, start_date=start_date, end_date=end_date)
    placeholders = ", ".join(["?"] * len(payload.symbols))
    query = f"""
        SELECT
            symbol,
            trade_date,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume
        FROM daily_prices
        WHERE symbol IN ({placeholders})
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, symbol
    """
    frame = db.query_df(query, payload.symbols + [payload.start_date, payload.end_date])
    rows = frame.to_dict(orient="records")
    return SQLResult(rows=rows, columns=frame.columns.tolist(), row_count=len(frame), sample_rows=rows[:5])


def build_chart_spec(question: str, available_columns: list[str], sample_rows: list[dict[str, Any]]) -> ChartSpec:
    """Create a constrained chart spec using deterministic heuristics."""
    payload = BuildChartSpecInput(question=question, available_columns=available_columns, sample_rows=sample_rows)
    lowered = payload.question.lower()

    if "candlestick" in lowered and {"open", "high", "low", "close", "trade_date"}.issubset(set(available_columns)):
        return ChartSpec(chart_type="candlestick", title=question.title(), x="trade_date", y="close", color=None)

    y_column = "close"
    if "drawdown" in lowered and "drawdown_from_90d_high" in available_columns:
        y_column = "drawdown_from_90d_high"
    elif "volatil" in lowered and "rolling_vol_20d" in available_columns:
        y_column = "rolling_vol_20d"
    elif "return" in lowered and "daily_return" in available_columns:
        y_column = "daily_return"
    elif "volume" in lowered and "volume" in available_columns:
        y_column = "volume"

    spec = ChartSpec(
        chart_type="line" if "scatter" not in lowered and "bar" not in lowered else ("scatter" if "scatter" in lowered else "bar"),
        title=question.strip().rstrip("?"),
        x="trade_date" if "trade_date" in available_columns else available_columns[0],
        y=y_column if y_column in available_columns else available_columns[1],
        color="symbol" if "symbol" in available_columns else None,
    )

    sample_frame = pd.DataFrame(sample_rows) if sample_rows else pd.DataFrame(columns=available_columns)
    validate_chart_spec(spec, sample_frame if not sample_frame.empty else pd.DataFrame(columns=available_columns))
    return spec


def forecast_series(db: DatabaseManager, symbol: str, horizon: int, method: str = "auto") -> ForecastResult:
    """Forecast a symbol's closing prices."""
    payload = ForecastInput(symbol=symbol.upper(), horizon=horizon, method=method)
    frame = db.query_df(
        """
        SELECT trade_date, close
        FROM daily_prices
        WHERE symbol = ?
        ORDER BY trade_date
        """,
        [payload.symbol],
    )
    return baseline_forecast(series=frame, symbol=payload.symbol, horizon=payload.horizon, method=payload.method)
