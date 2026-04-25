"""Shared Pydantic models for SignalCanvas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


AllowedChartType = Literal["line", "bar", "scatter", "candlestick"]


class ChartFilters(BaseModel):
    """Filters embedded in the chart spec."""

    symbols: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None


class ChartSpec(BaseModel):
    """Validated model output for charts."""

    chart_type: AllowedChartType
    title: str
    x: str
    y: str
    color: str | None = None
    filters: ChartFilters = Field(default_factory=ChartFilters)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    y2: str | None = None


class SQLResult(BaseModel):
    """Structured SQL result payload."""

    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    sample_rows: list[dict[str, Any]]


class ForecastResult(BaseModel):
    """Structured output from the forecast tool."""

    historical_values: list[dict[str, Any]]
    forecast_values: list[dict[str, Any]]
    confidence_band: list[dict[str, Any]] | None = None
    metadata: dict[str, Any]
    caution_text: str


class EvaluatorResult(BaseModel):
    """Groundedness and quality checks for an answer."""

    score: float
    issues: list[str] = Field(default_factory=list)
    grounded: bool = True


class AgentFinalResponse(BaseModel):
    """Final response object sent from the agent to the app."""

    answer: str
    chart_spec: ChartSpec | None = None
    sql_used: str | None = None
    forecast_used: bool = False
    symbols: list[str] = Field(default_factory=list)
    evaluator: EvaluatorResult


class ToolCallContext(BaseModel):
    """UI context passed into the agent for defaults."""

    default_symbols: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    forecast_horizon: int = 10
    forecast_method: str = "auto"


class AnalysisType(str):
    """Simple string constants for supported request types."""

    VISUALIZE = "visualize"
    COMPARE = "compare"
    ANOMALY = "anomaly"
    FORECAST = "forecast"


class BuildChartSpecInput(BaseModel):
    """Input payload for chart spec building."""

    question: str
    available_columns: list[str]
    sample_rows: list[dict[str, Any]]


class GetPriceHistoryInput(BaseModel):
    """Arguments for price history retrieval."""

    symbols: list[str]
    start_date: str
    end_date: str

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, symbols: list[str]) -> list[str]:
        if not symbols:
            raise ValueError("At least one symbol is required.")
        return [symbol.upper() for symbol in symbols]


class ForecastInput(BaseModel):
    """Arguments for forecasting."""

    symbol: str
    horizon: int = 10
    method: str = "auto"


class RunSQLInput(BaseModel):
    """Arguments for the SQL execution tool."""

    query: str
