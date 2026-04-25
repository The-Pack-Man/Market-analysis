"""Chart spec validation and Plotly rendering."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from signalcanvas.models import ChartSpec, ForecastResult


def validate_chart_spec(spec: ChartSpec | dict[str, Any], frame: pd.DataFrame) -> ChartSpec:
    """Validate a chart spec against a concrete dataframe."""
    chart_spec = spec if isinstance(spec, ChartSpec) else ChartSpec.model_validate(spec)
    available_columns = set(frame.columns.tolist())
    required_columns = {chart_spec.x, chart_spec.y}
    if chart_spec.color:
        required_columns.add(chart_spec.color)
    if chart_spec.y2:
        required_columns.add(chart_spec.y2)

    missing = sorted(required_columns - available_columns)
    if missing:
        raise ValueError(f"Chart spec refers to missing columns: {missing}")
    if chart_spec.chart_type == "candlestick":
        needed = {"trade_date", "open", "high", "low", "close"}
        if not needed.issubset(available_columns):
            raise ValueError("Candlestick charts require trade_date, open, high, low, and close columns.")
    if chart_spec.chart_type in {"line", "scatter"} and chart_spec.x == chart_spec.y:
        raise ValueError("The x and y axes must refer to different columns.")
    return chart_spec


def render_chart(frame: pd.DataFrame, spec: ChartSpec | dict[str, Any]) -> go.Figure:
    """Render a Plotly figure from a validated chart spec."""
    chart_spec = validate_chart_spec(spec, frame)
    plot_frame = frame.copy()
    if chart_spec.x == "trade_date" and "trade_date" in plot_frame.columns:
        plot_frame["trade_date"] = pd.to_datetime(plot_frame["trade_date"])

    if chart_spec.chart_type == "line":
        figure = px.line(plot_frame, x=chart_spec.x, y=chart_spec.y, color=chart_spec.color, title=chart_spec.title)
    elif chart_spec.chart_type == "bar":
        figure = px.bar(plot_frame, x=chart_spec.x, y=chart_spec.y, color=chart_spec.color, title=chart_spec.title)
    elif chart_spec.chart_type == "scatter":
        figure = px.scatter(plot_frame, x=chart_spec.x, y=chart_spec.y, color=chart_spec.color, title=chart_spec.title)
    elif chart_spec.chart_type == "candlestick":
        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=plot_frame["trade_date"],
                    open=plot_frame["open"],
                    high=plot_frame["high"],
                    low=plot_frame["low"],
                    close=plot_frame["close"],
                    name=chart_spec.title,
                )
            ]
        )
        figure.update_layout(title=chart_spec.title)
    else:
        raise ValueError(f"Unsupported chart type: {chart_spec.chart_type}")

    if chart_spec.y2:
        figure.add_scatter(x=plot_frame[chart_spec.x], y=plot_frame[chart_spec.y2], name=chart_spec.y2, yaxis="y2")
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right"})

    for annotation in chart_spec.annotations:
        figure.add_annotation(**annotation)

    figure.update_layout(legend_title_text=chart_spec.color or "", margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return figure


def render_forecast_chart(history_frame: pd.DataFrame, forecast: ForecastResult, symbol: str) -> go.Figure:
    """Render a historical close series with forecast and optional confidence band."""
    plot_frame = history_frame.copy()
    plot_frame["trade_date"] = pd.to_datetime(plot_frame["trade_date"])
    forecast_frame = pd.DataFrame(forecast.forecast_values)
    forecast_frame["trade_date"] = pd.to_datetime(forecast_frame["trade_date"])
    last_history_date = plot_frame["trade_date"].iloc[-1]
    last_history_close = float(plot_frame["close"].iloc[-1])
    forecast_frame = pd.concat(
        [
            pd.DataFrame([{"trade_date": last_history_date, "forecast_close": last_history_close}]),
            forecast_frame,
        ],
        ignore_index=True,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot_frame["trade_date"],
            y=plot_frame["close"],
            mode="lines",
            name=f"{symbol} historical",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_frame["trade_date"],
            y=forecast_frame["forecast_close"],
            mode="lines",
            name=f"{symbol} forecast",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
        )
    )

    if forecast.confidence_band:
        band_frame = pd.DataFrame(forecast.confidence_band)
        band_frame["trade_date"] = pd.to_datetime(band_frame["trade_date"])
        band_frame = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "trade_date": last_history_date,
                            "lower": last_history_close,
                            "upper": last_history_close,
                        }
                    ]
                ),
                band_frame,
            ],
            ignore_index=True,
        )
        figure.add_trace(
            go.Scatter(
                x=band_frame["trade_date"],
                y=band_frame["upper"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=band_frame["trade_date"],
                y=band_frame["lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(214, 39, 40, 0.15)",
                name="confidence band",
            )
        )

    figure.update_layout(
        title=f"{symbol} forecast",
        xaxis_title="Date",
        yaxis_title="Close",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def render_multi_forecast_chart(history_frame: pd.DataFrame, forecasts: dict[str, ForecastResult]) -> go.Figure:
    """Render multiple historical and forecast series on one chart."""
    figure = go.Figure()
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    for idx, (symbol, forecast) in enumerate(forecasts.items()):
        color = palette[idx % len(palette)]
        symbol_history = history_frame.loc[history_frame["symbol"] == symbol].copy().sort_values("trade_date")
        if symbol_history.empty:
            continue
        symbol_history["trade_date"] = pd.to_datetime(symbol_history["trade_date"])

        forecast_frame = pd.DataFrame(forecast.forecast_values)
        forecast_frame["trade_date"] = pd.to_datetime(forecast_frame["trade_date"])
        last_history_date = symbol_history["trade_date"].iloc[-1]
        last_history_close = float(symbol_history["close"].iloc[-1])
        forecast_frame = pd.concat(
            [
                pd.DataFrame([{"trade_date": last_history_date, "forecast_close": last_history_close}]),
                forecast_frame,
            ],
            ignore_index=True,
        )

        figure.add_trace(
            go.Scatter(
                x=symbol_history["trade_date"],
                y=symbol_history["close"],
                mode="lines",
                name=f"{symbol} historical",
                line={"color": color, "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["trade_date"],
                y=forecast_frame["forecast_close"],
                mode="lines",
                name=f"{symbol} forecast",
                line={"color": color, "width": 2, "dash": "dash"},
            )
        )

    figure.update_layout(
        title="Forecast comparison",
        xaxis_title="Date",
        yaxis_title="Close",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure
