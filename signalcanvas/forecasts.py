"""Baseline forecasting methods for SignalCanvas."""

from __future__ import annotations

from typing import Any

import math
import pandas as pd

from signalcanvas.models import ForecastResult


def _future_business_dates(last_date: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _moving_average_path(ordered: pd.DataFrame, horizon: int) -> list[float]:
    """Mean-revert gradually toward a recent average without jumping on day one."""
    last_close = float(ordered["close"].iloc[-1])
    recent_mean = float(ordered["close"].tail(min(10, len(ordered))).mean())
    values: list[float] = []
    for step in range(1, horizon + 1):
        weight = 1 - math.exp(-step / max(horizon / 2, 1))
        values.append(last_close + (recent_mean - last_close) * weight)
    return values


def _damped_trend_path(ordered: pd.DataFrame, horizon: int) -> list[float]:
    """Project forward from the last close using a capped recent drift with damping."""
    last_close = float(ordered["close"].iloc[-1])
    returns = ordered["close"].pct_change().dropna()
    if returns.empty:
        return [last_close] * horizon

    recent_returns = returns.tail(min(15, len(returns)))
    weights = pd.Series(range(1, len(recent_returns) + 1), dtype="float64")
    weighted_drift = float((recent_returns.reset_index(drop=True) * weights).sum() / weights.sum())
    weighted_drift = _clamp(weighted_drift, -0.03, 0.03)

    values: list[float] = []
    current = last_close
    for step in range(1, horizon + 1):
        damped_drift = weighted_drift * math.exp(-(step - 1) / max(horizon / 2, 1))
        current = current * (1 + damped_drift)
        values.append(float(current))
    return values


def baseline_forecast(series: pd.DataFrame, symbol: str, horizon: int, method: str = "auto") -> ForecastResult:
    """Forecast a price series with simple baseline methods."""
    if series.empty:
        raise ValueError(f"No history available for {symbol}.")

    ordered = series.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    chosen_method = method
    if method == "auto":
        chosen_method = "damped_trend" if len(ordered) >= 20 else "naive"

    last_close = float(ordered["close"].iloc[-1])
    future_dates = _future_business_dates(ordered["trade_date"].iloc[-1], horizon)

    if chosen_method == "naive":
        values = [last_close] * horizon
    elif chosen_method == "moving_average":
        values = _moving_average_path(ordered, horizon)
    elif chosen_method == "damped_trend":
        values = _damped_trend_path(ordered, horizon)
    elif chosen_method == "linear_trend":
        if len(ordered) < 2:
            values = [last_close] * horizon
        else:
            x = pd.Series(range(len(ordered)), dtype="float64")
            denominator = ((x - x.mean()) ** 2).sum()
            slope = 0.0 if denominator == 0 else ((x - x.mean()) * (ordered["close"] - ordered["close"].mean())).sum() / denominator
            intercept = ordered["close"].mean() - slope * x.mean()
            values = [float(intercept + slope * (len(ordered) + idx)) for idx in range(horizon)]
    else:
        raise ValueError(f"Unsupported forecast method: {chosen_method}")

    residual_scale = float(ordered["close"].pct_change().dropna().std() or 0.02)
    forecast_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for idx, (forecast_date, value) in enumerate(zip(future_dates, values, strict=True), start=1):
        spread = abs(value) * residual_scale * (1 + idx / max(horizon, 1))
        forecast_rows.append({"trade_date": forecast_date.date().isoformat(), "forecast_close": round(value, 4)})
        band_rows.append(
            {
                "trade_date": forecast_date.date().isoformat(),
                "lower": round(value - spread, 4),
                "upper": round(value + spread, 4),
            }
        )

    caution_text = (
        "This is a simple baseline estimate using historical price patterns. It is uncertain, should not be treated "
        "as a confident prediction, and may miss regime changes or market news."
    )

    return ForecastResult(
        historical_values=ordered.tail(60).assign(trade_date=lambda frame: frame["trade_date"].dt.date.astype(str)).to_dict(orient="records"),
        forecast_values=forecast_rows,
        confidence_band=band_rows,
        metadata={"method": chosen_method, "horizon": horizon, "symbol": symbol},
        caution_text=caution_text,
    )
