"""Tests for baseline forecasts."""

from __future__ import annotations

import pandas as pd

from signalcanvas.charts import render_forecast_chart
from signalcanvas.tools import forecast_series


def test_forecast_series_returns_requested_horizon(db) -> None:
    result = forecast_series(db, symbol="AAPL", horizon=10, method="moving_average")
    assert len(result.forecast_values) == 10
    assert result.metadata["method"] == "moving_average"
    assert "uncertain" in result.caution_text.lower()


def test_moving_average_forecast_stays_anchored_to_last_close(db) -> None:
    history = db.query_df(
        """
        SELECT trade_date, close
        FROM daily_prices
        WHERE symbol = 'AAPL'
        ORDER BY trade_date
        """
    )
    last_close = float(history["close"].iloc[-1])
    recent_mean = float(history["close"].tail(10).mean())

    result = forecast_series(db, symbol="AAPL", horizon=10, method="moving_average")
    first_forecast = float(result.forecast_values[0]["forecast_close"])

    assert min(last_close, recent_mean) <= first_forecast <= max(last_close, recent_mean)
    assert abs(first_forecast - last_close) < abs(recent_mean - last_close)


def test_auto_forecast_uses_damped_trend_for_longer_history(db) -> None:
    result = forecast_series(db, symbol="AAPL", horizon=10, method="auto")
    assert result.metadata["method"] == "damped_trend"


def test_render_forecast_chart_adds_forecast_trace(db) -> None:
    result = forecast_series(db, symbol="AAPL", horizon=5, method="moving_average")
    history = db.query_df(
        """
        SELECT trade_date, close, symbol
        FROM daily_prices
        WHERE symbol = 'AAPL'
        ORDER BY trade_date
        """
    )
    figure = render_forecast_chart(history, result, "AAPL")
    assert len(figure.data) >= 2
    assert figure.data[0].name == "AAPL historical"
    assert figure.data[1].name == "AAPL forecast"
    assert pd.Timestamp(figure.data[1].x[0]) == pd.Timestamp(history["trade_date"].iloc[-1])
    assert float(figure.data[1].y[0]) == float(history["close"].iloc[-1])
