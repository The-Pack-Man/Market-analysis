"""Feature engineering for daily price history."""

from __future__ import annotations

import pandas as pd

from signalcanvas.db import DatabaseManager


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive rolling features from daily price history."""
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "close",
                "daily_return",
                "rolling_7d_avg",
                "rolling_30d_avg",
                "rolling_vol_20d",
                "drawdown_from_90d_high",
            ]
        )

    ordered = prices.sort_values(["symbol", "trade_date"]).copy()
    ordered["daily_return"] = ordered.groupby("symbol")["close"].pct_change()
    ordered["rolling_7d_avg"] = ordered.groupby("symbol")["close"].transform(lambda series: series.rolling(window=7, min_periods=1).mean())
    ordered["rolling_30d_avg"] = ordered.groupby("symbol")["close"].transform(lambda series: series.rolling(window=30, min_periods=1).mean())
    ordered["rolling_vol_20d"] = ordered.groupby("symbol")["daily_return"].transform(lambda series: series.rolling(window=20, min_periods=5).std())
    ordered["rolling_90d_high"] = ordered.groupby("symbol")["close"].transform(lambda series: series.rolling(window=90, min_periods=1).max())
    ordered["drawdown_from_90d_high"] = ordered["close"] / ordered["rolling_90d_high"] - 1
    return ordered[
        [
            "symbol",
            "trade_date",
            "close",
            "daily_return",
            "rolling_7d_avg",
            "rolling_30d_avg",
            "rolling_vol_20d",
            "drawdown_from_90d_high",
        ]
    ]


def rebuild_features(db: DatabaseManager) -> None:
    """Recompute the feature table from `daily_prices`."""
    prices = db.query_df(
        """
        SELECT symbol, trade_date, close
        FROM daily_prices
        ORDER BY symbol, trade_date
        """
    )
    feature_frame = build_feature_frame(prices)
    db.replace_price_features(feature_frame)
