"""Market data ingestion for SignalCanvas."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from signalcanvas.db import DatabaseManager
from signalcanvas.features import rebuild_features


def _normalize_download(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance output into the app's raw and daily schema."""
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "ingested_at"])

    working = frame.copy()
    if isinstance(working.columns, pd.MultiIndex):
        if "Ticker" in working.columns.names:
            working = working.droplevel("Ticker", axis=1)
        else:
            working.columns = working.columns.get_level_values(0)
    working = working.loc[:, ~working.columns.duplicated()]

    normalized = working.reset_index().rename(
        columns={
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    )
    normalized["symbol"] = symbol.upper()
    normalized["source"] = "yfinance"
    normalized["ingested_at"] = datetime.utcnow()
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.date
    for column in ["open", "high", "low", "close", "adjusted_close"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(0).astype("int64")
    return normalized[["symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "ingested_at"]]


def refresh_market_data(
    db: DatabaseManager,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> int:
    """Refresh daily price history for a list of symbols."""
    refreshed = 0
    for symbol in symbols:
        frame = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=False, progress=False)
        normalized = _normalize_download(symbol, frame)
        if normalized.empty:
            continue
        db.append_raw_prices(normalized)
        daily_frame = normalized.drop(columns=["source", "ingested_at"]).drop_duplicates(subset=["symbol", "trade_date"])
        db.upsert_daily_prices(daily_frame)
        refreshed += 1
    if refreshed:
        rebuild_features(db=db)
    return refreshed
