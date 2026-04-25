"""Shared fixtures for SignalCanvas tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signalcanvas.db import DatabaseManager
from signalcanvas.features import rebuild_features


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Return a temporary DuckDB-backed database with sample prices."""
    manager = DatabaseManager(tmp_path / "test.duckdb")
    rows = []
    dates = pd.bdate_range("2026-01-02", periods=40)
    for symbol, offset in [("AAPL", 0.8), ("MSFT", 1.1), ("SPY", 0.5), ("NVDA", 1.6)]:
        for idx, trade_date in enumerate(dates, start=1):
            close = 100 + idx * offset
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date.date(),
                    "open": close - 1,
                    "high": close + 1,
                    "low": close - 2,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000 + idx,
                }
            )
    frame = pd.DataFrame(rows)
    manager.upsert_daily_prices(frame)
    rebuild_features(manager)
    return manager
