"""Tests for the read-only SQL tool."""

from __future__ import annotations

import pytest

from signalcanvas.tools import run_sql


def test_run_sql_allows_select(db) -> None:
    result = run_sql(db, "SELECT symbol, close FROM daily_prices ORDER BY trade_date LIMIT 2")
    assert result.row_count == 2
    assert result.columns == ["symbol", "close"]


def test_run_sql_rejects_write_statements(db) -> None:
    with pytest.raises(ValueError):
        run_sql(db, "DELETE FROM daily_prices")
