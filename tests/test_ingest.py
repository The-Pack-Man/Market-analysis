"""Tests for ingestion normalization."""

from __future__ import annotations

import pandas as pd

from signalcanvas.ingest import _normalize_download


def test_normalize_download_handles_multiindex_columns() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
        ],
        names=["Price", "Ticker"],
    )
    frame = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.4, 1_000_000]],
        columns=columns,
        index=pd.to_datetime(["2026-04-01"]),
    )
    frame.index.name = "Date"

    normalized = _normalize_download("AAPL", frame)

    assert list(normalized.columns) == [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        "ingested_at",
    ]
    assert normalized.loc[0, "symbol"] == "AAPL"
    assert float(normalized.loc[0, "close"]) == 100.5
