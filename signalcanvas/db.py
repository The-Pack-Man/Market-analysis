"""DuckDB access layer for SignalCanvas."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from signalcanvas.utils import make_id


READ_ONLY_SQL_PATTERN = re.compile(r"^\s*(WITH\b[\s\S]+SELECT\b|SELECT\b|SHOW\b|DESCRIBE\b)", re.IGNORECASE)
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|CALL|MERGE|TRUNCATE|REPLACE)\b",
    re.IGNORECASE,
)


class DatabaseManager:
    """Manage schema, queries, and logging against a persistent DuckDB database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.db_path))
        self.initialize()

    def initialize(self) -> None:
        """Create the schema and views if they do not exist."""
        root = Path(__file__).resolve().parent.parent
        schema_sql = (root / "sql" / "schema.sql").read_text(encoding="utf-8")
        views_sql = (root / "sql" / "views.sql").read_text(encoding="utf-8")
        self.connection.execute(schema_sql)
        self.connection.execute(views_sql)

    def execute(self, query: str, parameters: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute arbitrary SQL for internal trusted code paths."""
        return self.connection.execute(query, parameters or [])

    def query_df(self, query: str, parameters: list[Any] | None = None) -> pd.DataFrame:
        """Return a DataFrame from a SQL query."""
        return self.connection.execute(query, parameters or []).fetchdf()

    def run_read_only_query(self, query: str) -> pd.DataFrame:
        """Validate and execute a read-only query."""
        normalized = query.strip().rstrip(";")
        if ";" in normalized:
            raise ValueError("Only a single SELECT statement is allowed.")
        if not READ_ONLY_SQL_PATTERN.match(normalized):
            raise ValueError("Only read-only SELECT queries are allowed.")
        if FORBIDDEN_SQL_PATTERN.search(normalized):
            raise ValueError("Query contains a forbidden SQL operation.")
        return self.connection.execute(normalized).fetchdf()

    def list_schema(self) -> dict[str, dict[str, list[str]]]:
        """Return tables, views, and columns available in the database."""
        tables_df = self.query_df(
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_type, table_name
            """
        )
        columns_df = self.query_df(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        )

        result: dict[str, dict[str, list[str]]] = {"tables": {}, "views": {}}
        for _, row in tables_df.iterrows():
            key = "views" if row["table_type"] == "VIEW" else "tables"
            result[key][row["table_name"]] = columns_df.loc[columns_df["table_name"] == row["table_name"], "column_name"].tolist()
        return result

    def upsert_daily_prices(self, frame: pd.DataFrame) -> None:
        """Upsert daily price rows into the deduplicated table."""
        if frame.empty:
            return
        self.connection.register("prices_frame", frame)
        self.connection.execute(
            """
            MERGE INTO daily_prices AS target
            USING prices_frame AS source
            ON target.symbol = source.symbol AND target.trade_date = source.trade_date
            WHEN MATCHED THEN UPDATE SET
                open = source.open,
                high = source.high,
                low = source.low,
                close = source.close,
                adjusted_close = source.adjusted_close,
                volume = source.volume
            WHEN NOT MATCHED THEN INSERT (
                symbol, trade_date, open, high, low, close, adjusted_close, volume
            ) VALUES (
                source.symbol, source.trade_date, source.open, source.high, source.low, source.close, source.adjusted_close, source.volume
            )
            """
        )
        self.connection.unregister("prices_frame")

    def append_raw_prices(self, frame: pd.DataFrame) -> None:
        """Append ingested rows to the raw table."""
        if frame.empty:
            return
        self.connection.register("raw_frame", frame)
        self.connection.execute("INSERT INTO raw_prices SELECT * FROM raw_frame")
        self.connection.unregister("raw_frame")

    def replace_price_features(self, frame: pd.DataFrame) -> None:
        """Replace computed feature rows for the symbols present in the frame."""
        if frame.empty:
            return
        symbols = frame["symbol"].dropna().unique().tolist()
        placeholders = ", ".join(["?"] * len(symbols))
        self.connection.execute(f"DELETE FROM price_features WHERE symbol IN ({placeholders})", symbols)
        self.connection.register("features_frame", frame)
        self.connection.execute("INSERT INTO price_features SELECT * FROM features_frame")
        self.connection.unregister("features_frame")

    def log_analysis(
        self,
        user_question: str,
        symbols: list[str],
        sql_used: str | None,
        chart_spec_json: str | None,
        narrative: str,
        evaluator_score: float,
    ) -> None:
        """Persist analysis metadata for history."""
        self.connection.execute(
            """
            INSERT INTO analysis_log (
                id, user_question, symbols, sql_used, chart_spec_json, narrative, evaluator_score, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                make_id(),
                user_question,
                ",".join(symbols),
                sql_used,
                chart_spec_json,
                narrative,
                evaluator_score,
            ],
        )

    def get_recent_analyses(self, limit: int = 10) -> pd.DataFrame:
        """Return recent analyses for display."""
        return self.query_df(
            """
            SELECT
                created_at,
                user_question,
                symbols,
                evaluator_score
            FROM analysis_log
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        )

    def daily_prices_row_count(self) -> int:
        """Return the number of rows in `daily_prices`."""
        return int(self.connection.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0])

    def get_available_symbols(self) -> list[str]:
        """Return distinct symbols currently available in the local database."""
        rows = self.connection.execute(
            """
            SELECT DISTINCT symbol
            FROM daily_prices
            ORDER BY symbol
            """
        ).fetchall()
        return [row[0] for row in rows]

    def get_latest_trade_date(self, symbols: list[str]) -> str | None:
        """Return the latest stored trade date for the requested symbols."""
        if not symbols:
            return None
        placeholders = ", ".join(["?"] * len(symbols))
        row = self.connection.execute(
            f"""
            SELECT MAX(trade_date)
            FROM daily_prices
            WHERE symbol IN ({placeholders})
            """,
            symbols,
        ).fetchone()
        return row[0].isoformat() if row and row[0] is not None else None
