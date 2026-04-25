"""Streamlit entry point for SignalCanvas."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from signalcanvas.agent import SignalCanvasAgent
from signalcanvas.config import AppConfig, get_config
from signalcanvas.db import DatabaseManager
from signalcanvas.ingest import refresh_market_data


DATE_PRESETS: dict[str, int] = {
    "30D": 30,
    "90D": 90,
    "YTD": -1,
    "1Y": 365,
}


def _default_date_range() -> tuple[date, date]:
    end_date = date.today()
    start_date = end_date - timedelta(days=90)
    return start_date, end_date


def _preset_to_range(preset: str) -> tuple[date, date]:
    end_date = date.today()
    if preset == "YTD":
        return date(end_date.year, 1, 1), end_date
    return end_date - timedelta(days=DATE_PRESETS[preset]), end_date


def _initialize_state(config: AppConfig, db: DatabaseManager) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = SignalCanvasAgent(config=config, db=db)
    if "date_preset" not in st.session_state:
        st.session_state.date_preset = "90D"
    if "start_date" not in st.session_state or "end_date" not in st.session_state:
        start_date, end_date = _default_date_range()
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
    if "forecast_horizon" not in st.session_state:
        st.session_state.forecast_horizon = 10
    if "forecast_method" not in st.session_state:
        st.session_state.forecast_method = "auto"


def _render_sidebar(config: AppConfig, db: DatabaseManager) -> dict[str, Any]:
    st.sidebar.header("Controls")
    symbols_text = st.sidebar.text_input(
        "Tracked/default symbols",
        value=", ".join(config.default_symbols),
        help="Comma-separated symbols used for refresh and default suggestions.",
    )
    preset = st.sidebar.selectbox("Date range preset", options=list(DATE_PRESETS), index=list(DATE_PRESETS).index(st.session_state.date_preset))
    if preset != st.session_state.date_preset:
        st.session_state.date_preset = preset
        start_date, end_date = _preset_to_range(preset)
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

    start_date = st.sidebar.date_input("Start date", value=st.session_state.start_date)
    end_date = st.sidebar.date_input("End date", value=st.session_state.end_date)
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    st.sidebar.subheader("Forecast Settings")
    st.session_state.forecast_horizon = st.sidebar.slider("Forecast horizon", min_value=3, max_value=30, value=st.session_state.forecast_horizon)
    st.session_state.forecast_method = st.sidebar.selectbox(
        "Forecast method",
        options=["auto", "damped_trend", "naive", "moving_average", "linear_trend"],
        index=["auto", "damped_trend", "naive", "moving_average", "linear_trend"].index(st.session_state.forecast_method),
    )

    refresh_clicked = st.sidebar.button("Refresh data", type="primary")
    if refresh_clicked:
        with st.sidebar:
            with st.spinner("Refreshing market data..."):
                refreshed = refresh_market_data(
                    db=db,
                    symbols=[symbol.strip().upper() for symbol in symbols_text.split(",") if symbol.strip()],
                    start_date=(date.today() - timedelta(days=400)).isoformat(),
                    end_date=date.today().isoformat(),
                )
                st.success(f"Refreshed {refreshed} symbols")

    return {
        "default_symbols": [symbol.strip().upper() for symbol in symbols_text.split(",") if symbol.strip()],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "forecast_horizon": st.session_state.forecast_horizon,
        "forecast_method": st.session_state.forecast_method,
    }


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chart"):
                st.plotly_chart(message["chart"], use_container_width=True)
            if message.get("sql_used"):
                with st.expander("SQL used"):
                    st.code(message["sql_used"], language="sql")
            if message.get("evaluator"):
                with st.expander("Evaluator notes"):
                    st.json(message["evaluator"])
            if message.get("external_context"):
                with st.expander("Possible external context"):
                    for article in message["external_context"].get("articles", []):
                        st.markdown(f"- [{article['title']}]({article['link']})")
                        if article.get("source") or article.get("published"):
                            st.caption(f"{article.get('source', '')} {article.get('published', '')}".strip())
                        if article.get("description"):
                            st.write(article["description"])
                    if message["external_context"].get("themes"):
                        st.write("Themes: " + ", ".join(message["external_context"]["themes"]))


def _render_recent_analyses(db: DatabaseManager) -> None:
    analyses = db.get_recent_analyses(limit=10)
    st.subheader("Recent analyses")
    if analyses.empty:
        st.info("No analyses logged yet.")
        return
    st.dataframe(analyses, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="SignalCanvas", layout="wide")
    st.title("SignalCanvas")
    st.caption("Ask grounded questions about local stock data, inspect the SQL, and view validated charts.")

    config = get_config()
    db = DatabaseManager(config.db_path)
    _initialize_state(config, db)
    sidebar_state = _render_sidebar(config, db)

    if db.daily_prices_row_count() == 0:
        st.info("No market data loaded yet. Use the sidebar refresh action to ingest daily prices for your default symbols.")

    _render_history()

    prompt = st.chat_input("Ask about a stock, comparison, drawdown, volatility, or forecast")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.agent.answer(
                    question=prompt,
                    ui_context=sidebar_state,
                )
                st.markdown(response.answer)
                if response.figure is not None:
                    st.plotly_chart(response.figure, use_container_width=True)
                with st.expander("SQL used"):
                    st.code(response.sql_used or "-- no SQL used --", language="sql")
                with st.expander("Evaluator notes"):
                    st.json(response.evaluator.model_dump())
                if response.external_context:
                    with st.expander("Possible external context"):
                        for article in response.external_context.get("articles", []):
                            st.markdown(f"- [{article['title']}]({article['link']})")
                            if article.get("source") or article.get("published"):
                                st.caption(f"{article.get('source', '')} {article.get('published', '')}".strip())
                            if article.get("description"):
                                st.write(article["description"])
                        if response.external_context.get("themes"):
                            st.write("Themes: " + ", ".join(response.external_context["themes"]))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "chart": response.figure,
                "sql_used": response.sql_used,
                "evaluator": response.evaluator.model_dump(),
                "external_context": response.external_context,
            }
        )

    st.divider()
    _render_recent_analyses(db)


if __name__ == "__main__":
    main()
