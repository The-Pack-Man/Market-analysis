"""Agent orchestration for SignalCanvas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from signalcanvas.charts import render_chart, render_forecast_chart, render_multi_forecast_chart
from signalcanvas.config import AppConfig
from signalcanvas.db import DatabaseManager
from signalcanvas.evaluator import evaluate_groundedness
from signalcanvas.news import build_event_context
from signalcanvas.models import AgentFinalResponse, ChartSpec, EvaluatorResult, ToolCallContext
from signalcanvas.prompts import SYSTEM_PROMPT
from signalcanvas.tools import build_chart_spec, forecast_series, get_price_history
from signalcanvas.utils import detect_analysis_type, detect_question_focus, infer_date_range, parse_symbols

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass
class AgentResponseBundle:
    """Response bundle returned to the Streamlit app."""

    answer: str
    chart_spec: ChartSpec | None
    sql_used: str | None
    forecast_used: bool
    symbols: list[str]
    evaluator: EvaluatorResult
    figure: go.Figure | None = None
    external_context: dict[str, Any] | None = None


class SignalCanvasAgent:
    """A small agent wrapper with optional OpenAI tool-assisted reasoning."""

    def __init__(self, config: AppConfig, db: DatabaseManager) -> None:
        self.config = config
        self.db = db
        self.client = self._build_client()

    def _build_client(self) -> Any:
        """Build the configured LLM client if available."""
        if not OpenAI:
            return None
        if self.config.llm_provider == "ollama":
            return OpenAI(
                api_key=self.config.openai_api_key or "ollama",
                base_url=self.config.llm_base_url or self.config.ollama_base_url,
            )
        if self.config.openai_api_key:
            kwargs: dict[str, Any] = {"api_key": self.config.openai_api_key}
            if self.config.llm_base_url:
                kwargs["base_url"] = self.config.llm_base_url
            return OpenAI(**kwargs)
        return None

    def answer(self, question: str, ui_context: dict[str, Any]) -> AgentResponseBundle:
        """Answer a user question by orchestrating retrieval, charting, evaluation, and logging."""
        context = ToolCallContext.model_validate(ui_context)
        analysis_type = detect_analysis_type(question)
        question_focus = detect_question_focus(question)
        available_symbols = self.db.get_available_symbols()
        symbols = parse_symbols(
            question,
            fallback=context.default_symbols[:2] if analysis_type == "compare" else context.default_symbols[:1],
            valid_symbols=available_symbols or context.default_symbols,
        )
        start_date, end_date = infer_date_range(question, context.start_date or "2026-01-01", context.end_date or "2026-12-31")

        price_result = get_price_history(self.db, symbols=symbols, start_date=start_date, end_date=end_date)
        if price_result.row_count == 0 and symbols:
            latest_available_date = self.db.get_latest_trade_date(symbols)
            if latest_available_date:
                start_date, end_date = self._clamp_range_to_available_data(start_date, end_date, latest_available_date)
                price_result = get_price_history(self.db, symbols=symbols, start_date=start_date, end_date=end_date)
        history_frame = pd.DataFrame(price_result.rows)

        sql_used = (
            "SELECT symbol, trade_date, open, high, low, close, adjusted_close, volume "
            f"FROM daily_prices WHERE symbol IN ({', '.join(repr(symbol) for symbol in symbols)}) "
            f"AND trade_date BETWEEN '{start_date}' AND '{end_date}' ORDER BY trade_date, symbol"
        )
        chart_spec = None
        figure = None
        forecast_payload = None
        external_context: dict[str, Any] = {}

        if not history_frame.empty:
            chart_spec = build_chart_spec(question, price_result.columns, price_result.sample_rows)
            chart_spec.filters.symbols = symbols
            chart_spec.filters.start_date = start_date
            chart_spec.filters.end_date = end_date
            figure = render_chart(history_frame, chart_spec)

        if analysis_type == "forecast" and symbols:
            if question_focus == "forward_performance" and len(symbols) >= 2:
                forecast_payload = {
                    symbol: forecast_series(
                        self.db,
                        symbol=symbol,
                        horizon=context.forecast_horizon,
                        method=context.forecast_method,
                    )
                    for symbol in symbols
                }
                if not history_frame.empty:
                    figure = render_multi_forecast_chart(history_frame, forecast_payload)
            else:
                forecast_payload = forecast_series(
                    self.db,
                    symbol=symbols[0],
                    horizon=context.forecast_horizon,
                    method=context.forecast_method,
                )
                symbol_history = history_frame.loc[history_frame["symbol"] == symbols[0]].copy() if not history_frame.empty else pd.DataFrame()
                if not symbol_history.empty:
                    figure = render_forecast_chart(symbol_history, forecast_payload, symbols[0])

        narrative = self._compose_answer(
            question=question,
            symbols=symbols,
            history_frame=history_frame,
            analysis_type=analysis_type,
            question_focus=question_focus,
            external_context_out=external_context,
            forecast_payload=forecast_payload,
        )

        evaluator = evaluate_groundedness(
            question=question,
            sql_used=sql_used,
            result_summary={"row_count": price_result.row_count, "columns": price_result.columns},
            chart_spec=chart_spec,
            narrative=narrative,
        )

        self.db.log_analysis(
            user_question=question,
            symbols=symbols,
            sql_used=sql_used,
            chart_spec_json=chart_spec.model_dump_json() if chart_spec else None,
            narrative=narrative,
            evaluator_score=evaluator.score,
        )

        return AgentResponseBundle(
            answer=narrative,
            chart_spec=chart_spec,
            sql_used=sql_used,
            forecast_used=forecast_payload is not None,
            symbols=symbols,
            evaluator=evaluator,
            figure=figure,
            external_context=external_context or None,
        )

    def _compose_answer(
        self,
        question: str,
        symbols: list[str],
        history_frame: pd.DataFrame,
        analysis_type: str,
        question_focus: str,
        external_context_out: dict[str, Any] | None,
        forecast_payload: Any,
    ) -> str:
        """Compose a grounded answer, optionally with model assistance."""
        summary = self._summarize_frame(history_frame, symbols)
        external_context = self._build_external_context(question_focus, summary, symbols)
        if external_context_out is not None:
            external_context_out.update(external_context)
        if self.client:
            try:
                return self._model_answer(
                    question=question,
                    summary=summary,
                    analysis_type=analysis_type,
                    question_focus=question_focus,
                    external_context=external_context,
                    forecast_payload=forecast_payload,
                )
            except Exception:
                pass
        return self._heuristic_answer(
            question=question,
            symbols=symbols,
            summary=summary,
            analysis_type=analysis_type,
            question_focus=question_focus,
            external_context=external_context,
            forecast_payload=forecast_payload,
        )

    def _model_answer(
        self,
        question: str,
        summary: dict[str, Any],
        analysis_type: str,
        question_focus: str,
        external_context: dict[str, Any],
        forecast_payload: Any,
    ) -> str:
        """Use the OpenAI API for the final narrative when configured."""
        if isinstance(forecast_payload, dict):
            forecast_text = json.dumps({symbol: payload.model_dump() for symbol, payload in forecast_payload.items()}, default=str)
        else:
            forecast_text = json.dumps(forecast_payload.model_dump(), default=str) if forecast_payload else "null"
        prompt = {
            "question": question,
            "analysis_type": analysis_type,
            "question_focus": question_focus,
            "summary": summary,
            "external_context": external_context,
            "forecast": forecast_text,
            "instructions": "Write a concise grounded answer with no markdown bullets. Mention uncertainty for forecasts.",
        }
        model_name = self.config.ollama_model if self.config.llm_provider == "ollama" else self.config.openai_model
        response = self.client.chat.completions.create(  # type: ignore[union-attr]
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        message = response.choices[0].message.content
        if isinstance(message, list):
            text = "".join(part.text for part in message if getattr(part, "type", None) == "text")
        else:
            text = message or ""
        return text.strip() or self._heuristic_answer(
            question,
            summary.get("symbols", []),
            summary,
            analysis_type,
            question_focus,
            external_context,
            forecast_payload,
        )

    def _heuristic_answer(
        self,
        question: str,
        symbols: list[str],
        summary: dict[str, Any],
        analysis_type: str,
        question_focus: str,
        external_context: dict[str, Any],
        forecast_payload: Any,
    ) -> str:
        """Fallback grounded response without remote model access."""
        if summary["row_count"] == 0:
            return "I could not find matching rows in the local database for that request. Try refreshing data or adjusting the date range."

        parts = [f"I found {summary['row_count']} rows for {', '.join(symbols)} from {summary['start_date']} to {summary['end_date']}."]

        if question_focus == "performance" and summary["pct_change"]:
            ranked = sorted(summary["pct_change"].items(), key=lambda item: item[1], reverse=True)
            leader, laggard = ranked[0], ranked[-1]
            compare_bits = [f"{symbol} returned {change:+.2%}" for symbol, change in ranked]
            parts.append(f"On price performance, {leader[0]} did better over this window.")
            parts.append("; ".join(compare_bits) + ".")
            if len(ranked) > 1:
                parts.append(f"That puts {leader[0]} ahead of {laggard[0]} by {(leader[1] - laggard[1]):+.2%}.")
        elif question_focus == "spike":
            event_parts = [self._format_biggest_spike(summary["events"].get(symbol, {}), symbol) for symbol in symbols]
            event_parts = [part for part in event_parts if part]
            if event_parts:
                parts.append(" ".join(event_parts))
        elif question_focus == "drop":
            event_parts = [self._format_biggest_drop(summary["events"].get(symbol, {}), symbol) for symbol in symbols]
            event_parts = [part for part in event_parts if part]
            if event_parts:
                parts.append(" ".join(event_parts))
        elif question_focus == "trend":
            trend_parts = [self._format_trend_explanation(summary["events"].get(symbol, {}), symbol) for symbol in symbols]
            trend_parts = [part for part in trend_parts if part]
            if trend_parts:
                parts.append(" ".join(trend_parts))
        elif question_focus == "explanation":
            explanation_parts = [self._format_data_only_explanation(summary["events"].get(symbol, {}), symbol, question) for symbol in symbols]
            explanation_parts = [part for part in explanation_parts if part]
            if explanation_parts:
                parts.append(" ".join(explanation_parts))
        elif question_focus == "volatility" and summary["volatility"]:
            ranked = sorted(summary["volatility"].items(), key=lambda item: item[1], reverse=True)
            more_volatile, less_volatile = ranked[0], ranked[-1]
            vol_bits = [f"{symbol} 20-day realized volatility is {value:.2%}" for symbol, value in ranked]
            parts.append(f"On recent volatility, {more_volatile[0]} has been more volatile than {less_volatile[0]}.")
            parts.append("; ".join(vol_bits) + ".")
            peak_parts = [self._format_peak_volatility(summary["events"].get(symbol, {}), symbol) for symbol in symbols]
            peak_parts = [part for part in peak_parts if part]
            if peak_parts:
                parts.append(" ".join(peak_parts))
        elif analysis_type == "compare" and summary["pct_change"]:
            compare_bits = [f"{symbol} changed {change:+.2%}" for symbol, change in summary["pct_change"].items()]
            parts.append("Over this window, " + "; ".join(compare_bits) + ".")
        elif analysis_type == "anomaly" and summary["volatility"]:
            anomaly_parts = []
            for symbol in symbols:
                symbol_events = summary["events"].get(symbol, {})
                anomaly_parts.append(self._format_biggest_spike(symbol_events, symbol))
                anomaly_parts.append(self._format_biggest_drop(symbol_events, symbol))
                anomaly_parts.append(self._format_peak_volatility(symbol_events, symbol))
            anomaly_parts = [part for part in anomaly_parts if part]
            if anomaly_parts:
                parts.append(" ".join(anomaly_parts))

        if external_context.get("articles"):
            parts.append(self._format_external_context(external_context))

        if summary["latest_closes"] and question_focus not in {"performance", "volatility", "spike", "drop", "trend", "explanation"}:
            close_bits = [f"{symbol} closed at {value:.2f}" for symbol, value in summary["latest_closes"].items()]
            parts.append("Latest available closes: " + "; ".join(close_bits) + ".")
        if isinstance(forecast_payload, dict) and forecast_payload:
            projected_returns = []
            for symbol, payload in forecast_payload.items():
                start_close = summary["latest_closes"].get(symbol)
                end_forecast = float(payload.forecast_values[-1]["forecast_close"])
                if start_close:
                    projected_returns.append((symbol, end_forecast / start_close - 1, payload))
            projected_returns.sort(key=lambda item: item[1], reverse=True)
            if projected_returns:
                winner_symbol, winner_return, winner_payload = projected_returns[0]
                loser_symbol, loser_return, _ = projected_returns[-1]
                return_bits = [f"{symbol} is projected at {change:+.2%}" for symbol, change, _ in projected_returns]
                parts.append(
                    f"On this baseline forecast, {winner_symbol} looks like the stronger buy over the next {winner_payload.metadata['horizon']} trading days."
                )
                parts.append("; ".join(return_bits) + ".")
                if len(projected_returns) > 1:
                    parts.append(
                        f"That puts {winner_symbol} ahead of {loser_symbol} by {(winner_return - loser_return):+.2%} on projected return."
                    )
                parts.append(
                    "This is only a simple forecast comparison based on recent price behavior, so treat it as a rough scenario rather than a recommendation."
                )
        elif forecast_payload:
            first_point = forecast_payload.forecast_values[0]
            last_point = forecast_payload.forecast_values[-1]
            parts.append(
                f"The baseline forecast for {forecast_payload.metadata['symbol']} uses the {forecast_payload.metadata['method']} method and estimates "
                f"{first_point['forecast_close']:.2f} on {first_point['trade_date']} through {last_point['forecast_close']:.2f} on {last_point['trade_date']}. "
                "This is a simple estimate and is uncertain."
            )
            parts.append(forecast_payload.caution_text)
        return " ".join(parts)

    def _build_external_context(self, question_focus: str, summary: dict[str, Any], symbols: list[str]) -> dict[str, Any]:
        """Fetch plausible news context for anomaly and explanation prompts."""
        if not self.config.enable_web_context or not symbols:
            return {}
        if question_focus not in {"spike", "drop", "trend", "volatility", "explanation"}:
            return {}

        symbol = symbols[0]
        symbol_events = summary.get("events", {}).get(symbol, {})
        event_lookup = {
            "spike": ("largest_spike", "price spike"),
            "drop": ("largest_drop", "price drop"),
            "trend": ("peak_volatility", "trend"),
            "volatility": ("peak_volatility", "volatility"),
            "explanation": ("recent_reference_move", "recent move"),
        }
        event_key, move_label = event_lookup.get(question_focus, ("peak_volatility", "price move"))
        event = symbol_events.get(event_key)
        if not event:
            return {}
        try:
            return build_event_context(symbol=symbol, event_date=event["trade_date"], move_label=move_label)
        except Exception:
            return {}

    def _format_external_context(self, external_context: dict[str, Any]) -> str:
        """Render external context with explicit uncertainty and attribution."""
        themes = external_context.get("themes", [])
        if themes:
            return (
                "Possible external context near that move points most consistently to "
                + ", ".join(themes[:3])
                + ". Supporting article links are listed below."
            )
        article_count = len(external_context.get("articles", []))
        if article_count:
            return "I found nearby articles, but they did not point to a clear repeated theme. Supporting links are listed below."
        return ""

    def _format_data_only_explanation(self, event_summary: dict[str, Any], symbol: str, question: str) -> str:
        """Explain a recent move using only price behavior observed in the local data."""
        recent_return = event_summary.get("recent_return_5d")
        recent_move = event_summary.get("recent_reference_move")
        peak = event_summary.get("peak_volatility")
        if recent_return is None:
            return ""
        direction = "up" if recent_return >= 0 else "down"
        parts = [f"{symbol} is {direction} {recent_return:+.2%} over the last 5 trading days in the local data."]
        if recent_move:
            move_word = "rose" if recent_move["daily_return"] >= 0 else "fell"
            parts.append(
                f"The most notable day in that stretch was {recent_move['trade_date']}, when it {move_word} {recent_move['daily_return']:+.2%} and closed at {recent_move['close']:.2f}."
            )
        if peak:
            parts.append(
                f"The highest 20-day realized volatility in this window was on {peak['trade_date']} at {peak['rolling_vol_20d']:.2%}, so the recent move happened alongside elevated price swings rather than a smooth trend."
            )
        parts.append("That explains the move from the price series itself, but not the external catalyst.")
        return " ".join(parts)

    def _clamp_range_to_available_data(self, start_date: str, end_date: str, latest_available_date: str) -> tuple[str, str]:
        """Shift a requested window back to the latest date present in local data."""
        requested_start = date.fromisoformat(start_date)
        requested_end = date.fromisoformat(end_date)
        available_end = date.fromisoformat(latest_available_date)
        if available_end >= requested_end:
            return start_date, end_date
        window_days = max((requested_end - requested_start).days, 1)
        adjusted_start = available_end.fromordinal(available_end.toordinal() - window_days)
        return adjusted_start.isoformat(), available_end.isoformat()

    def _format_biggest_spike(self, event_summary: dict[str, Any], symbol: str) -> str:
        """Describe the largest positive daily move for a symbol."""
        spike = event_summary.get("largest_spike")
        if not spike:
            return ""
        return (
            f"The biggest upside move in {symbol} was on {spike['trade_date']}, when it rose {spike['daily_return']:+.2%} "
            f"and closed at {spike['close']:.2f}."
        )

    def _format_biggest_drop(self, event_summary: dict[str, Any], symbol: str) -> str:
        """Describe the largest negative daily move for a symbol."""
        drop = event_summary.get("largest_drop")
        if not drop:
            return ""
        return (
            f"The sharpest drop in {symbol} was on {drop['trade_date']}, when it fell {drop['daily_return']:+.2%} "
            f"to {drop['close']:.2f}."
        )

    def _format_peak_volatility(self, event_summary: dict[str, Any], symbol: str) -> str:
        """Describe the date of highest recent rolling volatility."""
        peak = event_summary.get("peak_volatility")
        if not peak:
            return ""
        return f"{symbol} reached its highest 20-day realized volatility on {peak['trade_date']} at {peak['rolling_vol_20d']:.2%}."

    def _format_trend_explanation(self, event_summary: dict[str, Any], symbol: str) -> str:
        """Explain a trend using only local price behavior."""
        if not event_summary:
            return ""
        total_return = event_summary.get("window_return")
        max_drawdown = event_summary.get("max_drawdown")
        peak = event_summary.get("peak_volatility")
        if total_return is None or max_drawdown is None:
            return ""
        direction = "upward" if total_return >= 0 else "downward"
        base = f"{symbol} showed a {direction} trend over this window, with a total return of {total_return:+.2%} and a worst drawdown of {max_drawdown:+.2%}."
        if peak:
            return base + f" Volatility was most elevated on {peak['trade_date']} at {peak['rolling_vol_20d']:.2%}, which lines up with the largest recent price swings in the series."
        return base

    def _summarize_frame(self, frame: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
        """Summarize retrieved data for the narrative and evaluator."""
        if frame.empty:
            return {
                "symbols": symbols,
                "row_count": 0,
                "start_date": None,
                "end_date": None,
                "latest_closes": {},
                "pct_change": {},
                "volatility": {},
                "drawdown": {},
                "events": {},
            }

        ordered = frame.sort_values(["symbol", "trade_date"]).copy()
        ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
        latest_rows = ordered.groupby("symbol").tail(1)
        first_rows = ordered.groupby("symbol").head(1).set_index("symbol")

        pct_change: dict[str, float] = {}
        for _, row in latest_rows.iterrows():
            symbol = row["symbol"]
            first_close = float(first_rows.loc[symbol, "close"])
            pct_change[symbol] = float(row["close"] / first_close - 1) if first_close else 0.0

        volatility = (
            ordered.groupby("symbol")["close"]
            .apply(lambda series: float(series.pct_change().dropna().tail(20).std() or 0.0))
            .to_dict()
        )
        drawdown: dict[str, float] = {}
        events: dict[str, dict[str, Any]] = {}
        for symbol, group in ordered.groupby("symbol"):
            group = group.sort_values("trade_date").copy()
            group["daily_return"] = group["close"].pct_change()
            group["rolling_vol_20d"] = group["daily_return"].rolling(window=20, min_periods=5).std()
            running_high = group["close"].cummax()
            drawdown[symbol] = float((group["close"] / running_high - 1).min())
            largest_spike_row = group.loc[group["daily_return"].idxmax()] if group["daily_return"].notna().any() else None
            largest_drop_row = group.loc[group["daily_return"].idxmin()] if group["daily_return"].notna().any() else None
            peak_vol_row = group.loc[group["rolling_vol_20d"].idxmax()] if group["rolling_vol_20d"].notna().any() else None
            recent_group = group.tail(min(5, len(group))).copy()
            recent_reference = recent_group.loc[recent_group["daily_return"].abs().idxmax()] if recent_group["daily_return"].notna().any() else None
            events[symbol] = {
                "window_return": float(group["close"].iloc[-1] / group["close"].iloc[0] - 1) if float(group["close"].iloc[0]) else 0.0,
                "max_drawdown": drawdown[symbol],
                "recent_return_5d": float(recent_group["close"].iloc[-1] / recent_group["close"].iloc[0] - 1) if len(recent_group) > 1 and float(recent_group["close"].iloc[0]) else 0.0,
                "largest_spike": self._event_row_to_dict(largest_spike_row, "daily_return"),
                "largest_drop": self._event_row_to_dict(largest_drop_row, "daily_return"),
                "peak_volatility": self._event_row_to_dict(peak_vol_row, "rolling_vol_20d"),
                "recent_reference_move": self._event_row_to_dict(recent_reference, "daily_return"),
            }

        return {
            "symbols": symbols,
            "row_count": len(ordered),
            "start_date": ordered["trade_date"].min().date().isoformat(),
            "end_date": ordered["trade_date"].max().date().isoformat(),
            "latest_closes": {row["symbol"]: float(row["close"]) for _, row in latest_rows.iterrows()},
            "pct_change": pct_change,
            "volatility": volatility,
            "drawdown": drawdown,
            "events": events,
        }

    def _event_row_to_dict(self, row: pd.Series | None, metric_column: str) -> dict[str, Any] | None:
        """Convert an event row into a compact serializable dictionary."""
        if row is None:
            return None
        metric_value = row.get(metric_column)
        if pd.isna(metric_value):
            return None
        return {
            "trade_date": pd.to_datetime(row["trade_date"]).date().isoformat(),
            "close": float(row["close"]),
            metric_column: float(metric_value),
        }
