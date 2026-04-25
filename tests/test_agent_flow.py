"""Tests for the local agent flow."""

from __future__ import annotations

import signalcanvas.agent as agent_module
from signalcanvas.agent import SignalCanvasAgent
from signalcanvas.config import AppConfig
from signalcanvas.utils import detect_analysis_type, detect_question_focus, parse_symbols


def test_agent_answers_compare_question(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["AAPL", "MSFT", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Show AAPL vs MSFT YTD",
        {
            "default_symbols": ["AAPL", "MSFT", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "AAPL" in response.answer
    assert response.chart_spec is not None
    assert response.chart_spec.color == "symbol"
    assert response.evaluator.grounded is True


def test_parse_symbols_ignores_common_words_and_keeps_nvda() -> None:
    symbols = parse_symbols(
        "Forecast the next 10 days of NVDA",
        fallback=["AAPL"],
        valid_symbols=["AAPL", "MSFT", "NVDA", "SPY"],
    )
    assert symbols == ["NVDA"]


def test_parse_symbols_filters_non_ticker_words_in_compare_question() -> None:
    symbols = parse_symbols(
        "Which of AAPL and MSFT performed better this year?",
        fallback=["SPY", "QQQ"],
        valid_symbols=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
    )
    assert symbols == ["AAPL", "MSFT"]


def test_detect_analysis_type_handles_better_question() -> None:
    assert detect_analysis_type("Which of AAPL and MSFT performed better this year?") == "compare"


def test_detect_question_focus_for_volatility() -> None:
    assert detect_question_focus("Compare the volatility of SPY and NVDA this year") == "volatility"


def test_detect_question_focus_for_better_buy() -> None:
    assert detect_question_focus("What is the better buy, NVDA or AAPL") == "forward_performance"


def test_detect_analysis_type_for_explanation_question() -> None:
    assert detect_analysis_type("Why did NVDA drop last week?") == "anomaly"


def test_agent_performance_answer_mentions_winner(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["AAPL", "MSFT", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Which of AAPL and MSFT performed better this year?",
        {
            "default_symbols": ["AAPL", "MSFT", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "did better" in response.answer
    assert "returned" in response.answer


def test_agent_volatility_answer_mentions_more_volatile(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["AAPL", "MSFT", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Compare the volatility of SPY and NVDA this year",
        {
            "default_symbols": ["SPY", "NVDA", "MSFT"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "more volatile" in response.answer
    assert "realized volatility" in response.answer


def test_agent_better_buy_question_forecasts_both_symbols(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["AAPL", "NVDA", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "What is the better buy, NVDA or AAPL?",
        {
            "default_symbols": ["AAPL", "NVDA", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert response.forecast_used is True
    assert "stronger buy" in response.answer
    assert "projected return" in response.answer
    assert response.figure is not None


def test_agent_spike_question_mentions_date_and_move(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["NVDA", "AAPL", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Find the biggest price spike in NVDA this year",
        {
            "default_symbols": ["NVDA", "AAPL", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "biggest upside move" in response.answer
    assert "rose" in response.answer
    assert "2026-" in response.answer


def test_agent_explanation_question_stays_price_grounded(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["AAPL", "NVDA", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Explain the trend in AAPL over the last month",
        {
            "default_symbols": ["AAPL", "NVDA", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "trend" in response.answer
    assert "total return" in response.answer
    assert "drawdown" in response.answer
    assert "earnings" not in response.answer.lower()


def test_agent_volatility_explanation_mentions_peak_date(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["TSLA", "NVDA", "SPY"])
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "What’s driving volatility in NVDA recently?",
        {
            "default_symbols": ["NVDA", "AAPL", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "highest 20-day realized volatility" in response.answer
    assert "2026-" in response.answer
    assert "investors" not in response.answer.lower()


def test_agent_anomaly_question_can_include_attributed_external_context(db, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_event_context",
        lambda symbol, event_date, move_label, max_items=3: {
            "themes": ["ai demand", "chip strength"],
            "articles": [
                {
                    "title": f"{symbol} headline near move",
                    "link": "https://example.com/story",
                    "published": event_date,
                    "source": "Example News",
                    "description": "Nearby article description",
                    "excerpt": "Nearby article excerpt",
                }
            ],
        },
    )
    config = AppConfig(
        openai_api_key=None,
        db_path=tmp_path / "test.duckdb",
        default_symbols=["NVDA", "AAPL", "SPY"],
        enable_web_context=True,
    )
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Why did NVDA drop last week?",
        {
            "default_symbols": ["NVDA", "AAPL", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "Possible external context" in response.answer
    assert "ai demand" in response.answer
    assert response.external_context is not None
    assert response.external_context["themes"] == ["ai demand", "chip strength"]


def test_agent_why_question_uses_explanation_path(db, tmp_path) -> None:
    config = AppConfig(openai_api_key=None, db_path=tmp_path / "test.duckdb", default_symbols=["NVDA", "AAPL", "SPY"], enable_web_context=False)
    agent = SignalCanvasAgent(config=config, db=db)
    response = agent.answer(
        "Why is NVDA up over the last few days?",
        {
            "default_symbols": ["NVDA", "AAPL", "SPY"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-17",
            "forecast_horizon": 10,
            "forecast_method": "auto",
        },
    )
    assert "over the last 5 trading days" in response.answer
    assert "external catalyst" in response.answer
