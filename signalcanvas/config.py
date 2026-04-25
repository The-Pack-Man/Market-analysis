"""Application configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Runtime configuration for SignalCanvas."""

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    llm_base_url: str | None = None
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434/v1/"
    enable_web_context: bool = True
    db_path: Path = Field(default_factory=lambda: Path("data/signalcanvas.duckdb"))
    default_symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load configuration from environment variables and defaults."""
    load_dotenv()
    raw_symbols = os.getenv("SIGNALCANVAS_DEFAULT_SYMBOLS", "AAPL,MSFT,NVDA,SPY,QQQ")
    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
        enable_web_context=os.getenv("SIGNALCANVAS_ENABLE_WEB_CONTEXT", "true").strip().lower() in {"1", "true", "yes", "on"},
        db_path=Path(os.getenv("SIGNALCANVAS_DB_PATH", "data/signalcanvas.duckdb")),
        default_symbols=[symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()],
    )
