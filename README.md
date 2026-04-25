# SignalCanvas

SignalCanvas is a local-first Streamlit app for conversational stock analysis. It routes natural-language questions through a constrained agent, retrieves data from DuckDB, renders validated Plotly charts, runs baseline forecasts, evaluates groundedness, and logs analyses for review.

## Features

- Streamlit chat UI with persistent session state
- DuckDB-backed local market data store
- Daily OHLCV ingestion using `yfinance`
- Derived features for returns, rolling averages, volatility, and drawdowns
- Explicit tool registry with read-only SQL enforcement
- Validated chart-spec contract mapped to Plotly figures
- Baseline forecasting methods with uncertainty language
- Groundedness evaluator and analysis log
- Graceful heuristic fallback when no OpenAI API key is configured
- Optional local Ollama model support through its OpenAI-compatible API

## Quickstart

1. Create a virtual environment with Python 3.11+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want live model reasoning.
4. Start the app:

```bash
streamlit run app.py
```

## Tests

```bash
pytest
```

## Notes

- The app never executes model-generated code.
- SQL tools are read-only and only allow a single `SELECT` or `WITH ... SELECT` statement.
- Forecasts are baseline estimates, not trading advice or predictive models.

## Local Ollama

SignalCanvas can use a local Ollama model instead of OpenAI.

1. Install and start Ollama.
2. Pull a model, for example:

```bash
ollama pull llama3.2
```

3. Set your `.env` like this:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1/
```

When `LLM_PROVIDER=ollama`, the app uses Ollama's OpenAI-compatible API. If Ollama is unavailable, the app still falls back to the local heuristic answer path.
