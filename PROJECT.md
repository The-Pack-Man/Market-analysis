---
name: SignalCanvas
repo: The-Pack-Man/Market-analysis
status: active
one_liner: Conversational stock-analysis agent with constrained tools, read-only SQL, validated charts, and a groundedness checker.
problem: LLM analytics assistants overclaim, run unsafe queries, and invent causation. Answers need guardrails and self-checks.
user: Personal use; portfolio demonstration.
tags: [eval-reliability, problem-selection, shipped-evidence]
stack: [Python, Streamlit, DuckDB, Plotly, yfinance, Pydantic, OpenAI API, Ollama, pytest]
run: streamlit run app.py
test: pytest

claims:
  - id: C1
    text: Built a tool-constrained agent with an explicit tool registry (price history, forecast, chart spec) rather than free-form code execution. The app never executes model-generated code.
    evidence: [signalcanvas/agent.py, signalcanvas/tools.py]
  - id: C2
    text: Enforced read-only SQL by allowing only a single SELECT or WITH…SELECT statement.
    evidence: [signalcanvas/tools.py, tests/test_run_sql.py]
  - id: C3
    text: Defined a validated chart-spec contract (Pydantic) mapped to Plotly figures, so the model emits specs, not plotting code.
    evidence: [signalcanvas/models.py, signalcanvas/charts.py, tests/test_chart_spec.py]
  - id: C4
    text: Added a rule-based groundedness evaluator that flags empty results, missing SQL provenance, unattributed external context, causal overclaiming, and forecasts without uncertainty language.
    evidence: [signalcanvas/evaluator.py]
  - id: C5
    text: Implemented baseline forecasts (naive / moving average / damped trend, auto-selected by history length) with explicit uncertainty framing.
    evidence: [signalcanvas/forecasts.py, tests/test_forecasts.py]
  - id: C6
    text: Built a DuckDB data layer with yfinance OHLCV ingestion and derived features (returns, rolling averages, volatility, drawdown).
    evidence: [signalcanvas/ingest.py, signalcanvas/features.py, sql/]
  - id: C7
    text: Wrote a 25-test pytest suite covering agent flow, SQL guard, chart specs, forecasts, and ingestion. All pass.
    evidence: [tests/]
  - id: C8
    text: Added a heuristic fallback when no LLM is configured, plus local Ollama support via the OpenAI-compatible API.
    evidence: [signalcanvas/agent.py]

metrics:
  - id: M1
    name: Unit/integration tests passing
    value: 25
    status: measured
    method: pytest -q (re-run 2026-10-17)
    artifact: ""
    date: 2026-10-17

decisions:
  - The model produces structured specs and SQL. Code execution is ruled out by design.
  - Forecasts are deliberately simple baselines. Positioned as context, not trading advice.

limitations:
  - The groundedness evaluator is keyword/rule-based and is not validated against human labels.
  - No offline eval set measuring answer correctness.
---
