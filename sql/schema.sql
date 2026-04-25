CREATE TABLE IF NOT EXISTS raw_prices (
    symbol TEXT,
    trade_date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adjusted_close DOUBLE,
    volume BIGINT,
    source TEXT,
    ingested_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT,
    trade_date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adjusted_close DOUBLE,
    volume BIGINT,
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS price_features (
    symbol TEXT,
    trade_date DATE,
    close DOUBLE,
    daily_return DOUBLE,
    rolling_7d_avg DOUBLE,
    rolling_30d_avg DOUBLE,
    rolling_vol_20d DOUBLE,
    drawdown_from_90d_high DOUBLE,
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS analysis_log (
    id TEXT PRIMARY KEY,
    user_question TEXT,
    symbols TEXT,
    sql_used TEXT,
    chart_spec_json TEXT,
    narrative TEXT,
    evaluator_score DOUBLE,
    created_at TIMESTAMP
);
