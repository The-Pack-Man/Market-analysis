CREATE OR REPLACE VIEW v_latest_prices AS
SELECT dp.*
FROM daily_prices AS dp
INNER JOIN (
    SELECT symbol, MAX(trade_date) AS max_trade_date
    FROM daily_prices
    GROUP BY symbol
) AS latest
    ON dp.symbol = latest.symbol
   AND dp.trade_date = latest.max_trade_date;

CREATE OR REPLACE VIEW v_returns AS
SELECT
    symbol,
    trade_date,
    close,
    close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date), 0) - 1 AS daily_return
FROM daily_prices;

CREATE OR REPLACE VIEW v_volatility AS
WITH base AS (
    SELECT
        symbol,
        trade_date,
        close,
        close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date), 0) - 1 AS daily_return
    FROM daily_prices
)
SELECT
    symbol,
    trade_date,
    close,
    STDDEV_SAMP(daily_return)
        OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS rolling_vol_20d
FROM base;

CREATE OR REPLACE VIEW v_drawdowns AS
SELECT
    symbol,
    trade_date,
    close,
    close / NULLIF(MAX(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW), 0) - 1
        AS drawdown_from_90d_high
FROM daily_prices;
