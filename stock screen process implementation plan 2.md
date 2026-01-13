<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# continue

Continuing from the MACD section and then giving you a **full, end‑to‑end implementation plan in one place** that your developer can follow.

***

## 🛠 3. TECHNICAL INDICATORS (CONTINUED)

### MACD usage example

```python
latest_macd = macd.iloc[-1]
latest_signal = signal.iloc[-1]
latest_hist = hist.iloc[-1]

def macd_status(latest_macd, latest_signal, latest_hist):
    if latest_macd > latest_signal and latest_hist > 0:
        return "bullish"
    elif latest_macd < latest_signal and latest_hist < 0:
        return "bearish"
    else:
        return "neutral"
```


### RSI

```python
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Usage
rsi = calculate_rsi(df['close'])
latest_rsi = rsi.iloc[-1]
```

Typical interpretation:

- RSI > 70 → overbought (pullback likely)
- RSI < 30 → oversold (bounce likely)


### Bollinger Bands

```python
def bollinger_bands(prices, period=20, std_mult=2):
    mid = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

upper, mid, lower = bollinger_bands(df['close'])
price = df['close'].iloc[-1]
bb_upper = upper.iloc[-1]
bb_lower = lower.iloc[-1]

def bb_position(price, upper, lower):
    if price >= upper:
        return "OVERBOUGHT"
    elif price <= lower:
        return "OVERSOLD"
    else:
        return "NORMAL"

latest_bb_pos = bb_position(price, bb_upper, bb_lower)
```


### Volume ratio

```python
def volume_ratio(current_volume, volume_series, lookback=20):
    avg_vol = volume_series.tail(lookback).mean()
    if avg_vol == 0:
        return 1.0
    return current_volume / avg_vol
```


***

## 🧠 4. COMPOSITE SCORING WITH TECHNICALS

Combine price abnormality, volume, MACD, RSI, and Bollinger:

```python
def enhanced_score(actual_pct, expected_pct,
                   vol_ratio,
                   macd_state,   # 'bullish'/'bearish'/'neutral'
                   rsi_value,
                   bb_pos,       # 'OVERBOUGHT'/'OVERSOLD'/'NORMAL'
                   direction):   # 'UP' or 'DOWN'
    """
    direction: +move → 'UP'; -move → 'DOWN'
    We assume a *reversion* strategy: we like
    UP move + bearish techs, or DOWN move + bullish techs.
    """
    if expected_pct == 0:
        return 0.0

    abnormality = abs(actual_pct) / abs(expected_pct)
    base = abnormality * 100

    # Volume multiplier
    if vol_ratio > 2:
        vol_mult = 1.5
    elif vol_ratio > 1.5:
        vol_mult = 1.3
    elif vol_ratio > 1.0:
        vol_mult = 1.1
    else:
        vol_mult = 0.7

    # MACD multiplier (reversion logic)
    macd_mult = 1.0
    if direction == 'UP' and macd_state == 'bearish':
        macd_mult = 1.4
    elif direction == 'DOWN' and macd_state == 'bullish':
        macd_mult = 1.4
    elif macd_state == 'neutral':
        macd_mult = 1.0
    else:
        macd_mult = 0.8  # move and MACD agree → less reversion edge

    # RSI multiplier
    if rsi_value >= 80 or rsi_value <= 20:
        rsi_mult = 1.25
    elif rsi_value >= 70 or rsi_value <= 30:
        rsi_mult = 1.10
    else:
        rsi_mult = 1.0

    # Bollinger multiplier
    if bb_pos in ("OVERBOUGHT", "OVERSOLD"):
        bb_mult = 1.15
    else:
        bb_mult = 1.0

    score = base * vol_mult * macd_mult * rsi_mult * bb_mult
    return min(score, 100.0)
```


***

## 🧱 5. MINIMAL DATA MODEL (NO ORM REQUIRED)

Your developer can start simple with in‑memory or CSV, but for real use:

```python
# Example simple SQLite schema (easier to start than Postgres)

CREATE TABLE stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE,
    beta REAL
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timestamp TEXT,
    actual_move REAL,
    expected_move REAL,
    abnormality REAL,
    rating REAL,
    signal_type TEXT,
    direction TEXT,
    volume_ratio REAL,
    macd_state TEXT,
    rsi REAL,
    bb_pos TEXT
);
```


***

## 🔁 6. REAL‑TIME SCREENER LOOP (SIMPLIFIED)

Your developer can wire this to IB’s API:

```python
import time
from datetime import datetime

def get_prev_close(symbol):
    # store yesterday’s close locally or query once each morning
    ...

def get_intraday_history(symbol):
    # pull last 50–100 1‑min bars via IB for MACD/RSI/BB
    ...

def screen_symbol(symbol, beta, vix_level):
    # 1) data
    last_price = ib.get_latest_price(symbol)
    prev_close = get_prev_close(symbol)
    if not last_price or not prev_close:
        return None

    actual_pct = (last_price - prev_close) / prev_close * 100
    direction = 'UP' if actual_pct >= 0 else 'DOWN'

    # 2) expected move
    exp_pct, _ = expected_move(beta, vix_level, last_price)
    if exp_pct == 0:
        return None

    # 3) techs
    hist = get_intraday_history(symbol)      # DataFrame: 'close','volume',...
    macd, sig, hist_macd = calculate_macd(hist['close'])
    rsi = calculate_rsi(hist['close'])
    upper, mid, lower = bollinger_bands(hist['close'])
    vr = volume_ratio(hist['volume'].iloc[-1], hist['volume'])

    macd_state = macd_status(macd.iloc[-1], sig.iloc[-1], hist_macd.iloc[-1])
    bb_pos = bb_position(hist['close'].iloc[-1], upper.iloc[-1], lower.iloc[-1])

    # 4) scoring
    score = enhanced_score(actual_pct, exp_pct,
                           vr, macd_state,
                           rsi.iloc[-1], bb_pos, direction)
    signal = classify_signal(score)

    return {
        "symbol": symbol,
        "price": last_price,
        "actual_pct": actual_pct,
        "expected_pct": exp_pct,
        "score": score,
        "signal": signal,
        "direction": direction,
        "volume_ratio": vr,
        "macd_state": macd_state,
        "rsi": float(rsi.iloc[-1]),
        "bb_pos": bb_pos,
    }

def screening_loop(watchlist, vix_source, threshold=1.5, sleep_sec=5):
    """
    watchlist: list of dicts {symbol, beta}
    vix_source: function returning latest VIX
    """
    while True:
        vix_level = vix_source()
        results = []
        for item in watchlist:
            res = screen_symbol(item['symbol'], item['beta'], vix_level)
            if not res:
                continue
            # abnormality filter
            score_abn = abnormality_score(res['actual_pct'], res['expected_pct'])
            if score_abn >= threshold and res['score'] >= 60:
                results.append(res)
                # TODO: send alert + log to DB

        # Sort and print top signals
        results.sort(key=lambda r: r['score'], reverse=True)
        for r in results:
            print(f"{datetime.now()} | {r['symbol']} | "
                  f"{r['signal']} {r['score']:.1f} | "
                  f"{r['actual_pct']:.2f}% vs exp {r['expected_pct']:.2f}% | "
                  f"Vratio {r['volume_ratio']:.2f} | MACD {r['macd_state']} | "
                  f"RSI {r['rsi']:.1f} | BB {r['bb_pos']}")
        time.sleep(sleep_sec)
```


***

## ⚙️ 7. PYTHON PROJECT STRUCTURE (LIGHTWEIGHT)

Your dev can start with a simplified structure:

```text
project/
  main.py                 # entry point, screening_loop()
  ib_gateway.py           # IB wrapper
  formulas.py             # expected_move, abnormality, scoring
  indicators.py           # MACD, RSI, BB, volume_ratio
  data_store.py           # caching prev close, storing alerts
  config.py               # API keys, ports, thresholds
```


***

## 🧪 8. MINIMAL BACKTESTER

Once data is downloaded (e.g., from IB or yfinance), a simple backtest to tune thresholds:

```python
def backtest_single_symbol(df, beta_series, vix_series,
                           abn_threshold=1.5,
                           hold_days=1):
    """
    df: daily OHLCV, index=dates
    beta_series: beta per day
    vix_series: VIX per day
    """
    trades = []

    for i in range(1, len(df) - hold_days):
        prev_close = df['close'].iloc[i-1]
        cur_close  = df['close'].iloc[i]
        next_close = df['close'].iloc[i+hold_days]

        actual_pct = (cur_close - prev_close) / prev_close * 100
        beta = beta_series.iloc[i]
        vix  = vix_series.iloc[i]

        exp_pct, _ = expected_move(beta, vix, cur_close)
        if exp_pct == 0:
            continue

        abn = abnormality_score(actual_pct, exp_pct)
        if abn < abn_threshold:
            continue   # no signal

        direction = 'UP' if actual_pct > 0 else 'DOWN'
        # mean reversion: fade the move
        # if UP then short next bar; if DOWN then go long
        if direction == 'UP':
            ret = (next_close - cur_close) / cur_close * -100  # short
        else:
            ret = (next_close - cur_close) / cur_close * 100   # long

        trades.append(ret)

    if not trades:
        return {"trades": 0}

    import numpy as np
    arr = np.array(trades)
    win_rate = (arr > 0).mean()
    avg_win  = arr[arr > 0].mean() if (arr > 0).any() else 0
    avg_loss = arr[arr < 0].mean() if (arr < 0).any() else 0
    profit_factor = (arr[arr > 0].sum() / abs(arr[arr < 0].sum())
                     if (arr < 0).any() else None)

    return {
        "trades": len(arr),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }
```


***

## 📌 9. CONFIGURATION (FOR YOUR ENV + IB)

```python
# config.py
IB_HOST = "127.0.0.1"
IB_PORT = 7497          # paper trading
IB_CLIENT_ID = 1

SCAN_INTERVAL = 5       # seconds
ABN_THRESHOLD_NORMAL = 1.5
ABN_THRESHOLD_HIGH_VIX = 1.2
VIX_HIGH_LEVEL = 25

WATCHLIST = [
    {"symbol": "AAPL", "beta": 1.2},
    {"symbol": "TSLA", "beta": 2.0},
    {"symbol": "QQQ",  "beta": 1.3},
    {"symbol": "IWM",  "beta": 1.1},
]
```


***

## ✅ 10. HOW TO HAND THIS TO YOUR DEVELOPER

Tell them:

1. **Goal:**
Implement a real‑time screener using IB Gateway + Python that:
    - Computes Expected Move = Beta × VIX / 100
    - Detects Abnormality Score > threshold
    - Confirms with Volume, MACD, RSI, Bollinger
    - Scores opportunities 0–100 and raises alerts
2. **Use this answer as the single specification:**
    - Implement `ib_gateway.py` from the snippet
    - Implement `formulas.py` with expected_move, abnormality_score, opportunity_rating, enhanced_score, classify_signal
    - Implement `indicators.py` with MACD/RSI/BB/volume_ratio
    - Implement `main.py` with `screening_loop()` as above
    - Add simple logging or database writes for alerts
3. **Integration specifics:**
    - Use IB Gateway / TWS API for live prices and 1‑min bars
    - Use `^VIX` or your preferred VIX source for volatility
    - Use your own watchlist with betas (from IB or your DB)

If you want, next step can be:

- a single monolithic `main.py` file consolidating all of this into one script your developer can run immediately.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: image.jpg

