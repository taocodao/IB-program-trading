"""
Historical Data Replay Test
============================

Simulates the full trading system using synthetic historical data:
1. Screen watchlist for signals
2. Trigger entries on breakout
3. Buy options
4. Monitor with trailing stops
5. Sell on stop trigger

This runs WITHOUT IB connection - pure Python simulation.
"""

import sys
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from screener.formulas import (
    expected_move, abnormality_score, enhanced_score, 
    classify_signal, get_direction
)
from screener.indicators import (
    calculate_macd, calculate_rsi, bollinger_bands,
    macd_status, bb_position, volume_ratio
)
from models import get_beta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

WATCHLIST_SAMPLE = [
    ("AAPL", 1.25, 259.0),
    ("TSLA", 2.00, 445.0),
    ("NVDA", 1.80, 185.0),
    ("AMD", 1.60, 203.0),
    ("META", 1.30, 653.0),
    ("MSFT", 1.10, 479.0),
    ("AMZN", 1.20, 247.0),
    ("GOOG", 1.15, 330.0),
    ("SPY", 1.00, 627.0),
    ("QQQ", 1.30, 627.0),
]

VIX_LEVEL = 22.0
ENTRY_TRAIL_PCT = 0.02
EXIT_TRAIL_PCT = 0.06
ABN_THRESHOLD = 2.0      # Raised from 1.5 to reduce noise
MIN_SCORE = 75           # Raised from 60 for higher quality signals


@dataclass 
class SimPosition:
    symbol: str
    beta: float
    entry_price: float
    entry_underlying: float
    entry_time: datetime
    stop_level: float
    underlying_high: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0


def generate_price_path(start: float, days: int = 5, volatility: float = 0.02) -> List[tuple]:
    """Generate realistic intraday price path."""
    random.seed(42)
    np.random.seed(42)
    
    prices = []
    current = start
    base_time = datetime(2026, 1, 6, 9, 30)  # Monday 9:30 AM
    
    for day in range(days):
        day_start = base_time + timedelta(days=day)
        
        # Daily bias
        daily_trend = random.choice([-0.01, -0.005, 0, 0.005, 0.01, 0.015])
        
        # 78 bars per day (6.5 hours @ 5-min bars)
        for bar in range(78):
            bar_time = day_start + timedelta(minutes=bar * 5)
            
            # Price movement
            noise = np.random.randn() * volatility / 100
            trend = daily_trend / 78
            current *= (1 + trend + noise)
            
            prices.append((bar_time, current))
    
    return prices


def generate_ohlcv(prices: List[tuple]) -> pd.DataFrame:
    """Convert price path to OHLCV DataFrame."""
    data = []
    for i in range(0, len(prices), 5):  # Group into 5-min bars
        chunk = prices[i:i+5]
        if not chunk:
            continue
        
        close_prices = [p[1] for p in chunk]
        data.append({
            'timestamp': chunk[-1][0],
            'open': close_prices[0],
            'high': max(close_prices) * 1.001,
            'low': min(close_prices) * 0.999,
            'close': close_prices[-1],
            'volume': random.randint(100000, 500000)
        })
    
    return pd.DataFrame(data)


def run_historical_replay():
    """Run full historical replay simulation."""
    print("\n" + "=" * 70)
    print("HISTORICAL DATA REPLAY TEST")
    print("=" * 70)
    print(f"Watchlist: {len(WATCHLIST_SAMPLE)} symbols")
    print(f"VIX: {VIX_LEVEL}")
    print(f"Entry Trail: {ENTRY_TRAIL_PCT*100:.0f}%")
    print(f"Exit Trail: {EXIT_TRAIL_PCT*100:.0f}%")
    print("=" * 70 + "\n")
    
    # Generate price paths for each symbol
    symbol_data = {}
    for symbol, beta, start_price in WATCHLIST_SAMPLE:
        vol = 0.02 * beta  # Higher beta = more volatile
        prices = generate_price_path(start_price, days=5, volatility=vol)
        df = generate_ohlcv(prices)
        symbol_data[symbol] = {
            'prices': prices,
            'df': df,
            'beta': beta,
            'prev_close': start_price
        }
    
    # Tracking
    entry_lows: Dict[str, float] = {s[0]: float('inf') for s in WATCHLIST_SAMPLE}
    entry_triggered: Dict[str, bool] = {s[0]: False for s in WATCHLIST_SAMPLE}
    positions: List[SimPosition] = []
    closed_positions: List[SimPosition] = []
    signals_found = 0
    
    print("--- Simulation Start ---\n")
    
    # Iterate through time
    all_times = sorted(set(p[0] for data in symbol_data.values() for p in data['prices']))
    
    for i, current_time in enumerate(all_times):
        # Update each symbol
        for symbol, data in symbol_data.items():
            # Find current price
            current_price = None
            for t, p in data['prices']:
                if t == current_time:
                    current_price = p
                    break
            
            if not current_price:
                continue
            
            beta = data['beta']
            prev_close = data['prev_close']
            
            # ===== SCREENING =====
            actual_pct = (current_price - prev_close) / prev_close * 100
            exp_pct, _ = expected_move(beta, VIX_LEVEL, current_price)
            abn = abnormality_score(actual_pct, exp_pct)
            
            # Get indicators from OHLCV
            df = data['df']
            df_current = df[df['timestamp'] <= current_time]
            
            if len(df_current) >= 26:
                close = df_current['close']
                macd, sig, hist = calculate_macd(close)
                rsi = calculate_rsi(close)
                upper, mid, lower = bollinger_bands(close)
                vol_r = volume_ratio(df_current['volume'].iloc[-1], df_current['volume'])
                
                macd_st = macd_status(macd.iloc[-1], sig.iloc[-1], hist.iloc[-1])
                bb_pos = bb_position(close.iloc[-1], upper.iloc[-1], lower.iloc[-1])
            else:
                macd_st, rsi_val, bb_pos, vol_r = 'neutral', 50, 'NORMAL', 1.0
                rsi = pd.Series([50])
            
            direction = get_direction(actual_pct)
            score = enhanced_score(
                actual_pct, exp_pct, vol_r, macd_st,
                rsi.iloc[-1] if len(rsi) > 0 else 50, bb_pos, direction
            )
            
            # Check for signal
            if abn >= ABN_THRESHOLD and score >= MIN_SCORE:
                if not entry_triggered[symbol]:
                    signals_found += 1
                    print(f"[{current_time.strftime('%m/%d %H:%M')}] SIGNAL: {symbol} | "
                          f"Move: {actual_pct:+.2f}% | Abn: {abn:.1f}x | "
                          f"Score: {score:.0f} ({classify_signal(score)})")
            
            # ===== TRAILING ENTRY =====
            if not entry_triggered[symbol] and not any(p.symbol == symbol for p in positions):
                # Update low
                if current_price < entry_lows[symbol]:
                    entry_lows[symbol] = current_price
                
                # Check trigger
                trail_level = entry_lows[symbol] * (1 + ENTRY_TRAIL_PCT)
                if current_price >= trail_level and abn >= ABN_THRESHOLD:
                    entry_triggered[symbol] = True
                    
                    # Simulate option purchase
                    option_price = current_price * 0.02  # ~2% of underlying
                    stop = current_price * (1 - EXIT_TRAIL_PCT)
                    
                    pos = SimPosition(
                        symbol=symbol,
                        beta=beta,
                        entry_price=option_price,
                        entry_underlying=current_price,
                        entry_time=current_time,
                        stop_level=stop,
                        underlying_high=current_price
                    )
                    positions.append(pos)
                    
                    print(f"[{current_time.strftime('%m/%d %H:%M')}] *** BUY *** {symbol} CALL @ ${option_price:.2f} | "
                          f"Underlying: ${current_price:.2f} | Stop: ${stop:.2f}")
            
            # ===== POSITION MONITORING =====
            for pos in positions[:]:  # Copy to allow removal
                if pos.symbol != symbol:
                    continue
                
                # Update trailing stop
                if current_price > pos.underlying_high:
                    pos.underlying_high = current_price
                    new_stop = current_price * (1 - EXIT_TRAIL_PCT)
                    if new_stop > pos.stop_level:
                        pos.stop_level = new_stop
                
                # Check stop trigger
                if current_price <= pos.stop_level:
                    # Calculate exit price
                    underlying_move = current_price - pos.entry_underlying
                    option_move = underlying_move * 0.5 * 0.01  # Delta approx
                    exit_price = max(pos.entry_price + option_move, 0.05)
                    
                    pos.exit_price = exit_price
                    pos.exit_time = current_time
                    pos.pnl = (exit_price - pos.entry_price) * 100
                    
                    print(f"[{current_time.strftime('%m/%d %H:%M')}] *** SELL *** {symbol} @ ${exit_price:.2f} | "
                          f"P&L: ${pos.pnl:+.2f}")
                    
                    closed_positions.append(pos)
                    positions.remove(pos)
    
    # Close remaining positions at end
    print("\n--- End of Simulation ---")
    for pos in positions:
        data = symbol_data[pos.symbol]
        final_price = data['prices'][-1][1]
        underlying_move = final_price - pos.entry_underlying
        option_move = underlying_move * 0.5 * 0.01
        exit_price = max(pos.entry_price + option_move, 0.05)
        
        pos.exit_price = exit_price
        pos.exit_time = all_times[-1]
        pos.pnl = (exit_price - pos.entry_price) * 100
        
        print(f"[END] Closing {pos.symbol} @ ${exit_price:.2f} | P&L: ${pos.pnl:+.2f}")
        closed_positions.append(pos)
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Signals Detected:    {signals_found}")
    print(f"  Positions Opened:    {len(closed_positions)}")
    
    if closed_positions:
        pnls = [p.pnl for p in closed_positions]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        print(f"\n  Trades:")
        for pos in closed_positions:
            result = "WIN" if pos.pnl > 0 else "LOSS"
            print(f"    {pos.symbol:6s}: Entry ${pos.entry_price:.2f} → Exit ${pos.exit_price:.2f} | "
                  f"P&L: ${pos.pnl:+7.2f} ({result})")
        
        print(f"\n  Total P&L:           ${sum(pnls):+.2f}")
        print(f"  Winners:             {len(winners)}/{len(pnls)} ({len(winners)/len(pnls)*100:.0f}%)")
        print(f"  Avg Winner:          ${sum(winners)/len(winners) if winners else 0:+.2f}")
        print(f"  Avg Loser:           ${sum(losers)/len(losers) if losers else 0:.2f}")
        
        if losers:
            pf = abs(sum(winners)) / abs(sum(losers))
            print(f"  Profit Factor:       {pf:.2f}")
    
    print("=" * 70 + "\n")
    
    return closed_positions


if __name__ == "__main__":
    run_historical_replay()
