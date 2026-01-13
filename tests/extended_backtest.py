"""
Extended Backtest Suite
=======================

Comprehensive backtesting with:
1. Different beta values (0.5, 1.0, 1.5, 2.0)
2. Different VIX levels (12, 20, 35, 50)
3. Different stop widths (4%, 6%, 8%, 10%)
4. Historical data simulation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest_engine import BacktestEngine, BacktestTrade
from models import get_beta

logging.basicConfig(level=logging.WARNING, format='%(message)s')


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def generate_price_path(
    start: float, 
    trend: float,  # % change per step 
    volatility: float,
    steps: int = 78  # 6.5 hours @ 5-min bars
) -> list:
    """Generate a price path with trend and noise."""
    random.seed(42)
    prices = [start]
    
    for _ in range(steps):
        drift = trend / steps
        noise = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + drift + noise/100)
        prices.append(max(new_price, start * 0.7))
    
    return prices


def run_beta_comparison():
    """Test how different beta stocks perform."""
    print_header("TEST 1: BETA COMPARISON")
    print("\nSame VIX (20), same price path, different betas")
    
    # Price path: moderate down then recovery
    base_prices = [100, 98, 96, 94, 92, 94, 96, 98, 100, 102]
    
    betas = [0.5, 1.0, 1.5, 2.0]
    results = []
    
    for beta in betas:
        engine = BacktestEngine(
            k_aggression=1.0,
            min_trail_pct=0.04,
            vix_level=20.0
        )
        
        entry_time = datetime(2026, 1, 9, 10, 30)
        
        key = engine.open_position(
            symbol="TEST",
            expiry="20260220",
            strike=100.0,
            right="C",
            entry_price=5.00,
            entry_underlying=100.0,
            beta=beta,
            entry_time=entry_time
        )
        
        # Scale price moves by beta
        for idx, price in enumerate(base_prices):
            sim_time = entry_time + timedelta(minutes=idx*30)
            
            # Option price moves with delta
            delta = 0.5
            underlying_move = price - 100
            option_mid = 5.00 + delta * underlying_move * beta
            option_mid = max(option_mid, 0.10)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            # Scale underlying move by beta
            scaled_underlying = 100 + (price - 100) * beta
            
            engine.update_position(key, bid, ask, scaled_underlying, sim_time)
            
            if key not in engine.positions:
                break
        
        # Close remaining
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(key, pos.current_bid, pos.current_underlying,
                                  engine.current_time, "end_of_day")
        
        if engine.closed_trades:
            trade = engine.closed_trades[0]
            stop_pct = engine.stop_calc.get_trail_percentage(beta, 0.0126, 42)
            results.append({
                'beta': beta,
                'stop_pct': stop_pct * 100,
                'pnl': trade.realized_pnl,
                'reason': trade.exit_reason
            })
    
    print(f"\n{'Beta':<8} {'Stop %':<10} {'P&L':<12} {'Exit Reason':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['beta']:<8.1f} {r['stop_pct']:<10.1f}% ${r['pnl']:+8.2f}   {r['reason']}")


def run_vix_comparison():
    """Test how different VIX levels affect stops."""
    print_header("TEST 2: VIX LEVEL COMPARISON")
    print("\nSame beta (1.0), same price path, different VIX levels")
    
    # Moderate decline
    prices = [585, 580, 575, 570, 565, 560, 555]
    
    vix_levels = [12, 20, 35, 50]
    results = []
    
    for vix in vix_levels:
        engine = BacktestEngine(
            k_aggression=1.0,
            min_trail_pct=0.04,
            vix_level=vix
        )
        
        entry_time = datetime(2026, 1, 9, 10, 30)
        
        key = engine.open_position(
            symbol="SPY",
            expiry="20260220",
            strike=585.0,
            right="C",
            entry_price=12.00,
            entry_underlying=585.0,
            beta=1.0,
            entry_time=entry_time
        )
        
        for idx, underlying in enumerate(prices):
            sim_time = entry_time + timedelta(minutes=idx*30)
            
            delta = 0.5
            option_mid = 12.00 + delta * (underlying - 585)
            option_mid = max(option_mid, 0.10)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(key, bid, ask, underlying, sim_time)
            
            if key not in engine.positions:
                break
        
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(key, pos.current_bid, pos.current_underlying,
                                  engine.current_time, "end_of_day")
        
        if engine.closed_trades:
            trade = engine.closed_trades[0]
            stop_pct = engine.stop_calc.get_trail_percentage(1.0, vix/100/15.87, 42)
            results.append({
                'vix': vix,
                'stop_pct': stop_pct * 100,
                'pnl': trade.realized_pnl,
                'reason': trade.exit_reason
            })
    
    print(f"\n{'VIX':<8} {'Stop %':<10} {'P&L':<12} {'Exit Reason':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['vix']:<8} {r['stop_pct']:<10.1f}% ${r['pnl']:+8.2f}   {r['reason']}")


def run_stop_width_comparison():
    """Test different stop widths."""
    print_header("TEST 3: STOP WIDTH COMPARISON")
    print("\nSame scenario, different minimum trail percentages")
    
    # Moderate decline then recovery
    prices = [585, 580, 575, 570, 565, 560, 565, 570, 580, 590]
    
    min_trails = [0.04, 0.06, 0.08, 0.10]
    results = []
    
    for min_trail in min_trails:
        engine = BacktestEngine(
            k_aggression=1.0,
            min_trail_pct=min_trail,
            vix_level=20.0
        )
        
        entry_time = datetime(2026, 1, 9, 10, 30)
        
        key = engine.open_position(
            symbol="SPY",
            expiry="20260220",
            strike=585.0,
            right="C",
            entry_price=12.00,
            entry_underlying=585.0,
            beta=1.0,
            entry_time=entry_time
        )
        
        for idx, underlying in enumerate(prices):
            sim_time = entry_time + timedelta(minutes=idx*30)
            
            delta = 0.5
            option_mid = 12.00 + delta * (underlying - 585)
            option_mid = max(option_mid, 0.10)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(key, bid, ask, underlying, sim_time)
            
            if key not in engine.positions:
                break
        
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(key, pos.current_bid, pos.current_underlying,
                                  engine.current_time, "end_of_day")
        
        if engine.closed_trades:
            trade = engine.closed_trades[0]
            results.append({
                'min_trail': min_trail * 100,
                'pnl': trade.realized_pnl,
                'reason': trade.exit_reason,
                'exit_underlying': trade.exit_underlying
            })
    
    print(f"\n{'Min Trail':<12} {'P&L':<12} {'Exit Reason':<15} {'Exit Price':<12}")
    print("-" * 55)
    for r in results:
        print(f"{r['min_trail']:<12.0f}% ${r['pnl']:+8.2f}   {r['reason']:<15} ${r['exit_underlying']:.2f}")


def run_monte_carlo():
    """Run many random simulations."""
    print_header("TEST 4: MONTE CARLO SIMULATION (100 trades)")
    print("\nRandom price paths with various trends and volatilities")
    
    engine = BacktestEngine(
        k_aggression=1.0,
        min_trail_pct=0.06,
        vix_level=22.0
    )
    
    random.seed(123)
    
    trends = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]  # Daily trends
    volatilities = [0.2, 0.3, 0.4, 0.5]  # Noise level
    
    for trade_num in range(100):
        trend = random.choice(trends)
        vol = random.choice(volatilities)
        
        entry_time = datetime(2026, 1, 9, 10, 30)
        engine.current_time = entry_time
        
        key = f"TRADE_{trade_num}"
        
        # Random entry
        entry_underlying = 100.0
        entry_price = 5.00
        beta = random.choice([0.8, 1.0, 1.2, 1.5])
        
        engine.positions.clear()  # Reset for each trade
        
        key = engine.open_position(
            symbol="TEST",
            expiry="20260220",
            strike=100.0,
            right="C",
            entry_price=entry_price,
            entry_underlying=entry_underlying,
            beta=beta,
            entry_time=entry_time
        )
        
        # Generate price path
        prices = generate_price_path(entry_underlying, trend, vol, 78)
        
        for idx, underlying in enumerate(prices[1:]):
            sim_time = entry_time + timedelta(minutes=idx*5)
            
            delta = 0.5
            option_mid = entry_price + delta * (underlying - entry_underlying)
            option_mid = max(option_mid, 0.05)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(key, bid, ask, underlying, sim_time)
            
            if key not in engine.positions:
                break
        
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(key, pos.current_bid, pos.current_underlying,
                                  engine.current_time, "end_of_day")
    
    # Summary
    engine.print_summary()
    
    # Distribution
    pnls = [t.realized_pnl for t in engine.closed_trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    
    print("\nP&L Distribution:")
    print(f"  Biggest Winner: ${max(pnls):+.2f}")
    print(f"  Biggest Loser:  ${min(pnls):+.2f}")
    print(f"  Median P&L:     ${sorted(pnls)[len(pnls)//2]:+.2f}")
    
    # Histogram
    print("\nP&L Histogram:")
    buckets = [(-1000, -500), (-500, -250), (-250, 0), (0, 250), (250, 500), (500, 1000), (1000, float('inf'))]
    
    for low, high in buckets:
        count = sum(1 for p in pnls if low <= p < high)
        bar = "█" * count
        label = f"${low:+d} to ${high:+d}" if high != float('inf') else f"${low:+d}+"
        print(f"  {label:18s} | {bar} ({count})")


def run_multi_position():
    """Test portfolio of different stocks."""
    print_header("TEST 5: MULTI-POSITION PORTFOLIO")
    print("\n5 positions with different betas, same market scenario")
    
    stocks = [
        ("JNJ", 0.65, 155.0, 3.50),
        ("AAPL", 1.25, 240.0, 8.00),
        ("SPY", 1.00, 585.0, 12.00),
        ("NVDA", 1.80, 140.0, 10.00),
        ("TSLA", 2.00, 400.0, 18.00),
    ]
    
    # Market scenario: -3% broad decline
    market_moves = [0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
    
    engine = BacktestEngine(
        k_aggression=1.0,
        min_trail_pct=0.04,
        vix_level=22.0
    )
    
    entry_time = datetime(2026, 1, 9, 10, 30)
    keys = []
    
    # Open all positions
    for symbol, beta, underlying, option_price in stocks:
        key = engine.open_position(
            symbol=symbol,
            expiry="20260220",
            strike=underlying,
            right="C",
            entry_price=option_price,
            entry_underlying=underlying,
            beta=beta,
            entry_time=entry_time
        )
        keys.append((key, stocks[keys.__len__() if keys else 0]))
    
    keys = [(f"{s[0]}_20260220_{s[2]}_C", s) for s in stocks]
    
    # Simulate market moves
    for idx, move_pct in enumerate(market_moves):
        sim_time = entry_time + timedelta(minutes=idx*30)
        engine.current_time = sim_time
        
        for key, (symbol, beta, entry_und, entry_opt) in keys:
            if key not in engine.positions:
                continue
            
            # Each stock moves by beta × market move
            stock_move = move_pct * beta
            current_und = entry_und * (1 + stock_move/100)
            
            delta = 0.5
            option_mid = entry_opt + delta * (current_und - entry_und)
            option_mid = max(option_mid, 0.10)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(key, bid, ask, current_und, sim_time)
    
    # Close remaining
    for key, _ in keys:
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(key, pos.current_bid, pos.current_underlying,
                                  engine.current_time, "end_of_day")
    
    # Summary
    print(f"\n{'Symbol':<8} {'Beta':<6} {'P&L':<12} {'Exit Reason':<15}")
    print("-" * 45)
    
    for trade in engine.closed_trades:
        print(f"{trade.symbol:<8} {trade.beta:<6.2f} ${trade.realized_pnl:+8.2f}   {trade.exit_reason}")
    
    total = sum(t.realized_pnl for t in engine.closed_trades)
    print("-" * 45)
    print(f"{'TOTAL':<8} {'':6} ${total:+8.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXTENDED BACKTEST SUITE")
    print("=" * 70)
    
    run_beta_comparison()
    run_vix_comparison()
    run_stop_width_comparison()
    run_monte_carlo()
    run_multi_position()
    
    print("\n" + "=" * 70)
    print("✓ All backtests complete")
    print("=" * 70 + "\n")
