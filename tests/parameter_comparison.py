"""
Parameter Tuning Test
=====================

Compare different stop-loss configurations to find the right balance between:
1. Protecting profits (tight stops)
2. Not getting bumped out (wide stops)

Run with: python tests/parameter_comparison.py
"""

import sys
from pathlib import Path
from typing import List
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import OptionPosition, VolatilityTracker
from stop_calculator import StopCalculator


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def simulate_day(pos_entry: float, beta: float, volatility: float, 
                 trend: float, stop_calc: StopCalculator, vol_tracker: VolatilityTracker) -> dict:
    """
    Simulate a day and return results.
    
    Args:
        trend: daily trend (-0.03 = 3% down, +0.02 = 2% up)
    """
    index_vol = vol_tracker.get_daily_vol_pct()
    dte = 30
    
    # Initialize
    underlying_high = pos_entry
    stop_level = stop_calc.compute_underlying_stop(pos_entry, beta, index_vol, dte)
    
    # Simulate 78 5-min bars
    price = pos_entry
    max_price = pos_entry
    min_price = pos_entry
    triggered = False
    exit_price = None
    
    random.seed(42)  # Reproducible
    
    for i in range(78):
        # Price movement: trend + noise
        noise = random.gauss(0, volatility * beta)
        drift = trend / 78
        price = price * (1 + drift + noise / 100)
        
        max_price = max(max_price, price)
        min_price = min(min_price, price)
        
        # Trail stop up
        if price > underlying_high:
            underlying_high = price
            new_stop = stop_calc.compute_trail_from_high(underlying_high, beta, index_vol, dte)
            stop_level = max(stop_level, new_stop)
        
        # Check trigger
        if price <= stop_level and not triggered:
            triggered = True
            exit_price = price
            break
    
    if not triggered:
        exit_price = price
    
    return {
        "triggered": triggered,
        "exit_price": exit_price,
        "pnl_pct": (exit_price - pos_entry) / pos_entry * 100,
        "max_gain": (max_price - pos_entry) / pos_entry * 100,
        "max_dd": (min_price - pos_entry) / pos_entry * 100,
        "stop_level": stop_level,
        "trail_pct": stop_calc.get_trail_percentage(beta, index_vol, dte) * 100
    }


def run_comparison():
    """Compare different parameter configurations."""
    
    print_header("PARAMETER TUNING: FIND YOUR IDEAL STOP SETTINGS")
    
    # Different configurations to test
    configs = [
        {"name": "TIGHT (Current)", "k": 1.0, "min": 0.04, "max": 0.40},
        {"name": "MODERATE", "k": 1.2, "min": 0.06, "max": 0.40},
        {"name": "WIDE (Recommended)", "k": 1.5, "min": 0.08, "max": 0.40},
        {"name": "VERY WIDE", "k": 2.0, "min": 0.10, "max": 0.50},
    ]
    
    vol_tracker = VolatilityTracker()
    vol_tracker.update_vix(22.0)
    
    # Test scenarios
    scenarios = [
        {"name": "Normal Bull Day (+1%)", "trend": 0.01, "vol": 0.3},
        {"name": "Sharp V-Dip (-3% then +4%)", "trend": -0.005, "vol": 0.5},
        {"name": "Steady Bear (-2%)", "trend": -0.02, "vol": 0.2},
        {"name": "Choppy Flat (±0.5%)", "trend": 0.0, "vol": 0.4},
    ]
    
    # Test stock: SPY (beta 1.0)
    entry = 585.0
    beta = 1.0
    
    print(f"\nTest: SPY at ${entry:.2f} (Beta: {beta})")
    print(f"VIX: {vol_tracker.vix_level}")
    
    # ========== SHOW STOP DISTANCES ==========
    print_header("STOP DISTANCE COMPARISON")
    
    print(f"\n{'Config':<20} {'Trail %':<10} {'Stop Level':<12} {'Room ($)':<10}")
    print("-" * 55)
    
    for cfg in configs:
        calc = StopCalculator(k_aggression=cfg["k"], min_trail_pct=cfg["min"], max_trail_pct=cfg["max"])
        index_vol = vol_tracker.get_daily_vol_pct()
        trail_pct = calc.get_trail_percentage(beta, index_vol, 30)
        stop = calc.compute_underlying_stop(entry, beta, index_vol, 30)
        room = entry - stop
        
        print(f"{cfg['name']:<20} {trail_pct*100:<10.1f}% ${stop:<11.2f} ${room:<9.2f}")
    
    # ========== SCENARIO TESTS ==========
    for scenario in scenarios:
        print_header(f"SCENARIO: {scenario['name']}")
        
        print(f"\n{'Config':<20} {'Stopped?':<10} {'Exit':<10} {'P&L':<10} {'Max Gain':<12} {'Max DD':<10}")
        print("-" * 75)
        
        for cfg in configs:
            calc = StopCalculator(k_aggression=cfg["k"], min_trail_pct=cfg["min"], max_trail_pct=cfg["max"])
            
            result = simulate_day(
                entry, beta, scenario["vol"], scenario["trend"],
                calc, vol_tracker
            )
            
            stopped = "YES" if result["triggered"] else "no"
            print(f"{cfg['name']:<20} {stopped:<10} ${result['exit_price']:<9.2f} "
                  f"{result['pnl_pct']:>+7.2f}%  {result['max_gain']:>+7.2f}%     "
                  f"{result['max_dd']:>+7.2f}%")
    
    # ========== V-DIP SPECIAL TEST ==========
    print_header("SPECIAL TEST: V-SHAPED DIP (Sharp drop then recovery)")
    
    print("\nSimulating: Price drops 5% in 30 min, then recovers to +2%")
    print("Goal: Stay in position through the dip to capture recovery\n")
    
    print(f"{'Config':<20} {'Survived Dip?':<15} {'Final P&L':<12}")
    print("-" * 50)
    
    for cfg in configs:
        calc = StopCalculator(k_aggression=cfg["k"], min_trail_pct=cfg["min"], max_trail_pct=cfg["max"])
        index_vol = vol_tracker.get_daily_vol_pct()
        
        # V-dip price path
        price = entry
        stop_level = calc.compute_underlying_stop(entry, beta, index_vol, 30)
        high = entry
        triggered = False
        
        # Phase 1: Drop 5%
        for i in range(12):
            price = price * 0.996  # ~5% total drop
            if price <= stop_level:
                triggered = True
                break
        
        if not triggered:
            # Phase 2: Recover to +2%
            for i in range(66):
                price = price * 1.001
                if price > high:
                    high = price
                    new_stop = calc.compute_trail_from_high(high, beta, index_vol, 30)
                    stop_level = max(stop_level, new_stop)
        
        final_pnl = (price - entry) / entry * 100 if not triggered else (price - entry) / entry * 100
        survived = "YES ✓" if not triggered else "NO (stopped)"
        
        print(f"{cfg['name']:<20} {survived:<15} {final_pnl:>+7.2f}%")
    
    # ========== RECOMMENDATION ==========
    print_header("RECOMMENDATION")
    
    print("""
Based on your goal of NOT getting bumped out from sharp declines:

RECOMMENDED SETTINGS:
  k_aggression = 1.5    (50% wider than default)
  min_trail_pct = 0.08  (8% minimum trail)
  
This gives high-beta stocks ~12% room and SPY ~8% room, which should
survive most intraday V-dips while still providing meaningful protection.

TO APPLY THESE SETTINGS:
Edit: src/config_advanced.py

  K_AGGRESSION = 1.5
  MIN_TRAIL_PCT = 0.08

Or set environment variables:
  $env:K_AGGRESSION = "1.5"
  $env:MIN_TRAIL_PCT = "0.08"
""")
    
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_comparison()
