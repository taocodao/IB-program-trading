"""
Test Trailing Entry Manager
===========================

Mock test to verify entry trigger logic works correctly.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from entry_manager import (
    TrailingEntry, 
    find_atm_strike, 
    find_near_expiry,
    load_watchlist
)


def print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_trailing_entry_logic():
    """Test the trailing entry trigger logic."""
    print_header("TEST 1: Trailing Entry Logic")
    
    entry = TrailingEntry(symbol="AAPL", beta=1.25, trail_pct=0.02)
    
    # Simulate price falling then rising
    prices = [
        (240.0, "Initial"),
        (238.0, "Drops 0.8%"),
        (235.0, "Drops 2.1% - new low"),
        (233.0, "Drops 2.9% - new low"),
        (234.5, "Bounces slightly"),
        (236.0, "Bounces more"),
        (237.0, "Rising..."),
        (238.0, "Above 2% from low - TRIGGER!"),
    ]
    
    print(f"\nTrail percentage: {entry.trail_pct*100:.1f}%")
    print(f"\n{'Price':<10} {'Low':<10} {'Trail':<10} {'Triggered':<10} {'Note':<20}")
    print("-" * 65)
    
    for price, note in prices:
        triggered = entry.update_price(price)
        status = "YES!" if triggered else "no"
        print(f"${price:<9.2f} ${entry.lowest_price:<9.2f} ${entry.trail_level:<9.2f} {status:<10} {note}")
        
        if triggered:
            break
    
    print(f"\n✓ Entry triggered at ${entry.trigger_price:.2f}")
    print(f"  Low was ${entry.lowest_price:.2f}, trail was ${entry.trail_level:.2f}")


def test_strike_selection():
    """Test ATM strike selection."""
    print_header("TEST 2: Strike Selection")
    
    test_prices = [150.0, 245.0, 585.0, 1275.0]
    
    print(f"\n{'Underlying':<15} {'1% OTM Strike':<15} {'2% OTM Strike':<15}")
    print("-" * 50)
    
    for price in test_prices:
        strike_1pct = find_atm_strike(price, 0.01)
        strike_2pct = find_atm_strike(price, 0.02)
        print(f"${price:<14.2f} ${strike_1pct:<14.0f} ${strike_2pct:<14.0f}")


def test_expiry_selection():
    """Test expiry date selection."""
    print_header("TEST 3: Expiry Selection")
    
    today = datetime.now()
    print(f"\nToday: {today.strftime('%Y-%m-%d (%A)')}")
    
    expiry_days = [7, 14, 21, 30]
    
    for days in expiry_days:
        expiry = find_near_expiry(days)
        exp_date = datetime.strptime(expiry, "%Y%m%d")
        weekday = exp_date.strftime("%A")
        actual_days = (exp_date - today).days
        print(f"  Target {days:<2} days → {expiry} ({weekday}) = {actual_days} days")


def test_watchlist_loading():
    """Test watchlist loading."""
    print_header("TEST 4: Watchlist Loading")
    
    symbols = load_watchlist("watchlist.csv")
    
    print(f"\nLoaded {len(symbols)} symbols")
    print(f"First 10: {', '.join(symbols[:10])}")
    print(f"Last 5: {', '.join(symbols[-5:])}")


def test_complete_scenario():
    """Test complete entry scenario with multiple symbols."""
    print_header("TEST 5: Complete Scenario - Multiple Symbols")
    
    entries = [
        TrailingEntry(symbol="AAPL", beta=1.25, trail_pct=0.02),
        TrailingEntry(symbol="NVDA", beta=1.80, trail_pct=0.02),
        TrailingEntry(symbol="TSLA", beta=2.00, trail_pct=0.02),
        TrailingEntry(symbol="SPY", beta=1.00, trail_pct=0.02),
    ]
    
    # Initialize with entry prices
    init_prices = {"AAPL": 259.0, "NVDA": 185.0, "TSLA": 445.0, "SPY": 627.0}
    for entry in entries:
        entry.update_price(init_prices[entry.symbol])
    
    # Simulate market movement: all drop 3% then some bounce
    scenarios = [
        # (aapl_mult, nvda_mult, tsla_mult, spy_mult)
        (0.99, 0.98, 0.97, 0.995),   # Day 1: drops
        (0.98, 0.96, 0.94, 0.99),    # Day 2: more drops
        (0.97, 0.95, 0.92, 0.985),   # Day 3: bottom?
        (0.99, 0.97, 0.94, 0.99),    # Day 4: bounce
        (1.01, 1.00, 0.98, 1.01),    # Day 5: recovery
    ]
    
    print(f"\nSimulating 5 days of market movement...")
    print(f"\n{'Day':<6} {'AAPL':<10} {'NVDA':<10} {'TSLA':<10} {'SPY':<10}")
    print("-" * 50)
    
    triggered_entries = []
    
    for day, mults in enumerate(scenarios, 1):
        prices = {
            "AAPL": init_prices["AAPL"] * mults[0],
            "NVDA": init_prices["NVDA"] * mults[1],
            "TSLA": init_prices["TSLA"] * mults[2],
            "SPY": init_prices["SPY"] * mults[3],
        }
        
        row = f"{day:<6}"
        
        for entry in entries:
            price = prices[entry.symbol]
            triggered = entry.update_price(price)
            
            if triggered and entry not in triggered_entries:
                triggered_entries.append(entry)
                row += f"${price:<9.2f}*"
            else:
                row += f"${price:<9.2f} "
        
        print(row)
    
    print("\nTriggered entries:")
    for entry in triggered_entries:
        strike = find_atm_strike(entry.trigger_price, 0.01)
        expiry = find_near_expiry(14)
        print(f"  {entry.symbol}: BUY {expiry} ${strike:.0f} CALL")
    
    not_triggered = [e for e in entries if e not in triggered_entries]
    print(f"\nNot triggered: {', '.join(e.symbol for e in not_triggered)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRAILING ENTRY MANAGER - TEST SUITE")
    print("=" * 60)
    
    test_trailing_entry_logic()
    test_strike_selection()
    test_expiry_selection()
    test_watchlist_loading()
    test_complete_scenario()
    
    print("\n" + "=" * 60)
    print("✓ All tests complete")
    print("=" * 60 + "\n")
