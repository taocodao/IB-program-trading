"""
Mock Simulation: Volatility-Aware Stop System
==============================================

Demonstrates how the stop calculator works with different:
- Beta values (low, medium, high)
- VIX levels (calm, normal, volatile)
- DTE (long, medium, short)

No IB connection required - pure simulation.
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import OptionPosition, VolatilityTracker, PositionStatus, get_beta
from stop_calculator import StopCalculator, compute_smart_limit_price, compute_theoretical_price


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_position_summary(pos: OptionPosition, vol_tracker: VolatilityTracker, stop_calc: StopCalculator):
    """Print detailed position info with stop calculation."""
    dte = pos.days_to_expiry()
    index_vol = vol_tracker.get_daily_vol_pct()
    
    trail_pct = stop_calc.get_trail_percentage(
        pos.underlying_beta, index_vol, dte
    )
    
    stop_level = stop_calc.compute_underlying_stop(
        entry_price=pos.underlying_entry_price,
        beta=pos.underlying_beta,
        index_vol_pct=index_vol,
        days_to_expiry=dte
    )
    
    print(f"\n{pos.symbol} {pos.strike} {'CALL' if pos.right == 'C' else 'PUT'}")
    print(f"  Beta: {pos.underlying_beta:.2f}")
    print(f"  DTE: {dte} days")
    print(f"  Entry: ${pos.underlying_entry_price:.2f}")
    print(f"  Trail: {trail_pct*100:.1f}%")
    print(f"  Stop Level: ${stop_level:.2f}")
    print(f"  Risk Distance: ${pos.underlying_entry_price - stop_level:.2f}")


def simulate_price_movement(pos: OptionPosition, vol_tracker: VolatilityTracker, 
                           stop_calc: StopCalculator, price_moves: List[float]) -> bool:
    """
    Simulate price movements and trail stop accordingly.
    Returns True if stop was triggered.
    """
    print(f"\n--- Simulating price movement for {pos.symbol} ---")
    
    index_vol = vol_tracker.get_daily_vol_pct()
    dte = pos.days_to_expiry()
    
    # Initial stop
    pos.underlying_stop_level = stop_calc.compute_underlying_stop(
        pos.underlying_entry_price, pos.underlying_beta, index_vol, dte
    )
    pos.underlying_high = pos.underlying_entry_price
    
    print(f"Initial: Price=${pos.underlying_entry_price:.2f}, Stop=${pos.underlying_stop_level:.2f}")
    
    for i, move_pct in enumerate(price_moves):
        # Apply price move
        pos.underlying_price = pos.underlying_entry_price * (1 + move_pct)
        
        # Update high
        if pos.underlying_price > pos.underlying_high:
            pos.underlying_high = pos.underlying_price
            
            # Compute new trailing stop from high
            new_stop = stop_calc.compute_trail_from_high(
                pos.underlying_high, pos.underlying_beta, index_vol, dte
            )
            
            # Only move up (ratchet)
            if new_stop > pos.underlying_stop_level:
                old_stop = pos.underlying_stop_level
                pos.underlying_stop_level = new_stop
                print(f"  Step {i+1}: Price=${pos.underlying_price:.2f} ↑ NEW HIGH! "
                      f"Stop: ${old_stop:.2f} → ${new_stop:.2f}")
            else:
                print(f"  Step {i+1}: Price=${pos.underlying_price:.2f} ↑ (stop unchanged)")
        else:
            # Price dropped
            print(f"  Step {i+1}: Price=${pos.underlying_price:.2f} ↓ "
                  f"(stop stays at ${pos.underlying_stop_level:.2f})")
        
        # Check trigger
        if pos.underlying_price <= pos.underlying_stop_level:
            print(f"  *** STOP TRIGGERED! *** Price ${pos.underlying_price:.2f} <= Stop ${pos.underlying_stop_level:.2f}")
            pos.exit_triggered = True
            pos.status = PositionStatus.EXIT_TRIGGERED
            return True
    
    return False


def run_mock_simulation():
    """Run the full mock simulation."""
    
    print_header("VOLATILITY-AWARE STOP SYSTEM - MOCK SIMULATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create stop calculator with default settings
    stop_calc = StopCalculator(
        k_aggression=1.0,
        min_trail_pct=0.04,
        max_trail_pct=0.40
    )
    
    # ========== SCENARIO 1: Different Beta Values ==========
    print_header("SCENARIO 1: Different Beta Values (VIX = 18)")
    
    vol_tracker = VolatilityTracker()
    vol_tracker.update_vix(18.0)  # Normal VIX
    
    print(f"\nVIX Level: {vol_tracker.vix_level}")
    print(f"Daily Vol: {vol_tracker.get_daily_vol_pct()*100:.2f}%")
    
    # Create positions with different betas
    positions = [
        OptionPosition(
            conid=1, symbol="KO", expiry="20250221", strike=60.0, right="C",
            quantity=5, avg_entry_price=2.50, underlying_beta=0.55,
            underlying_entry_price=62.0, underlying_symbol="KO"
        ),
        OptionPosition(
            conid=2, symbol="SPY", expiry="20250221", strike=580.0, right="C",
            quantity=2, avg_entry_price=12.00, underlying_beta=1.00,
            underlying_entry_price=585.0, underlying_symbol="SPY"
        ),
        OptionPosition(
            conid=3, symbol="NVDA", expiry="20250221", strike=140.0, right="C",
            quantity=3, avg_entry_price=8.50, underlying_beta=1.80,
            underlying_entry_price=142.0, underlying_symbol="NVDA"
        ),
        OptionPosition(
            conid=4, symbol="TSLA", expiry="20250221", strike=400.0, right="C",
            quantity=1, avg_entry_price=15.00, underlying_beta=2.00,
            underlying_entry_price=410.0, underlying_symbol="TSLA"
        ),
    ]
    
    for pos in positions:
        print_position_summary(pos, vol_tracker, stop_calc)
    
    # ========== SCENARIO 2: Different VIX Levels ==========
    print_header("SCENARIO 2: Same Stock (SPY), Different VIX Levels")
    
    spy = OptionPosition(
        conid=10, symbol="SPY", expiry="20250221", strike=580.0, right="C",
        quantity=2, avg_entry_price=12.00, underlying_beta=1.00,
        underlying_entry_price=585.0, underlying_symbol="SPY"
    )
    
    vix_levels = [12, 18, 25, 35, 50]
    
    for vix in vix_levels:
        vol_tracker.update_vix(vix)
        index_vol = vol_tracker.get_daily_vol_pct()
        
        trail_pct = stop_calc.get_trail_percentage(spy.underlying_beta, index_vol, spy.days_to_expiry())
        stop = stop_calc.compute_underlying_stop(
            spy.underlying_entry_price, spy.underlying_beta, index_vol, spy.days_to_expiry()
        )
        
        print(f"\nVIX {vix}: Trail={trail_pct*100:.1f}%, Stop=${stop:.2f}, "
              f"Distance=${spy.underlying_entry_price - stop:.2f}")
    
    # ========== SCENARIO 3: Different DTE ==========
    print_header("SCENARIO 3: Same Stock (AAPL), Different DTE")
    
    vol_tracker.update_vix(20.0)
    
    dte_values = [60, 30, 14, 7, 3, 1]
    
    for dte in dte_values:
        # Create expiry date
        from datetime import timedelta
        expiry = (datetime.now() + timedelta(days=dte)).strftime("%Y%m%d")
        
        aapl = OptionPosition(
            conid=20, symbol="AAPL", expiry=expiry, strike=180.0, right="C",
            quantity=5, avg_entry_price=5.00, underlying_beta=1.25,
            underlying_entry_price=182.0, underlying_symbol="AAPL"
        )
        
        index_vol = vol_tracker.get_daily_vol_pct()
        trail_pct = stop_calc.get_trail_percentage(aapl.underlying_beta, index_vol, dte)
        stop = stop_calc.compute_underlying_stop(182.0, 1.25, index_vol, dte)
        
        mult = "1.0x" if dte > 30 else "1.5x" if dte >= 7 else "2.0x"
        print(f"\nDTE {dte:2d}: Trail={trail_pct*100:.1f}% ({mult}), "
              f"Stop=${stop:.2f}, Distance=${182.0 - stop:.2f}")
    
    # ========== SCENARIO 4: Trailing Stop Simulation ==========
    print_header("SCENARIO 4: Trailing Stop Simulation (TSLA)")
    
    vol_tracker.update_vix(22.0)
    
    tsla = OptionPosition(
        conid=30, symbol="TSLA", expiry="20250221", strike=400.0, right="C",
        quantity=2, avg_entry_price=18.00, underlying_beta=2.00,
        underlying_entry_price=410.0, underlying_symbol="TSLA"
    )
    
    # Simulate: price goes up, then drops, then up again
    price_moves = [
        0.02,   # +2%: 418.20
        0.04,   # +4%: 426.40
        0.03,   # +3%: 422.30 (down from high)
        0.05,   # +5%: 430.50 (new high)
        0.02,   # +2%: 418.20 (down)
        0.00,   # 0%: 410.00 (back to entry)
        -0.03,  # -3%: 397.70 (below stop?)
    ]
    
    triggered = simulate_price_movement(tsla, vol_tracker, stop_calc, price_moves)
    
    if triggered:
        print(f"\nResult: Stop was triggered at ${tsla.underlying_price:.2f}")
    else:
        print(f"\nResult: Position still open, stop at ${tsla.underlying_stop_level:.2f}")
    
    # ========== SCENARIO 5: Smart Limit Pricing ==========
    print_header("SCENARIO 5: Smart Limit Price Calculation")
    
    print("\nWhen exit triggered, compute smart limit between bid and theoretical:")
    
    scenarios = [
        {"bid": 5.00, "ask": 5.50, "theo": 5.20, "desc": "Normal spread"},
        {"bid": 3.00, "ask": 4.00, "theo": 3.50, "desc": "Wide spread"},
        {"bid": 8.00, "ask": 8.20, "theo": 8.10, "desc": "Tight spread"},
        {"bid": 2.00, "ask": 2.50, "theo": 1.80, "desc": "Theo below bid"},
    ]
    
    for s in scenarios:
        limit = compute_smart_limit_price(
            s["bid"], s["ask"], s["theo"],
            spread_participation=0.5
        )
        print(f"\n{s['desc']}:")
        print(f"  Bid=${s['bid']:.2f}, Ask=${s['ask']:.2f}, Theo=${s['theo']:.2f}")
        print(f"  → Smart Limit: ${limit:.2f}")
    
    # ========== SUMMARY ==========
    print_header("SUMMARY: Key Insights")
    
    print("""
1. BETA EFFECT:
   - Low beta (0.55): Tighter stops (defensive stocks move less)
   - High beta (2.00): Wider stops (volatile stocks need room)

2. VIX EFFECT:
   - Low VIX (12): Tighter stops (calm market)
   - High VIX (50): Wider stops (volatile market)

3. DTE EFFECT:
   - Long DTE (60+): Normal trail (1.0x)
   - Medium DTE (7-30): 1.5x wider (gamma increasing)
   - Short DTE (<7): 2.0x wider (high gamma, fast moves)

4. TRAILING:
   - Stop moves UP when price rises (locks in gains)
   - Stop STAYS when price drops (protects from whipsaw)
   - Triggered when underlying hits stop level

5. SMART EXECUTION:
   - Limit set between bid and theoretical
   - Never below bid (worst case)
   - Never above theoretical (overly optimistic)
""")
    
    print("\n" + "=" * 70)
    print("MOCK SIMULATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_mock_simulation()
