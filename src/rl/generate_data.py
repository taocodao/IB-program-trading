"""
Generate Synthetic Training Data
================================

Generates synthetic historical trade data for RL training.
Simulates market conditions, signals, and outcomes.
"""

import json
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_data(num_trades: int = 1000, output_file: str = "data/historical_trades.json"):
    """Generate synthetic trades."""
    
    trades = []
    start_time = datetime.now() - timedelta(days=365)
    
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    signal_types = ['BUY_CALL', 'BUY_PUT']
    
    print(f"Generating {num_trades} synthetic trades...")
    
    for i in range(num_trades):
        # Time
        trade_time = start_time + timedelta(hours=i * 2 + random.randint(0, 4))
        
        # Signal context
        symbol = random.choice(symbols)
        signal_type = random.choice(signal_types)
        signal_score = random.randint(60, 95)
        
        # Market context (Synthetic indicators)
        indicators = {
            'rsi': random.uniform(30, 70),
            'adx': random.uniform(15, 45),
            'supertrend_trend': random.choice(['UP', 'DOWN', 'SIDEWAYS']),
            'iv_percentile': random.uniform(10, 90),
            'support_distance_pct': random.uniform(0.0, 0.10),
            'resistance_distance_pct': random.uniform(0.0, 0.10),
            'momentum': random.uniform(-1, 1),
            'iv_change_since_entry': random.uniform(-0.1, 0.1),
        }
        
        # Correlate outcome with signal score
        # Higher score = higher probability of success
        win_prob = signal_score / 120  # e.g., 60 -> 0.5, 90 -> 0.75
        is_win = random.random() < win_prob
        
        entry_price = random.uniform(100, 500)
        
        if is_win:
            # Winner: +5% to +50%
            pnl_pct = random.uniform(0.05, 0.50)
        else:
            # Loser: -5% to -100%
            pnl_pct = random.uniform(-0.05, -0.80)
            
        exit_price = entry_price * (1 + pnl_pct)
        
        # Position sizing used (randomized to simulate diversity)
        position_size_pct = random.choice([0.5, 1.0, 2.0, 3.0, 5.0])
        
        # Duration
        hold_time_minutes = random.randint(15, 240)
        
        trade = {
            "symbol": symbol,
            "signal_type": signal_type,
            "signal_score": signal_score,
            "indicators": indicators,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "entry_time": trade_time.isoformat(),
            "exit_time": (trade_time + timedelta(minutes=hold_time_minutes)).isoformat(),
            "position_size_pct": position_size_pct,
            "account_state": {
                "daily_pnl": random.uniform(-0.02, 0.02),
                "win_streak": random.randint(0, 5),
                "max_drawdown": random.uniform(-0.10, 0.0),
                "has_open_positions": random.choice([True, False]),
            }
        }
        
        trades.append(trade)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(trades, f, indent=2)
        
    print(f"Saved to {output_file}")
    return len(trades)

if __name__ == "__main__":
    generate_synthetic_data()
