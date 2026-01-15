import json
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple
from datetime import datetime

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
TRADES_FILE = os.path.join(DATA_DIR, 'historical_trades.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'training_data.json')

def load_trades(file_path: str) -> List[Dict]:
    """Load trades from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_reward(trade: Dict) -> float:
    """
    Calculate reward based on trade result.
    This is a simplified version; use the full RewardCalculator in rl_core for training.
    """
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    signal_type = trade['signal_type']
    
    if 'CALL' in signal_type:
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
        
    return pnl_pct * 100  # Return percentage as reward

def prepare_data():
    """Load trades and format into episodes."""
    print(f"Loading trades from {TRADES_FILE}...")
    trades = load_trades(TRADES_FILE)
    print(f"Loaded {len(trades)} trades.")
    
    episodes = []
    
    for trade in trades:
        # Extract state features (matching StateEncoder expected input)
        indicators = trade['indicators']
        account = trade['account_state']
        
        # Construct state dictionary (flat structure often cleaner for preprocessing)
        state = {
            'rsi': indicators.get('rsi', 50),
            'adx': indicators.get('adx', 0),
            'supertrend_trend': 1 if indicators.get('supertrend_trend') == 'UP' else -1 if indicators.get('supertrend_trend') == 'DOWN' else 0,
            'iv_percentile': indicators.get('iv_percentile', 0),
            'support_distance_pct': indicators.get('support_distance_pct', 0),
            'resistance_distance_pct': indicators.get('resistance_distance_pct', 0),
            'momentum': indicators.get('momentum', 0),
            'iv_change': indicators.get('iv_change_since_entry', 0),
            'daily_pnl': account.get('daily_pnl', 0),
            'win_streak': account.get('win_streak', 0),
            'max_drawdown': account.get('max_drawdown', 0),
            'has_positions': 1 if account.get('has_open_positions') else 0
        }
        
        # Calculate ground truth reward
        reward = calculate_reward(trade)
        
        episode_step = {
            'state': state,
            'action_taken': trade['signal_type'], # Simplified, normally we map this specific actions
            'reward': reward,
            'position_size_pct': trade['position_size_pct']
        }
        episodes.append(episode_step)
        
    print(f"Processed {len(episodes)} steps.")
    
    # Save processed data
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(episodes, f, indent=2)
    print(f"Saved training data to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
