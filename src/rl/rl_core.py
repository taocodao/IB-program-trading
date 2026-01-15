"""
RL Core Infrastructure
======================

Base classes, state encoding, reward calculation, and experience replay buffer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import random
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL agents."""
    
    # Network architecture
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    learning_rate: float = 0.001
    
    # Training
    gamma: float = 0.99  # Discount factor
    batch_size: int = 32
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 10000
    
    # PPO specific
    ppo_epsilon_clip: float = 0.2
    ppo_epochs: int = 10
    
    # Target network (DQN)
    target_update_freq: int = 100
    
    # Safety limits
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.08
    min_win_rate: float = 0.60
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/rl"


class StateEncoder:
    """
    Converts technical indicators into normalized state vectors for RL agents.
    
    State Space (17 inputs → compressed to ~12):
    - VCP: consolidation_bars, range_pct
    - SuperTrend: trend direction, volatility class
    - RSI: divergence strength, consensus length
    - Market: IV percentile, IV rank, time of day
    - Account: equity, daily P&L, win streak, drawdown
    - Order book: bid-ask spread, volume percentile
    """
    
    def __init__(self):
        # Normalization ranges from historical data
        self.feature_ranges = {
            'vcp_consolidation_bars': (5, 30),
            'vcp_range_pct': (0.05, 0.15),
            'rsi_divergence_strength': (0, 100),
            'rsi_consensus_length': (1, 6),
            'iv_percentile': (0, 100),
            'iv_rank': (0, 100),
            'bid_ask_spread': (0.01, 0.50),
            'volume_percentile': (0, 100),
            'win_streak': (-5, 5),
            'daily_pnl': (-0.05, 0.05),
            'max_drawdown': (-0.10, 0),
        }
        
        # Categorical encodings
        self.trend_encoding = {
            'UP': [1, 0, 0],
            'DOWN': [0, 1, 0],
            'SIDEWAYS': [0, 0, 1],
        }
        
        self.volatility_encoding = {
            'LOW': [1, 0, 0],
            'MEDIUM': [0, 1, 0],
            'HIGH': [0, 0, 1],
        }
        
        self.time_of_day_encoding = {
            'PRE_MARKET': [1, 0, 0, 0],
            'MORNING': [0, 1, 0, 0],
            'AFTERNOON': [0, 0, 1, 0],
            'LATE': [0, 0, 0, 1],
        }
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range."""
        if max_val == min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _get_time_of_day(self) -> str:
        """Get current market time of day."""
        hour = datetime.now().hour
        if hour < 9:
            return 'PRE_MARKET'
        elif hour < 12:
            return 'MORNING'
        elif hour < 15:
            return 'AFTERNOON'
        else:
            return 'LATE'
    
    def encode_state(self, indicators: Dict[str, Any], 
                     account_state: Optional[Dict] = None) -> np.ndarray:
        """
        Convert technical indicators + account state → normalized numpy array.
        
        Args:
            indicators: Dict with VCP, SuperTrend, RSI, IV data
            account_state: Dict with equity, daily_pnl, win_streak, etc.
        
        Returns:
            Normalized state vector (shape: ~20 features)
        """
        features = []
        
        # VCP features (2 continuous)
        vcp_bars = indicators.get('vcp_consolidation_bars', 15)
        vcp_range = indicators.get('vcp_range_pct', 0.10)
        features.append(self._normalize(vcp_bars, 5, 30))
        features.append(self._normalize(vcp_range, 0.05, 0.15))
        
        # SuperTrend features (3 categorical + 1 continuous)
        trend = indicators.get('supertrend_trend', 'SIDEWAYS')
        features.extend(self.trend_encoding.get(trend, [0, 0, 1]))
        
        volatility = indicators.get('supertrend_volatility_class', 'MEDIUM')
        features.extend(self.volatility_encoding.get(volatility, [0, 1, 0]))
        
        # RSI features (2 continuous)
        rsi_strength = indicators.get('rsi_divergence_strength', 50)
        rsi_consensus = indicators.get('rsi_consensus_length', 3)
        features.append(self._normalize(rsi_strength, 0, 100))
        features.append(self._normalize(rsi_consensus, 1, 6))
        
        # IV features (2 continuous)
        iv_pct = indicators.get('iv_percentile', 50)
        iv_rank = indicators.get('iv_rank', 50)
        features.append(self._normalize(iv_pct, 0, 100))
        features.append(self._normalize(iv_rank, 0, 100))
        
        # Time of day (4 categorical)
        time_of_day = self._get_time_of_day()
        features.extend(self.time_of_day_encoding.get(time_of_day, [0, 0, 1, 0]))
        
        # Account state (4 continuous) - optional
        if account_state:
            equity = account_state.get('equity', 100000)
            daily_pnl = account_state.get('daily_pnl', 0)
            win_streak = account_state.get('win_streak', 0)
            drawdown = account_state.get('max_drawdown', 0)
            
            features.append(self._normalize(np.log(equity), np.log(10000), np.log(1000000)))
            features.append(self._normalize(daily_pnl, -0.05, 0.05))
            features.append(self._normalize(win_streak, -5, 5))
            features.append(self._normalize(drawdown, -0.10, 0))
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        # Signal confidence from AI generator (1 continuous)
        signal_score = indicators.get('signal_score', 70) / 100
        features.append(signal_score)
        
        return np.array(features, dtype=np.float32)
    
    @property
    def state_size(self) -> int:
        """Return size of encoded state vector."""
        return 21  # 2+3+3+2+2+4+4+1


class RewardCalculator:
    """
    Calculates rewards for RL agents based on trade outcomes.
    
    Reward Design (critical for learning):
    - Entry: Better fill price vs baseline
    - Sizing: Risk-adjusted return optimization
    - Exit: Maximize gains, minimize losses
    """
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        
        # Weights for different reward components
        self.entry_weight = 0.2
        self.sizing_weight = 0.4
        self.exit_weight = 0.3
        self.ensemble_weight = 0.1
    
    def calculate_entry_reward(self, 
                               fill_price: float,
                               baseline_price: float,
                               action: str,
                               transaction_cost: float = 0.001) -> float:
        """
        Reward for entry timing decision.
        
        Args:
            fill_price: Actual fill price achieved
            baseline_price: Price at immediate entry (baseline)
            action: Entry action taken (IMMEDIATE, WAIT_5S, etc.)
            transaction_cost: Est. cost as fraction of position
        
        Returns:
            Reward value [-1, +1]
        """
        if baseline_price == 0:
            return 0
        
        # Price improvement (positive if got better price)
        price_improvement = (baseline_price - fill_price) / baseline_price
        
        # Reward for better fills (capped at ±1)
        reward = np.clip(price_improvement * 100 - transaction_cost, -1, 1)
        
        # Penalty for canceled signals that would have been profitable
        if action == 'CANCEL':
            reward = -0.2  # Small penalty, but could have been right
        
        return reward
    
    def calculate_sizing_reward(self,
                                pnl_pct: float,
                                position_size_pct: float,
                                max_drawdown: float) -> float:
        """
        Reward for position sizing decision.
        
        Rewards:
        - High returns with appropriate risk
        - Penalizes over-sizing that increases drawdown
        """
        # Risk-adjusted return (Sharpe-like)
        risk_adjusted = pnl_pct / max(position_size_pct, 0.01)
        
        # Drawdown penalty
        drawdown_penalty = min(0, max_drawdown + 0.05) * 10  # Penalty if DD > 5%
        
        # Combined reward
        reward = np.clip(risk_adjusted + drawdown_penalty, -1, 1)
        
        return reward
    
    def calculate_exit_reward(self,
                              entry_price: float,
                              exit_price: float,
                              max_favorable: float,
                              action: str) -> float:
        """
        Reward for exit strategy decision.
        
        Rewards:
        - Taking profits at good levels
        - Avoiding losses through proper stops
        - Penalizes holding too long or exiting too early
        """
        if entry_price == 0:
            return 0
        
        pnl_pct = (exit_price - entry_price) / entry_price
        max_pnl_pct = (max_favorable - entry_price) / entry_price if max_favorable > entry_price else 0
        
        # Strong winner (>3%)
        if pnl_pct > 0.03:
            return 2.0  # Great exit
        
        # Moderate winner (1-3%)
        elif pnl_pct > 0.01:
            # Did we leave money on the table?
            if max_pnl_pct > pnl_pct * 1.5:
                return 0.5  # Could have done better
            else:
                return 1.0  # Good exit
        
        # Small gain or breakeven
        elif pnl_pct > -0.01:
            if action in ['HOLD_5%_STOP', 'HOLD_8%_STOP']:
                return 0.3  # Held appropriately
            return 0.0  # Neutral
        
        # Loss (-1% to -3%)
        elif pnl_pct > -0.03:
            if action == 'HOLD_2%_STOP':
                return 0.5  # Stop worked as intended
            return -0.5  # Held too long
        
        # Large loss (>3%)
        else:
            return -1.0  # Bad exit
    
    def calculate_ensemble_reward(self, 
                                  agents_agreed: int,
                                  outcome_profitable: bool) -> float:
        """
        Reward for ensemble voting agreement.
        
        Higher reward when agents agree AND outcome is profitable.
        """
        if agents_agreed >= 2:  # 2/3 or 3/3 agreed
            return 0.5 if outcome_profitable else -0.2
        else:
            return -0.3 if outcome_profitable else 0.1  # Missed opportunity
    
    def calculate_total_reward(self,
                               entry_reward: float,
                               sizing_reward: float,
                               exit_reward: float,
                               ensemble_reward: float = 0) -> float:
        """Calculate weighted total reward."""
        return (
            self.entry_weight * entry_reward +
            self.sizing_weight * sizing_reward +
            self.exit_weight * exit_reward +
            self.ensemble_weight * ensemble_reward
        )


class ExperienceReplayBuffer:
    """
    Stores transitions (state, action, reward, next_state, done) for training.
    
    Used by DQN and can be adapted for A2C/PPO.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, 
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             info: Optional[Dict] = None):
        """Add transition to buffer."""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {},
            'timestamp': datetime.now().isoformat(),
        })
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch for training."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def sample_numpy(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch as numpy arrays for efficient training."""
        batch = self.sample(batch_size)
        
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size
    
    def save(self, filepath: str):
        """Save buffer to file."""
        data = [
            {
                'state': t['state'].tolist(),
                'action': int(t['action']),
                'reward': float(t['reward']),
                'next_state': t['next_state'].tolist(),
                'done': bool(t['done']),
                'info': t['info'],
                'timestamp': t['timestamp'],
            }
            for t in self.buffer
        ]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved {len(data)} transitions to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.buffer.clear()
        for t in data:
            self.buffer.append({
                'state': np.array(t['state'], dtype=np.float32),
                'action': t['action'],
                'reward': t['reward'],
                'next_state': np.array(t['next_state'], dtype=np.float32),
                'done': t['done'],
                'info': t.get('info', {}),
                'timestamp': t.get('timestamp', ''),
            })
        logger.info(f"Loaded {len(self.buffer)} transitions from {filepath}")


class TrajectoryBuffer:
    """
    Stores complete trajectories for PPO training.
    
    A trajectory is a sequence of (state, action, reward) from entry to exit.
    """
    
    def __init__(self):
        self.trajectories = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
        }
    
    def add_step(self,
                 state: np.ndarray,
                 action: int,
                 reward: float,
                 log_prob: float,
                 value: float):
        """Add step to current trajectory."""
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['log_probs'].append(log_prob)
        self.current_trajectory['values'].append(value)
    
    def end_trajectory(self, final_value: float = 0):
        """End current trajectory and compute returns."""
        if not self.current_trajectory['states']:
            return
        
        # Compute returns using GAE (Generalized Advantage Estimation)
        rewards = self.current_trajectory['rewards']
        values = self.current_trajectory['values'] + [final_value]
        
        returns = []
        gae = 0
        gamma = 0.99
        lam = 0.95  # GAE lambda
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + values[t])
        
        self.current_trajectory['returns'] = returns
        self.current_trajectory['advantages'] = [r - v for r, v in zip(returns, values[:-1])]
        
        self.trajectories.append(self.current_trajectory)
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
        }
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get all trajectories as batch for training."""
        if not self.trajectories:
            return None
        
        states = np.concatenate([np.array(t['states']) for t in self.trajectories])
        actions = np.concatenate([np.array(t['actions']) for t in self.trajectories])
        returns = np.concatenate([np.array(t['returns']) for t in self.trajectories])
        advantages = np.concatenate([np.array(t['advantages']) for t in self.trajectories])
        old_log_probs = np.concatenate([np.array(t['log_probs']) for t in self.trajectories])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': old_log_probs,
        }
    
    def clear(self):
        """Clear all trajectories."""
        self.trajectories = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
        }
    
    def __len__(self) -> int:
        return sum(len(t['states']) for t in self.trajectories)


# --- Utility Functions ---

def soft_update(target_weights: List, source_weights: List, tau: float = 0.005):
    """Soft update target network weights."""
    for target, source in zip(target_weights, source_weights):
        target.assign(tau * source + (1 - tau) * target)


def hard_update(target_weights: List, source_weights: List):
    """Hard update target network weights."""
    for target, source in zip(target_weights, source_weights):
        target.assign(source)
