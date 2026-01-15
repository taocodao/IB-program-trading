"""
PPO Exit Strategy Agent
=======================

Proximal Policy Optimization (PPO) agent for exit strategy optimization.

Actions:
- TAKE_PROFIT: Exit now with current gains
- HOLD_TIGHT_2: Hold with 2% trailing stop (tight)
- HOLD_NORMAL_5: Hold with 5% trailing stop (normal)
- HOLD_LOOSE_8: Hold with 8% trailing stop (loose)

The agent learns:
1. When to take profits vs hold for more
2. Optimal stop loss level based on position context
3. Adapt to volatility and momentum changes
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import IntEnum
import logging
import os

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - using numpy-only implementation")


class ExitAction(IntEnum):
    """Exit strategy actions."""
    TAKE_PROFIT = 0    # Exit now
    HOLD_TIGHT_2 = 1   # Hold with 2% stop
    HOLD_NORMAL_5 = 2  # Hold with 5% stop
    HOLD_LOOSE_8 = 3   # Hold with 8% stop


# Stop loss percentages
STOP_LEVELS = {
    ExitAction.TAKE_PROFIT: 0.0,    # N/A - exit now
    ExitAction.HOLD_TIGHT_2: 0.02,  # 2% stop
    ExitAction.HOLD_NORMAL_5: 0.05, # 5% stop
    ExitAction.HOLD_LOOSE_8: 0.08,  # 8% stop
}


@dataclass
class ExitAgentConfig:
    """Configuration for Exit Agent."""
    state_size: int = 25  # Position state has extra features
    action_size: int = 4
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    learning_rate: float = 0.0003  # Lower for PPO
    gamma: float = 0.99
    
    # PPO-specific
    epsilon_clip: float = 0.2   # Clipping parameter
    ppo_epochs: int = 10        # Epochs per update
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95    # GAE parameter
    
    # Batch training
    batch_size: int = 64
    max_trajectory_length: int = 100


if TF_AVAILABLE:
    class PPONetwork(Model):
        """
        Actor-Critic network for PPO.
        
        Outputs:
        - Policy (action probabilities)
        - Value (state value estimate)
        """
        
        def __init__(self, config: ExitAgentConfig):
            super(PPONetwork, self).__init__()
            
            # Shared layers
            self.dense1 = layers.Dense(config.hidden_size_1, activation='relu',
                                       kernel_initializer='he_normal')
            self.dense2 = layers.Dense(config.hidden_size_2, activation='relu',
                                       kernel_initializer='he_normal')
            
            # Policy head
            self.policy_dense = layers.Dense(32, activation='relu')
            self.policy_output = layers.Dense(config.action_size, activation='softmax')
            
            # Value head
            self.value_dense = layers.Dense(32, activation='relu')
            self.value_output = layers.Dense(1)
        
        def call(self, inputs, training=False):
            # Shared features
            x = self.dense1(inputs)
            x = self.dense2(x)
            
            # Policy
            pi = self.policy_dense(x)
            action_probs = self.policy_output(pi)
            
            # Value
            v = self.value_dense(x)
            value = self.value_output(v)
            
            return action_probs, value
        
        def get_action(self, state: np.ndarray, 
                       deterministic: bool = False) -> Tuple[int, float, float]:
            """Select action and return log_prob and value."""
            state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
            action_probs, value = self(state, training=False)
            
            probs = action_probs[0]
            val = value[0, 0]
            
            if deterministic:
                action = tf.argmax(probs).numpy()
            else:
                dist = tf.random.categorical(tf.math.log(probs[np.newaxis]), 1)
                action = dist[0, 0].numpy()
            
            log_prob = tf.math.log(probs[action] + 1e-8).numpy()
            
            return int(action), float(log_prob), float(val)


class ExitAgent:
    """
    PPO agent for exit strategy optimization.
    
    Features:
    - Clipped objective for stable training
    - GAE (Generalized Advantage Estimation)
    - Multiple epochs per update batch
    """
    
    def __init__(self, config: Optional[ExitAgentConfig] = None):
        self.config = config or ExitAgentConfig()
        
        if TF_AVAILABLE:
            self.network = PPONetwork(self.config)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            )
            
            # Build network
            dummy_input = np.zeros((1, self.config.state_size), dtype=np.float32)
            self.network(dummy_input)
        else:
            self.network = None
        
        # Trajectory storage
        self.trajectories = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
        }
        
        # Training counter
        self.train_step_count = 0
    
    def select_action(self, state: np.ndarray, 
                      deterministic: bool = False
                     ) -> Tuple[ExitAction, float, float]:
        """
        Select exit action based on position state.
        
        Args:
            state: Position state vector (includes P&L, time, etc.)
            deterministic: If True, select most probable action
        
        Returns:
            action: ExitAction enum
            log_prob: Log probability (for training)
            value: State value estimate
        """
        if self.network is None:
            return ExitAction.HOLD_NORMAL_5, 0.0, 0.0
        
        action_idx, log_prob, value = self.network.get_action(state, deterministic)
        return ExitAction(action_idx), log_prob, value
    
    def get_stop_level(self, action: ExitAction, entry_price: float) -> float:
        """Get stop price for given action."""
        if action == ExitAction.TAKE_PROFIT:
            return entry_price  # Will exit at market, not stop
        
        stop_pct = STOP_LEVELS[action]
        return entry_price * (1 - stop_pct)
    
    def should_exit_now(self, action: ExitAction) -> bool:
        """Check if action is to exit immediately."""
        return action == ExitAction.TAKE_PROFIT
    
    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         log_prob: float,
                         value: float,
                         done: bool):
        """Store transition in current trajectory."""
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['log_probs'].append(log_prob)
        self.current_trajectory['values'].append(value)
        self.current_trajectory['dones'].append(done)
    
    def end_trajectory(self, final_value: float = 0):
        """End current trajectory and compute returns."""
        if not self.current_trajectory['states']:
            return
        
        # Compute returns and advantages using GAE
        rewards = self.current_trajectory['rewards']
        values = self.current_trajectory['values'] + [final_value]
        dones = self.current_trajectory['dones']
        
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        self.current_trajectory['returns'] = returns
        self.current_trajectory['advantages'] = advantages
        
        self.trajectories.append(self.current_trajectory)
        
        # Reset current trajectory
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
        }
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Train on collected trajectories using PPO.
        
        Returns dict with loss metrics or None if no data.
        """
        if not self.trajectories or not TF_AVAILABLE:
            return None
        
        # Flatten trajectories into batch
        states = np.concatenate([np.array(t['states']) for t in self.trajectories])
        actions = np.concatenate([np.array(t['actions']) for t in self.trajectories])
        returns = np.concatenate([np.array(t['returns']) for t in self.trajectories])
        advantages = np.concatenate([np.array(t['advantages']) for t in self.trajectories])
        old_log_probs = np.concatenate([np.array(t['log_probs']) for t in self.trajectories])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        
        # Multiple PPO epochs
        total_losses = []
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]
                
                batch_states = tf.gather(states, batch_idx)
                batch_actions = tf.gather(actions, batch_idx)
                batch_returns = tf.gather(returns, batch_idx)
                batch_advantages = tf.gather(advantages, batch_idx)
                batch_old_log_probs = tf.gather(old_log_probs, batch_idx)
                
                loss = self._train_step(
                    batch_states, batch_actions, batch_returns,
                    batch_advantages, batch_old_log_probs
                )
                total_losses.append(loss)
        
        # Clear trajectories
        self.trajectories = []
        self.train_step_count += 1
        
        return {
            'total_loss': float(np.mean(total_losses)),
            'num_samples': len(states),
            'num_epochs': self.config.ppo_epochs,
        }
    
    def _train_step(self,
                    states,  # tf.Tensor when TF available
                    actions,  # tf.Tensor when TF available
                    returns,  # tf.Tensor when TF available
                    advantages,  # tf.Tensor when TF available
                    old_log_probs) -> float:  # tf.Tensor when TF available
        """Single PPO training step."""
        
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs, values = self.network(states, training=True)
            
            # Get log probs for taken actions
            batch_size = tf.shape(states)[0]
            indices = tf.stack([tf.range(batch_size), actions], axis=1)
            new_log_probs = tf.math.log(tf.gather_nd(action_probs, indices) + 1e-8)
            
            # Importance sampling ratio
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            clipped_ratio = tf.clip_by_value(
                ratio,
                1 - self.config.epsilon_clip,
                1 + self.config.epsilon_clip
            )
            
            # Policy loss (take minimum of clipped and unclipped)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            
            # Value loss
            value_loss = self.config.value_coef * tf.reduce_mean(
                tf.square(returns - values[:, 0])
            )
            
            # Entropy bonus (encourages exploration)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            )
            entropy_loss = -self.config.entropy_coef * entropy
            
            # Total loss
            total_loss = policy_loss + value_loss + entropy_loss
        
        # Gradient update
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables)
        )
        
        return float(total_loss)
    
    def save(self, filepath: str):
        """Save model weights."""
        if self.network is None:
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.network.save_weights(filepath)
        logger.info(f"ExitAgent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        if self.network is None:
            return
        
        self.network.load_weights(filepath)
        logger.info(f"ExitAgent loaded from {filepath}")
    
    def get_action_name(self, action: ExitAction) -> str:
        """Get human-readable action name."""
        names = {
            ExitAction.TAKE_PROFIT: "EXIT NOW",
            ExitAction.HOLD_TIGHT_2: "HOLD (2% stop)",
            ExitAction.HOLD_NORMAL_5: "HOLD (5% stop)",
            ExitAction.HOLD_LOOSE_8: "HOLD (8% stop)",
        }
        return names[action]
    
    def get_policy(self, state: np.ndarray) -> Dict[str, float]:
        """Get action probabilities for state."""
        if self.network is None:
            return {a.name: 0.25 for a in ExitAction}
        
        state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        action_probs, _ = self.network(state, training=False)
        probs = action_probs[0].numpy()
        
        return {ExitAction(i).name: float(probs[i]) for i in range(len(probs))}


# --- Position state encoder ---

def encode_position_state(entry_price: float,
                          current_price: float,
                          current_pnl_pct: float,
                          time_in_trade_seconds: float,
                          indicators: Dict,
                          account_state: Optional[Dict] = None) -> np.ndarray:
    """
    Encode position state for exit agent.
    
    Includes:
    - Current P&L
    - Time in trade
    - IV change since entry
    - Technical indicator updates
    - Support/resistance distance
    """
    features = []
    
    # P&L features (3)
    features.append(np.clip(current_pnl_pct, -0.20, 0.20))  # -20% to +20%
    features.append(1 if current_pnl_pct > 0 else 0)  # Profitable flag
    features.append(np.clip(abs(current_pnl_pct) / 0.10, 0, 1))  # Magnitude normalized
    
    # Time features (2)
    time_hours = time_in_trade_seconds / 3600
    features.append(np.clip(time_hours / 4, 0, 1))  # Normalize to 4 hours
    features.append(1 if time_hours > 1 else 0)  # Long hold flag
    
    # Technical updates (from indicators dict)
    # RSI current
    rsi = indicators.get('rsi', 50)
    features.append((rsi - 50) / 50)  # -1 to +1
    
    # Momentum
    momentum = indicators.get('momentum', 0)
    features.append(np.clip(momentum / 2, -1, 1))  # Normalized
    
    # SuperTrend direction  
    trend = indicators.get('supertrend_trend', 'SIDEWAYS')
    features.extend([
        1 if trend == 'UP' else 0,
        1 if trend == 'DOWN' else 0,
        1 if trend == 'SIDEWAYS' else 0,
    ])
    
    # IV change since entry
    iv_change = indicators.get('iv_change_since_entry', 0)
    features.append(np.clip(iv_change, -0.20, 0.20))
    
    # Distance to support/resistance
    support_dist = indicators.get('support_distance_pct', 0.05)
    resist_dist = indicators.get('resistance_distance_pct', 0.05)
    features.append(np.clip(support_dist, 0, 0.10) / 0.10)
    features.append(np.clip(resist_dist, 0, 0.10) / 0.10)
    
    # Account state (4)
    if account_state:
        features.append(np.clip(account_state.get('daily_pnl', 0), -0.05, 0.05) / 0.05)
        features.append(np.clip(account_state.get('win_streak', 0), -5, 5) / 5)
        features.append(np.clip(abs(account_state.get('max_drawdown', 0)), 0, 0.10) / 0.10)
        features.append(1 if account_state.get('has_open_positions', False) else 0)
    else:
        features.extend([0, 0, 0, 0])
    
    # Pad to state_size if needed
    while len(features) < 25:
        features.append(0)
    
    return np.array(features[:25], dtype=np.float32)


# --- Rule-based fallback ---

def select_exit_rule_based(current_pnl_pct: float,
                           time_in_trade_seconds: float,
                           indicators: Dict) -> ExitAction:
    """
    Rule-based exit strategy fallback.
    """
    time_hours = time_in_trade_seconds / 3600
    
    # Strong profit (>5%) - take it
    if current_pnl_pct >= 0.05:
        return ExitAction.TAKE_PROFIT
    
    # Good profit (3-5%)
    if current_pnl_pct >= 0.03:
        # Hold with tight stop if still in uptrend
        trend = indicators.get('supertrend_trend', 'SIDEWAYS')
        if trend == 'UP':
            return ExitAction.HOLD_TIGHT_2
        return ExitAction.TAKE_PROFIT
    
    # Moderate profit (1-3%)
    if current_pnl_pct >= 0.01:
        return ExitAction.HOLD_TIGHT_2
    
    # Small profit or breakeven
    if current_pnl_pct >= -0.01:
        if time_hours > 2:  # Held for 2+ hours with no gain
            return ExitAction.HOLD_TIGHT_2
        return ExitAction.HOLD_NORMAL_5
    
    # Small loss (-1% to -3%)
    if current_pnl_pct >= -0.03:
        return ExitAction.HOLD_NORMAL_5
    
    # Larger loss (>-3%)
    return ExitAction.HOLD_LOOSE_8


def create_exit_agent(config: Optional[ExitAgentConfig] = None) -> ExitAgent:
    """Factory function to create ExitAgent."""
    return ExitAgent(config)
