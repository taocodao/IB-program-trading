"""
DQN Position Sizing Agent
=========================

Deep Q-Network (DQN) agent for optimal position sizing.

Actions (6 discrete position sizes):
- 0.5%: Very conservative (low confidence, high IV)
- 1.0%: Conservative
- 2.0%: Moderate
- 3.0%: Standard
- 5.0%: Aggressive (high confidence, low IV)
- 8.0%: Very aggressive (strong signals only)

The agent learns:
1. Optimal position size based on signal strength
2. Risk adjustment for market conditions
3. Account state awareness (drawdown, win streak)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import IntEnum
from collections import deque
import random
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


class SizingAction(IntEnum):
    """Position sizing actions."""
    SIZE_0_5 = 0   # 0.5% of account
    SIZE_1_0 = 1   # 1.0%
    SIZE_2_0 = 2   # 2.0%
    SIZE_3_0 = 3   # 3.0%
    SIZE_5_0 = 4   # 5.0%
    SIZE_8_0 = 5   # 8.0%


# Map action to actual position size percentage
POSITION_SIZES = {
    SizingAction.SIZE_0_5: 0.005,
    SizingAction.SIZE_1_0: 0.010,
    SizingAction.SIZE_2_0: 0.020,
    SizingAction.SIZE_3_0: 0.030,
    SizingAction.SIZE_5_0: 0.050,
    SizingAction.SIZE_8_0: 0.080,
}


@dataclass
class SizingAgentConfig:
    """Configuration for Position Sizing Agent."""
    state_size: int = 21
    action_size: int = 6
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    hidden_size_3: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    
    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 32
    
    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Target network
    target_update_freq: int = 100


if TF_AVAILABLE:
    class QNetwork(Model):
        """
        Q-Network for position sizing.
        
        Outputs Q-values for each position size action.
        Q(s, a) = expected cumulative reward for taking action a in state s
        """
        
        def __init__(self, config: SizingAgentConfig):
            super(QNetwork, self).__init__()
            
            self.dense1 = layers.Dense(config.hidden_size_1, activation='relu',
                                       kernel_initializer='he_normal')
            self.dense2 = layers.Dense(config.hidden_size_2, activation='relu',
                                       kernel_initializer='he_normal')
            self.dense3 = layers.Dense(config.hidden_size_3, activation='relu',
                                       kernel_initializer='he_normal')
            self.dropout = layers.Dropout(0.2)
            self.output_layer = layers.Dense(config.action_size)  # Q-values for each action
        
        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            if training:
                x = self.dropout(x, training=training)
            q_values = self.output_layer(x)
            return q_values


class SizingAgent:
    """
    DQN agent for position sizing optimization.
    
    Features:
    - Experience replay for stable learning
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration
    """
    
    def __init__(self, config: Optional[SizingAgentConfig] = None):
        self.config = config or SizingAgentConfig()
        
        # Exploration rate
        self.epsilon = self.config.epsilon_start
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.config.buffer_size)
        
        if TF_AVAILABLE:
            # Q-networks (main and target)
            self.q_network = QNetwork(self.config)
            self.target_network = QNetwork(self.config)
            
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            )
            
            # Build networks
            dummy_input = np.zeros((1, self.config.state_size), dtype=np.float32)
            self.q_network(dummy_input)
            self.target_network(dummy_input)
            
            # Initialize target network with same weights
            self.update_target_network()
        else:
            self.q_network = None
            self.target_network = None
        
        # Training counter
        self.train_step_count = 0
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        if self.q_network is None:
            return
        
        for target_var, source_var in zip(
            self.target_network.trainable_variables,
            self.q_network.trainable_variables
        ):
            target_var.assign(source_var)
    
    def select_action(self, state: np.ndarray, 
                      training: bool = True) -> Tuple[SizingAction, np.ndarray]:
        """
        Select position size using epsilon-greedy policy.
        
        Args:
            state: State vector
            training: If True, use epsilon-greedy; else pure greedy
        
        Returns:
            action: SizingAction enum
            q_values: Q-values for all actions (for analysis)
        """
        if self.q_network is None:
            # Fallback: moderate sizing
            return SizingAction.SIZE_2_0, np.zeros(6)
        
        state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        q_values = self.q_network(state, training=False)[0].numpy()
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.config.action_size)
        else:
            action_idx = np.argmax(q_values)
        
        return SizingAction(action_idx), q_values
    
    def get_position_size(self, action: SizingAction) -> float:
        """Get position size as fraction (0.005 = 0.5%)."""
        return POSITION_SIZES[action]
    
    def get_position_size_pct(self, action: SizingAction) -> float:
        """Get position size as percentage (0.5, 1.0, etc.)."""
        return POSITION_SIZES[action] * 100
    
    def remember(self, 
                 state: np.ndarray,
                 action: int,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool):
        """Store transition in experience replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """
        Train on batch from experience replay.
        
        Returns loss value or None if buffer not ready.
        """
        if len(self.memory) < self.config.batch_size:
            return None
        
        if not TF_AVAILABLE or self.q_network is None:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        with tf.GradientTape() as tape:
            # Current Q-values
            q_values = self.q_network(states, training=True)
            
            # Target Q-values (from target network)
            next_q_values = self.target_network(next_states, training=False)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            
            # Bellman target: r + γ * max Q(s', a')
            targets = rewards + self.config.gamma * max_next_q * (1 - dones)
            
            # Get Q-values for taken actions
            batch_indices = tf.range(self.config.batch_size)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_for_actions = tf.gather_nd(q_values, action_indices)
            
            # Loss: MSE between predicted and target Q-values
            loss = tf.reduce_mean(tf.square(targets - q_for_actions))
        
        # Gradient update
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        
        self.train_step_count += 1
        
        # Update target network periodically
        if self.train_step_count % self.config.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        
        return float(loss)
    
    def train_step(self,
                   state: np.ndarray,
                   action: int,
                   reward: float,
                   next_state: np.ndarray,
                   done: bool) -> Optional[float]:
        """
        Add experience and train if buffer ready.
        
        Convenience method that combines remember() and replay().
        """
        self.remember(state, action, reward, next_state, done)
        return self.replay()
    
    def save(self, filepath: str):
        """Save model weights."""
        if self.q_network is None:
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save_weights(filepath)
        
        # Also save epsilon
        np.save(filepath + '.epsilon.npy', self.epsilon)
        
        logger.info(f"SizingAgent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        if self.q_network is None:
            return
        
        self.q_network.load_weights(filepath)
        self.update_target_network()  # Sync target network
        
        # Load epsilon if exists
        epsilon_path = filepath + '.epsilon.npy'
        if os.path.exists(epsilon_path):
            self.epsilon = float(np.load(epsilon_path))
        
        logger.info(f"SizingAgent loaded from {filepath}")
    
    def get_action_name(self, action: SizingAction) -> str:
        """Get human-readable action name."""
        return f"{self.get_position_size_pct(action):.1f}%"
    
    def get_q_values(self, state: np.ndarray) -> Dict[str, float]:
        """Get Q-values for all actions (for analysis)."""
        if self.q_network is None:
            return {f"{p*100:.1f}%": 0.0 for p in POSITION_SIZES.values()}
        
        state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        q_values = self.q_network(state, training=False)[0].numpy()
        
        return {
            f"{POSITION_SIZES[SizingAction(i)]*100:.1f}%": float(q_values[i])
            for i in range(len(q_values))
        }


# --- Rule-based fallback ---

def select_size_rule_based(indicators: Dict,
                           signal_score: float,
                           account_state: Optional[Dict] = None) -> SizingAction:
    """
    Rule-based position sizing fallback.
    
    Logic:
    - High confidence + low IV → larger position
    - Low confidence + high IV → smaller position
    - Consider account drawdown
    """
    # Get context
    iv_pct = indicators.get('iv_percentile', 50)
    drawdown = (account_state or {}).get('max_drawdown', 0)
    
    # Safety check: reduce size if in drawdown
    if drawdown < -0.05:  # >5% drawdown
        if signal_score >= 80:
            return SizingAction.SIZE_2_0
        return SizingAction.SIZE_1_0
    
    # High confidence signals
    if signal_score >= 85:
        if iv_pct < 30:  # Low IV = cheap options
            return SizingAction.SIZE_8_0
        elif iv_pct < 50:
            return SizingAction.SIZE_5_0
        else:
            return SizingAction.SIZE_3_0
    
    # Medium-high confidence
    elif signal_score >= 75:
        if iv_pct < 40:
            return SizingAction.SIZE_3_0
        else:
            return SizingAction.SIZE_2_0
    
    # Medium confidence
    elif signal_score >= 65:
        return SizingAction.SIZE_1_0
    
    # Low confidence
    else:
        return SizingAction.SIZE_0_5


def create_sizing_agent(config: Optional[SizingAgentConfig] = None) -> SizingAgent:
    """Factory function to create SizingAgent."""
    return SizingAgent(config)
