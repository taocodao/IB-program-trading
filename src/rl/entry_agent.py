"""
A2C Entry Optimization Agent
============================

Advantage Actor-Critic (A2C) agent for optimizing entry timing.

Actions:
- IMMEDIATE: Enter trade immediately at current price
- WAIT_5S: Wait 5 seconds for potentially better fill
- WAIT_10S: Wait 10 seconds for better fill
- CANCEL: Cancel this signal (filter out)

The agent learns to:
1. Identify signals where waiting improves fill price
2. Cancel signals that are likely to fail
3. Execute immediately when price is optimal
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import IntEnum
import logging
import os

logger = logging.getLogger(__name__)

# Try to import TensorFlow, fall back to numpy-only implementation
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - using numpy-only implementation")


class EntryAction(IntEnum):
    """Entry timing actions."""
    IMMEDIATE = 0
    WAIT_5S = 1
    WAIT_10S = 2
    CANCEL = 3


@dataclass
class EntryAgentConfig:
    """Configuration for Entry Agent."""
    state_size: int = 21
    action_size: int = 4
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    value_coef: float = 0.5  # Value loss weight
    max_grad_norm: float = 0.5  # Gradient clipping
    

if TF_AVAILABLE:
    class ActorCriticNetwork(Model):
        """
        Neural network for A2C agent.
        
        Architecture:
        - Shared layers: 128 → 64 (ReLU)
        - Actor head: 64 → 4 (softmax) - action probabilities
        - Critic head: 64 → 1 (linear) - state value
        """
        
        def __init__(self, config: EntryAgentConfig):
            super(ActorCriticNetwork, self).__init__()
            
            # Shared feature extraction
            self.dense1 = layers.Dense(config.hidden_size_1, activation='relu',
                                       kernel_initializer='he_normal')
            self.dense2 = layers.Dense(config.hidden_size_2, activation='relu',
                                       kernel_initializer='he_normal')
            self.dropout = layers.Dropout(0.2)
            
            # Actor head (outputs action probabilities)
            self.actor_dense = layers.Dense(32, activation='relu',
                                           kernel_initializer='he_normal')
            self.actor_output = layers.Dense(config.action_size, activation='softmax')
            
            # Critic head (outputs state value)
            self.critic_dense = layers.Dense(32, activation='relu',
                                            kernel_initializer='he_normal')
            self.critic_output = layers.Dense(1)
        
        def call(self, inputs, training=False):
            """Forward pass."""
            # Shared layers
            x = self.dense1(inputs)
            x = self.dense2(x)
            if training:
                x = self.dropout(x, training=training)
            
            # Actor and Critic branches
            actor_features = self.actor_dense(x)
            action_probs = self.actor_output(actor_features)
            
            critic_features = self.critic_dense(x)
            value = self.critic_output(critic_features)
            
            return action_probs, value
        
        def get_action(self, state: np.ndarray, deterministic: bool = False
                      ) -> Tuple[int, float, float]:
            """
            Select action based on current policy.
            
            Args:
                state: Current state vector
                deterministic: If True, select most probable action
            
            Returns:
                action: Selected action index
                log_prob: Log probability of selected action
                value: State value estimate
            """
            state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
            action_probs, value = self(state, training=False)
            
            action_probs = action_probs[0]
            value = value[0, 0]
            
            if deterministic:
                action = tf.argmax(action_probs).numpy()
            else:
                # Sample from probability distribution
                action = tf.random.categorical(tf.math.log(action_probs[np.newaxis]), 1)[0, 0].numpy()
            
            log_prob = tf.math.log(action_probs[action] + 1e-8).numpy()
            
            return int(action), float(log_prob), float(value)


class EntryAgent:
    """
    A2C agent for entry timing optimization.
    
    Learns when to:
    - Enter immediately (good current price)
    - Wait for better fill (price likely to improve)
    - Cancel signal (likely to fail)
    """
    
    # Action mapping
    ACTIONS = {
        EntryAction.IMMEDIATE: {'wait_seconds': 0, 'name': 'IMMEDIATE'},
        EntryAction.WAIT_5S: {'wait_seconds': 5, 'name': 'WAIT_5S'},
        EntryAction.WAIT_10S: {'wait_seconds': 10, 'name': 'WAIT_10S'},
        EntryAction.CANCEL: {'wait_seconds': -1, 'name': 'CANCEL'},  # -1 = cancel
    }
    
    def __init__(self, config: Optional[EntryAgentConfig] = None):
        self.config = config or EntryAgentConfig()
        
        if TF_AVAILABLE:
            self.network = ActorCriticNetwork(self.config)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=self.config.max_grad_norm
            )
            
            # Build network with dummy input
            dummy_input = np.zeros((1, self.config.state_size), dtype=np.float32)
            self.network(dummy_input)
        else:
            self.network = None
            logger.warning("EntryAgent initialized without TensorFlow - training disabled")
        
        # Training metrics
        self.train_step_count = 0
        self.episode_rewards = []
        self.episode_actions = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False
                     ) -> Tuple[EntryAction, float, float]:
        """
        Select entry action based on current state.
        
        Args:
            state: State vector from StateEncoder
            deterministic: If True, always select best action
        
        Returns:
            action: EntryAction enum value
            log_prob: Log probability (for training)
            value: State value estimate (for training)
        """
        if self.network is None:
            # Fallback: always immediate
            return EntryAction.IMMEDIATE, 0.0, 0.0
        
        action_idx, log_prob, value = self.network.get_action(state, deterministic)
        return EntryAction(action_idx), log_prob, value
    
    def get_wait_time(self, action: EntryAction) -> int:
        """Get wait time in seconds for action (-1 = cancel)."""
        return self.ACTIONS[action]['wait_seconds']
    
    def should_cancel(self, action: EntryAction) -> bool:
        """Check if action is to cancel the signal."""
        return action == EntryAction.CANCEL
    
    def train_step(self, 
                   state: np.ndarray,
                   action: int,
                   reward: float,
                   next_state: np.ndarray,
                   done: bool) -> float:
        """
        Single training step using A2C.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        
        Returns:
            Total loss value
        """
        if not TF_AVAILABLE or self.network is None:
            return 0.0
        
        state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state[np.newaxis], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Current state evaluation
            action_probs, value = self.network(state, training=True)
            
            # Next state evaluation (for TD target)
            _, next_value = self.network(next_state, training=False)
            next_value = next_value[0, 0] if not done else 0.0
            
            # TD Error = Advantage
            td_target = reward + self.config.gamma * next_value
            td_error = td_target - value[0, 0]
            advantage = td_error
            
            # Actor loss: -log(π(a|s)) * A
            action_log_prob = tf.math.log(action_probs[0, action] + 1e-8)
            actor_loss = -action_log_prob * tf.stop_gradient(advantage)
            
            # Entropy bonus (encourages exploration)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8))
            entropy_loss = -self.config.entropy_coef * entropy
            
            # Critic loss: MSE of value function
            critic_loss = self.config.value_coef * tf.square(td_error)
            
            # Total loss
            total_loss = actor_loss + critic_loss + entropy_loss
        
        # Gradient update
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        
        self.train_step_count += 1
        
        return total_loss.numpy()
    
    def train_batch(self, 
                    states: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_states: np.ndarray,
                    dones: np.ndarray) -> Dict[str, float]:
        """
        Train on batch of experience.
        
        Returns dict with loss metrics.
        """
        if not TF_AVAILABLE or self.network is None:
            return {'total_loss': 0.0}
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs, values = self.network(states, training=True)
            _, next_values = self.network(next_states, training=False)
            
            # Compute advantages
            next_values = tf.where(dones.astype(bool), 0.0, next_values[:, 0])
            td_targets = rewards + self.config.gamma * next_values
            advantages = td_targets - values[:, 0]
            
            # Actor loss
            batch_size = states.shape[0]
            action_indices = tf.stack([tf.range(batch_size), actions], axis=1)
            selected_probs = tf.gather_nd(action_probs, action_indices)
            log_probs = tf.math.log(selected_probs + 1e-8)
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
            
            # Entropy loss
            entropy = -tf.reduce_mean(tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-8), axis=1
            ))
            entropy_loss = -self.config.entropy_coef * entropy
            
            # Critic loss
            critic_loss = self.config.value_coef * tf.reduce_mean(tf.square(advantages))
            
            # Total
            total_loss = actor_loss + critic_loss + entropy_loss
        
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        
        self.train_step_count += 1
        
        return {
            'total_loss': float(total_loss),
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'entropy': float(entropy),
        }
    
    def save(self, filepath: str):
        """Save model weights."""
        if self.network is None:
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.network.save_weights(filepath)
        logger.info(f"EntryAgent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        if self.network is None:
            return
        
        self.network.load_weights(filepath)
        logger.info(f"EntryAgent loaded from {filepath}")
    
    def get_action_name(self, action: EntryAction) -> str:
        """Get human-readable action name."""
        return self.ACTIONS[action]['name']
    
    def get_policy(self, state: np.ndarray) -> Dict[str, float]:
        """Get action probabilities for state."""
        if self.network is None:
            return {a.name: 0.25 for a in EntryAction}
        
        state = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        action_probs, _ = self.network(state, training=False)
        probs = action_probs[0].numpy()
        
        return {EntryAction(i).name: float(probs[i]) for i in range(len(probs))}


# --- Standalone functions for non-TF usage ---

def create_entry_agent(config: Optional[EntryAgentConfig] = None) -> EntryAgent:
    """Factory function to create EntryAgent."""
    return EntryAgent(config)


def select_entry_action_rule_based(indicators: Dict,
                                    signal_score: float) -> EntryAction:
    """
    Rule-based fallback for entry action selection.
    
    Used when TensorFlow is not available or agent not trained.
    """
    # High confidence signals: enter immediately
    if signal_score >= 80:
        return EntryAction.IMMEDIATE
    
    # Medium confidence: wait for better fill
    elif signal_score >= 70:
        iv_high = indicators.get('iv_percentile', 50) > 70
        if iv_high:
            return EntryAction.WAIT_5S  # Wait when IV is high
        return EntryAction.IMMEDIATE
    
    # Lower confidence: wait or cancel
    elif signal_score >= 60:
        return EntryAction.WAIT_10S
    
    else:
        return EntryAction.CANCEL
