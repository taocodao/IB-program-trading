"""
RL Agent Training Pipeline
==========================

Scripts for training the 3 RL agents on historical trade data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import os
import json
from datetime import datetime

try:
    from src.rl.rl_core import StateEncoder, RewardCalculator, ExperienceReplayBuffer, TrajectoryBuffer
    from src.rl.entry_agent import EntryAgent, EntryAction, EntryAgentConfig
    from src.rl.sizing_agent import SizingAgent, SizingAction, SizingAgentConfig
    from src.rl.exit_agent import ExitAgent, ExitAction, ExitAgentConfig
    from src.rl.ensemble import EnsembleDecisionMaker, EnsembleConfig
except ImportError:
    # Fallback for relative import if run as module
    from .rl_core import StateEncoder, RewardCalculator, ExperienceReplayBuffer, TrajectoryBuffer
    from .entry_agent import EntryAgent, EntryAction, EntryAgentConfig
    from .sizing_agent import SizingAgent, SizingAction, SizingAgentConfig
    from .exit_agent import ExitAgent, ExitAction, ExitAgentConfig
    from .ensemble import EnsembleDecisionMaker, EnsembleConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    
    # Data
    data_path: str = "data/historical_trades.json"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Early stopping
    patience: int = 10
    min_improvement: float = 0.001
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/rl"
    save_every: int = 10
    
    # Logging
    log_every: int = 10
    

class RLTrainer:
    """
    Trainer for all 3 RL agents.
    
    Workflow:
    1. Load historical trade data
    2. Encode states using StateEncoder
    3. Train each agent separately
    4. Validate on held-out data
    5. Save best checkpoints
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator()
        
        # Initialize agents
        self.entry_agent = EntryAgent()
        self.sizing_agent = SizingAgent()
        self.exit_agent = ExitAgent()
        
        # Training data
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        # Metrics
        self.training_history = {
            'entry': [],
            'sizing': [],
            'exit': [],
        }
    
    def load_data(self, filepath: str = None) -> int:
        """
        Load historical trade data.
        
        Expected format:
        [
            {
                "symbol": "AAPL",
                "signal_type": "BUY_CALL",
                "signal_score": 75,
                "indicators": {...},
                "entry_price": 150.0,
                "exit_price": 155.0,
                "entry_time": "2024-01-01T10:00:00",
                "exit_time": "2024-01-01T14:00:00",
                "position_size_pct": 2.0,
                "account_state": {...}
            },
            ...
        ]
        """
        filepath = filepath or self.config.data_path
        
        if not os.path.exists(filepath):
            logger.warning(f"Data file not found: {filepath}")
            return 0
        
        with open(filepath, 'r') as f:
            all_data = json.load(f)
        
        # Shuffle
        np.random.shuffle(all_data)
        
        # Split
        n = len(all_data)
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        
        self.train_data = all_data[:train_end]
        self.val_data = all_data[train_end:val_end]
        self.test_data = all_data[val_end:]
        
        logger.info(f"Loaded {n} trades: train={len(self.train_data)}, "
                   f"val={len(self.val_data)}, test={len(self.test_data)}")
        
        return n
    
    def prepare_training_batch(self, 
                               trades: List[Dict],
                               batch_size: int = None
                              ) -> Tuple[np.ndarray, ...]:
        """
        Prepare batch of training data.
        
        Returns:
            states, entry_actions, sizing_actions, exit_actions, rewards
        """
        batch_size = batch_size or self.config.batch_size
        batch = np.random.choice(trades, size=min(batch_size, len(trades)), replace=False)
        
        states = []
        entry_actions = []
        sizing_actions = []
        exit_actions = []
        entry_rewards = []
        sizing_rewards = []
        exit_rewards = []
        
        for trade in batch:
            # Encode state
            indicators = trade.get('indicators', {})
            indicators['signal_score'] = trade.get('signal_score', 70)
            account_state = trade.get('account_state', None)
            
            state = self.state_encoder.encode_state(indicators, account_state)
            states.append(state)
            
            # Calculate P&L
            entry_price = trade.get('entry_price', 100)
            exit_price = trade.get('exit_price', 100)
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Infer "correct" actions from outcome (hindsight labeling)
            # Entry: Should have traded immediately if profitable
            if pnl_pct > 0.02:
                entry_action = EntryAction.IMMEDIATE
            elif pnl_pct > 0:
                entry_action = EntryAction.WAIT_5S
            elif pnl_pct > -0.02:
                entry_action = EntryAction.WAIT_10S
            else:
                entry_action = EntryAction.CANCEL
            entry_actions.append(entry_action)
            
            # Sizing: Larger for winners, smaller for losers
            position_used = trade.get('position_size_pct', 2.0)
            if pnl_pct > 0.03:
                sizing_action = SizingAction.SIZE_5_0 if position_used >= 3 else SizingAction.SIZE_8_0
            elif pnl_pct > 0:
                sizing_action = SizingAction.SIZE_3_0
            elif pnl_pct > -0.02:
                sizing_action = SizingAction.SIZE_1_0
            else:
                sizing_action = SizingAction.SIZE_0_5
            sizing_actions.append(sizing_action)
            
            # Exit: Based on final outcome
            if pnl_pct > 0.03:
                exit_action = ExitAction.TAKE_PROFIT
            elif pnl_pct > 0:
                exit_action = ExitAction.HOLD_TIGHT_2
            elif pnl_pct > -0.02:
                exit_action = ExitAction.HOLD_NORMAL_5
            else:
                exit_action = ExitAction.HOLD_LOOSE_8
            exit_actions.append(exit_action)
            
            # Rewards (simplified)
            entry_rewards.append(1.0 if pnl_pct > 0 else -0.5)
            sizing_rewards.append(pnl_pct * position_used / 2)  # Risk-adjusted
            exit_rewards.append(pnl_pct * 5)  # Scale for training
        
        return (
            np.array(states, dtype=np.float32),
            np.array(entry_actions, dtype=np.int32),
            np.array(sizing_actions, dtype=np.int32),
            np.array(exit_actions, dtype=np.int32),
            np.array(entry_rewards, dtype=np.float32),
            np.array(sizing_rewards, dtype=np.float32),
            np.array(exit_rewards, dtype=np.float32),
        )
    
    def train_entry_agent(self, epochs: int = None) -> Dict:
        """Train entry agent on historical data."""
        epochs = epochs or self.config.epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            batch = self.prepare_training_batch(self.train_data)
            states, entry_actions, _, _, entry_rewards, _, _ = batch
            
            # Create next_states (simplified: same as states)
            next_states = np.roll(states, -1, axis=0)
            dones = np.zeros(len(states))
            dones[-1] = 1
            
            metrics = self.entry_agent.train_batch(
                states, entry_actions, entry_rewards, next_states, dones
            )
            
            # Validation
            if epoch % self.config.log_every == 0:
                val_batch = self.prepare_training_batch(self.val_data)
                val_loss = metrics.get('total_loss', 0)
                
                logger.info(f"Entry Agent - Epoch {epoch}: loss={val_loss:.4f}")
                
                self.training_history['entry'].append({
                    'epoch': epoch,
                    'train_loss': metrics.get('total_loss', 0),
                    'val_loss': val_loss,
                })
                
                # Early stopping
                if val_loss < best_val_loss - self.config.min_improvement:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.entry_agent.save(f"{self.config.checkpoint_dir}/entry_best.weights.h5")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info("Early stopping triggered for entry agent")
                        break
        
        return {
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
        }
    
    def train_sizing_agent(self, epochs: int = None) -> Dict:
        """Train sizing agent on historical data."""
        epochs = epochs or self.config.epochs
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            batch = self.prepare_training_batch(self.train_data)
            states, _, sizing_actions, _, _, sizing_rewards, _ = batch
            
            next_states = np.roll(states, -1, axis=0)
            dones = np.zeros(len(states))
            dones[-1] = 1
            
            # Train using DQN experience replay
            for i in range(len(states)):
                self.sizing_agent.remember(
                    states[i], sizing_actions[i], sizing_rewards[i],
                    next_states[i], bool(dones[i])
                )
            
            loss = self.sizing_agent.replay()
            
            if epoch % self.config.log_every == 0:
                loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                logger.info(f"Sizing Agent - Epoch {epoch}: loss={loss_str}, eps={self.sizing_agent.epsilon:.3f}")
                
                if loss and loss < best_val_loss:
                    best_val_loss = loss
                    self.sizing_agent.save(f"{self.config.checkpoint_dir}/sizing_best.weights.h5")
        
        return {
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
        }
    
    def train_exit_agent(self, epochs: int = None) -> Dict:
        """Train exit agent on historical data."""
        epochs = epochs or self.config.epochs
        
        for epoch in range(epochs):
            batch = self.prepare_training_batch(self.train_data)
            states, _, _, exit_actions, _, _, exit_rewards = batch
            
            # Build trajectories for PPO
            for i in range(len(states)):
                # Pad state to match ExitAgent expectation (25 features)
                # Historical training approximation: 21 features + 4 padding
                exit_state = np.pad(states[i], (0, 4), 'constant')
                
                # Get action and log_prob
                action, log_prob, value = self.exit_agent.select_action(exit_state)
                
                self.exit_agent.store_transition(
                    exit_state, exit_actions[i], exit_rewards[i],
                    log_prob, value, i == len(states) - 1
                )
            
            # End trajectory and train
            self.exit_agent.end_trajectory()
            metrics = self.exit_agent.train()
            
            if epoch % self.config.log_every == 0:
                loss = metrics.get('total_loss', 0) if metrics else 0
                logger.info(f"Exit Agent - Epoch {epoch}: loss={loss:.4f}")
        
        self.exit_agent.save(f"{self.config.checkpoint_dir}/exit_best.weights.h5")
        
        return {'epochs_trained': epoch + 1}
    
    def train_all(self, epochs: int = None) -> Dict:
        """Train all agents."""
        epochs = epochs or self.config.epochs
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Training Entry Agent (A2C)")
        logger.info("=" * 50)
        entry_results = self.train_entry_agent(epochs)
        
        logger.info("=" * 50)
        logger.info("Training Sizing Agent (DQN)")
        logger.info("=" * 50)
        sizing_results = self.train_sizing_agent(epochs)
        
        logger.info("=" * 50)
        logger.info("Training Exit Agent (PPO)")
        logger.info("=" * 50)
        exit_results = self.train_exit_agent(epochs)
        
        return {
            'entry': entry_results,
            'sizing': sizing_results,
            'exit': exit_results,
        }
    
    def evaluate(self) -> Dict:
        """Evaluate trained agents on test data."""
        if not self.test_data:
            return {'error': 'No test data available'}
        
        correct_entry = 0
        correct_sizing = 0
        correct_exit = 0
        total = 0
        
        for trade in self.test_data:
            indicators = trade.get('indicators', {})
            indicators['signal_score'] = trade.get('signal_score', 70)
            state = self.state_encoder.encode_state(indicators)
            
            # Get agent predictions
            entry_pred, _, _ = self.entry_agent.select_action(state, deterministic=True)
            sizing_pred, _ = self.sizing_agent.select_action(state, training=False)
            
            # Pad state for ExitAgent
            exit_state = np.pad(state, (0, 4), 'constant')
            exit_pred, _, _ = self.exit_agent.select_action(exit_state, deterministic=True)
            
            # Compare with outcome
            pnl_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            
            # Entry: correct if didn't cancel on winner, or canceled on loser
            if (pnl_pct > 0 and entry_pred != EntryAction.CANCEL) or \
               (pnl_pct < -0.02 and entry_pred == EntryAction.CANCEL):
                correct_entry += 1
            
            # Sizing: correct direction (larger for winners)
            if (pnl_pct > 0.02 and sizing_pred >= SizingAction.SIZE_3_0) or \
               (pnl_pct <= 0 and sizing_pred <= SizingAction.SIZE_2_0):
                correct_sizing += 1
            
            # Exit: correct if took profit on winner or held through dip
            if (pnl_pct > 0.03 and exit_pred == ExitAction.TAKE_PROFIT) or \
               (pnl_pct > 0 and exit_pred == ExitAction.HOLD_TIGHT_2):
                correct_exit += 1
            
            total += 1
        
        return {
            'entry_accuracy': correct_entry / total if total > 0 else 0,
            'sizing_accuracy': correct_sizing / total if total > 0 else 0,
            'exit_accuracy': correct_exit / total if total > 0 else 0,
            'total_samples': total,
        }


def run_training(data_path: str = None, 
                 epochs: int = 100,
                 checkpoint_dir: str = "checkpoints/rl"):
    """
    Main training entry point.
    
    Usage:
    ```python
    from src.rl.training import run_training
    run_training(data_path="data/trades.json", epochs=100)
    ```
    """
    config = TrainingConfig(
        data_path=data_path or "data/historical_trades.json",
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )
    
    trainer = RLTrainer(config)
    
    # Load data
    n_trades = trainer.load_data()
    if n_trades == 0:
        logger.error("No training data found")
        return None
    
    # Train all agents
    results = trainer.train_all()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    logger.info("=" * 50)
    logger.info("Training Complete")
    logger.info(f"Entry Accuracy: {eval_results['entry_accuracy']*100:.1f}%")
    logger.info(f"Sizing Accuracy: {eval_results['sizing_accuracy']*100:.1f}%")
    logger.info(f"Exit Accuracy: {eval_results['exit_accuracy']*100:.1f}%")
    logger.info("=" * 50)
    
    return {
        'training': results,
        'evaluation': eval_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
