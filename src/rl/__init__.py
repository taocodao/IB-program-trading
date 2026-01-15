"""
Reinforcement Learning Module for IB Options Trading
=====================================================

This module integrates 3 RL agents to enhance the VCP + SuperTrend + RSI trading system:

1. Entry Agent (A2C) - Optimizes entry timing
2. Sizing Agent (DQN) - Adapts position sizing
3. Exit Agent (PPO) - Optimizes exit strategy

Together with ensemble voting, expected to improve win rate from 70% to 75-85%.
"""

from .rl_core import (
    StateEncoder,
    RewardCalculator,
    ExperienceReplayBuffer,
    TrajectoryBuffer,
    RLConfig,
)
from .entry_agent import EntryAgent, EntryAction, EntryAgentConfig
from .sizing_agent import SizingAgent, SizingAction, SizingAgentConfig
from .exit_agent import ExitAgent, ExitAction, ExitAgentConfig
from .ensemble import EnsembleDecisionMaker, EnsembleConfig, TradeDecision
from .integration import RLEnhancedExecutor, integrate_rl_with_trading_system

__all__ = [
    # Core
    'StateEncoder',
    'RewardCalculator', 
    'ExperienceReplayBuffer',
    'TrajectoryBuffer',
    'RLConfig',
    # Entry Agent
    'EntryAgent',
    'EntryAction',
    'EntryAgentConfig',
    # Sizing Agent
    'SizingAgent',
    'SizingAction',
    'SizingAgentConfig',
    # Exit Agent
    'ExitAgent',
    'ExitAction',
    'ExitAgentConfig',
    # Ensemble
    'EnsembleDecisionMaker',
    'EnsembleConfig',
    'TradeDecision',
    # Integration
    'RLEnhancedExecutor',
    'integrate_rl_with_trading_system',
]
