"""
Ensemble Decision Maker
=======================

Combines all 3 RL agents (Entry, Sizing, Exit) into unified trading decisions.

Features:
- Voting mechanism: 2/3 agents must agree to execute
- Confidence scoring based on vote agreement
- Fallback to rule-based when agents disagree
- Safety limits enforcement
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .entry_agent import EntryAgent, EntryAction, EntryAgentConfig
from .sizing_agent import SizingAgent, SizingAction, SizingAgentConfig, POSITION_SIZES
from .exit_agent import ExitAgent, ExitAction, ExitAgentConfig
from .rl_core import StateEncoder, RewardCalculator, RLConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for Ensemble Decision Maker."""
    
    # Voting thresholds
    min_agreement: int = 2  # Minimum agents that must agree (out of 3)
    high_confidence_agreement: int = 3  # All agree = high confidence
    
    # Agent-specific configs (use defaults if None)
    entry_config: Optional[EntryAgentConfig] = None
    sizing_config: Optional[SizingAgentConfig] = None
    exit_config: Optional[ExitAgentConfig] = None
    
    # Rollout percentage (for gradual deployment)
    rl_usage_pct: float = 1.0  # 1.0 = 100% RL, 0.5 = 50% RL + 50% rule-based
    
    # Safety limits
    max_position_size: float = 0.10  # 10% max regardless of agent decision
    min_signal_score: float = 55.0   # Don't trade below this confidence
    
    # Override flags
    allow_cancel: bool = True   # Allow entry agent to cancel signals
    force_immediate: bool = False  # Always use immediate entry


@dataclass
class TradeDecision:
    """Output from ensemble decision making."""
    
    # Decision
    should_trade: bool = False
    
    # Entry decision
    entry_action: EntryAction = EntryAction.IMMEDIATE
    wait_seconds: int = 0
    
    # Sizing decision
    sizing_action: SizingAction = SizingAction.SIZE_2_0
    position_size_pct: float = 2.0
    
    # Exit decision
    exit_action: ExitAction = ExitAction.HOLD_NORMAL_5
    stop_loss_pct: float = 5.0
    
    # Confidence metrics
    confidence_score: float = 0.0
    agents_agreed: int = 0
    
    # Agent votes
    entry_vote: Dict = field(default_factory=dict)
    sizing_vote: Dict = field(default_factory=dict)
    exit_vote: Dict = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    decision_source: str = "ensemble"  # "ensemble", "rule_based", "hybrid"


class EnsembleDecisionMaker:
    """
    Orchestrates all 3 RL agents for unified trade decisions.
    
    Workflow:
    1. Encode state from technical indicators
    2. Get entry decision (should we wait/cancel?)
    3. Get sizing decision (how much to trade?)
    4. Get exit decision (what stop strategy?)
    5. Apply voting to ensure agent agreement
    6. Output unified TradeDecision
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # Initialize agents
        self.entry_agent = EntryAgent(self.config.entry_config)
        self.sizing_agent = SizingAgent(self.config.sizing_config)
        self.exit_agent = ExitAgent(self.config.exit_config)
        
        # State encoder
        self.state_encoder = StateEncoder()
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Statistics
        self.decisions_made = 0
        self.trades_executed = 0
        self.trades_filtered = 0
    
    def make_decision(self,
                      indicators: Dict,
                      signal_score: float,
                      signal_type: str,
                      account_state: Optional[Dict] = None,
                      use_rl: bool = True) -> TradeDecision:
        """
        Make unified trading decision from all agents.
        
        Args:
            indicators: Dict with VCP, SuperTrend, RSI, IV data
            signal_score: AI signal generator confidence (0-100)
            signal_type: "BUY_CALL", "BUY_PUT", etc.
            account_state: Dict with equity, drawdown, etc.
            use_rl: If False, use rule-based fallback
        
        Returns:
            TradeDecision with all parameters
        """
        self.decisions_made += 1
        
        # Check minimum signal score
        if signal_score < self.config.min_signal_score:
            return self._create_no_trade_decision("Below minimum score")
        
        # Decide whether to use RL or rule-based
        if use_rl and np.random.random() < self.config.rl_usage_pct:
            decision = self._make_rl_decision(
                indicators, signal_score, signal_type, account_state
            )
        else:
            decision = self._make_rule_based_decision(
                indicators, signal_score, signal_type, account_state
            )
        
        # Track statistics
        if decision.should_trade:
            self.trades_executed += 1
        else:
            self.trades_filtered += 1
        
        return decision
    
    def _make_rl_decision(self,
                          indicators: Dict,
                          signal_score: float,
                          signal_type: str,
                          account_state: Optional[Dict]) -> TradeDecision:
        """Make decision using RL agents."""
        
        # Add signal score to indicators for encoding
        indicators = dict(indicators)
        indicators['signal_score'] = signal_score
        
        # Encode state
        state = self.state_encoder.encode_state(indicators, account_state)
        
        # Get entry decision
        entry_action, entry_log_prob, entry_value = self.entry_agent.select_action(state)
        entry_vote = {
            'action': entry_action.name,
            'log_prob': entry_log_prob,
            'value': entry_value,
            'policy': self.entry_agent.get_policy(state),
        }
        
        # If entry agent cancels, don't trade
        if entry_action == EntryAction.CANCEL and self.config.allow_cancel:
            return self._create_no_trade_decision(
                "Entry agent canceled signal",
                entry_vote=entry_vote
            )
        
        # Force immediate if configured
        if self.config.force_immediate:
            entry_action = EntryAction.IMMEDIATE
        
        # Get sizing decision
        sizing_action, q_values = self.sizing_agent.select_action(state, training=False)
        sizing_vote = {
            'action': sizing_action.name,
            'position_size': self.sizing_agent.get_position_size_pct(sizing_action),
            'q_values': self.sizing_agent.get_q_values(state),
        }
        
        # Get exit decision (use same state for now, will update in position)
        exit_action, exit_log_prob, exit_value = self.exit_agent.select_action(state)
        exit_vote = {
            'action': exit_action.name,
            'log_prob': exit_log_prob,
            'value': exit_value,
            'policy': self.exit_agent.get_policy(state),
        }
        
        # Apply voting
        agents_agreed = self._count_agreement(entry_action, sizing_action, exit_action)
        confidence = self._calculate_confidence(
            entry_vote, sizing_vote, exit_vote, agents_agreed
        )
        
        # Check agreement threshold
        if agents_agreed < self.config.min_agreement:
            # Use rule-based fallback when agents disagree
            return self._make_rule_based_decision(
                indicators, signal_score, signal_type, account_state,
                partial_rl={
                    'entry_vote': entry_vote,
                    'sizing_vote': sizing_vote,
                    'exit_vote': exit_vote,
                }
            )
        
        # Apply safety limits
        position_size = self.sizing_agent.get_position_size(sizing_action)
        position_size = min(position_size, self.config.max_position_size)
        
        # Build decision
        decision = TradeDecision(
            should_trade=True,
            entry_action=entry_action,
            wait_seconds=self.entry_agent.get_wait_time(entry_action),
            sizing_action=sizing_action,
            position_size_pct=position_size * 100,
            exit_action=exit_action,
            stop_loss_pct=self._get_stop_pct(exit_action),
            confidence_score=confidence,
            agents_agreed=agents_agreed,
            entry_vote=entry_vote,
            sizing_vote=sizing_vote,
            exit_vote=exit_vote,
            timestamp=datetime.now().isoformat(),
            decision_source="ensemble",
        )
        
        return decision
    
    def _make_rule_based_decision(self,
                                   indicators: Dict,
                                   signal_score: float,
                                   signal_type: str,
                                   account_state: Optional[Dict],
                                   partial_rl: Optional[Dict] = None) -> TradeDecision:
        """Make decision using rule-based fallback."""
        from .entry_agent import select_entry_action_rule_based
        from .sizing_agent import select_size_rule_based
        from .exit_agent import select_exit_rule_based
        
        # Entry
        entry_action = select_entry_action_rule_based(indicators, signal_score)
        
        if entry_action == EntryAction.CANCEL:
            return self._create_no_trade_decision("Rule-based: signal too weak")
        
        # Sizing
        sizing_action = select_size_rule_based(indicators, signal_score, account_state)
        
        # Exit (use 0 P&L since we haven't entered yet)
        exit_action = select_exit_rule_based(0, 0, indicators)
        
        # Apply safety limit
        position_size = min(
            POSITION_SIZES[sizing_action],
            self.config.max_position_size
        )
        
        decision = TradeDecision(
            should_trade=True,
            entry_action=entry_action,
            wait_seconds=5 if entry_action == EntryAction.WAIT_5S else (
                10 if entry_action == EntryAction.WAIT_10S else 0
            ),
            sizing_action=sizing_action,
            position_size_pct=position_size * 100,
            exit_action=exit_action,
            stop_loss_pct=self._get_stop_pct(exit_action),
            confidence_score=signal_score / 100,
            agents_agreed=0,
            entry_vote=partial_rl.get('entry_vote', {}) if partial_rl else {},
            sizing_vote=partial_rl.get('sizing_vote', {}) if partial_rl else {},
            exit_vote=partial_rl.get('exit_vote', {}) if partial_rl else {},
            timestamp=datetime.now().isoformat(),
            decision_source="hybrid" if partial_rl else "rule_based",
        )
        
        return decision
    
    def _count_agreement(self,
                         entry: EntryAction,
                         sizing: SizingAction,
                         exit: ExitAction) -> int:
        """
        Count how many agents agree the trade should proceed.
        
        Agreement criteria:
        - Entry: Not CANCEL
        - Sizing: At least 1% position
        - Exit: Not immediate take profit (willing to hold)
        """
        agreeing = 0
        
        # Entry agrees if not canceling
        if entry != EntryAction.CANCEL:
            agreeing += 1
        
        # Sizing agrees if recommending reasonable position
        if sizing in [SizingAction.SIZE_1_0, SizingAction.SIZE_2_0,
                      SizingAction.SIZE_3_0, SizingAction.SIZE_5_0,
                      SizingAction.SIZE_8_0]:
            agreeing += 1
        
        # Exit agrees if willing to hold (not immediate exit)
        if exit != ExitAction.TAKE_PROFIT:
            agreeing += 1
        
        return agreeing
    
    def _calculate_confidence(self,
                              entry_vote: Dict,
                              sizing_vote: Dict,
                              exit_vote: Dict,
                              agents_agreed: int) -> float:
        """Calculate overall confidence score."""
        
        # Base confidence from agreement
        agreement_conf = agents_agreed / 3
        
        # Agent confidence from value estimates
        entry_value = entry_vote.get('value', 0)
        exit_value = exit_vote.get('value', 0)
        
        value_conf = (entry_value + exit_value) / 2
        value_conf = np.clip(value_conf, 0, 1)
        
        # Q-value confidence from sizing
        q_values = sizing_vote.get('q_values', {})
        if q_values:
            max_q = max(q_values.values())
            avg_q = sum(q_values.values()) / len(q_values)
            q_conf = np.clip((max_q - avg_q), 0, 1)
        else:
            q_conf = 0.5
        
        # Weighted average
        confidence = (
            0.5 * agreement_conf +
            0.3 * value_conf +
            0.2 * q_conf
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _get_stop_pct(self, exit_action: ExitAction) -> float:
        """Get stop loss percentage for exit action."""
        stops = {
            ExitAction.TAKE_PROFIT: 0.0,
            ExitAction.HOLD_TIGHT_2: 2.0,
            ExitAction.HOLD_NORMAL_5: 5.0,
            ExitAction.HOLD_LOOSE_8: 8.0,
        }
        return stops.get(exit_action, 5.0)
    
    def _create_no_trade_decision(self,
                                   reason: str,
                                   entry_vote: Optional[Dict] = None
                                  ) -> TradeDecision:
        """Create decision for filtered/canceled trade."""
        return TradeDecision(
            should_trade=False,
            confidence_score=0.0,
            agents_agreed=0,
            entry_vote=entry_vote or {},
            timestamp=datetime.now().isoformat(),
            decision_source=f"filtered: {reason}",
        )
    
    def update_exit_decision(self,
                             entry_price: float,
                             current_price: float,
                             time_in_trade: float,
                             indicators: Dict,
                             account_state: Optional[Dict] = None
                            ) -> Tuple[ExitAction, Dict]:
        """
        Update exit decision during active position.
        
        Called periodically to check if exit strategy should change.
        """
        from .exit_agent import encode_position_state
        
        # Current P&L
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # Encode position state
        state = encode_position_state(
            entry_price, current_price, pnl_pct,
            time_in_trade, indicators, account_state
        )
        
        # Get updated decision
        exit_action, log_prob, value = self.exit_agent.select_action(
            state, deterministic=True  # Use best action for active position
        )
        
        vote = {
            'action': exit_action.name,
            'log_prob': log_prob,
            'value': value,
            'current_pnl_pct': pnl_pct * 100,
            'time_in_trade_min': time_in_trade / 60,
        }
        
        return exit_action, vote
    
    def save(self, directory: str):
        """Save all agent weights."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.entry_agent.save(f"{directory}/entry_best.weights.h5")
        self.sizing_agent.save(f"{directory}/sizing_best.weights.h5")
        self.exit_agent.save(f"{directory}/exit_best.weights.h5")
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load(self, directory: str):
        """Load all agent weights."""
        self.entry_agent.load(f"{directory}/entry_best.weights.h5")
        self.sizing_agent.load(f"{directory}/sizing_best.weights.h5")
        self.exit_agent.load(f"{directory}/exit_best.weights.h5")
        
        logger.info(f"Ensemble loaded from {directory}")
    
    def get_stats(self) -> Dict:
        """Get decision statistics."""
        return {
            'decisions_made': self.decisions_made,
            'trades_executed': self.trades_executed,
            'trades_filtered': self.trades_filtered,
            'execution_rate': (
                self.trades_executed / self.decisions_made
                if self.decisions_made > 0 else 0
            ),
            'filter_rate': (
                self.trades_filtered / self.decisions_made
                if self.decisions_made > 0 else 0
            ),
        }


def create_ensemble(config: Optional[EnsembleConfig] = None) -> EnsembleDecisionMaker:
    """Factory function to create EnsembleDecisionMaker."""
    return EnsembleDecisionMaker(config)
