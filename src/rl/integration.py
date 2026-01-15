"""
RL Integration with Trading System
===================================

Provides the RLEnhancedExecutor that wraps the ensemble decision maker
and integrates with the existing trading_system.py execution flow.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time

from .ensemble import EnsembleDecisionMaker, TradeDecision, EnsembleConfig
from .entry_agent import EntryAction
from .exit_agent import ExitAction
from .rl_core import RewardCalculator

logger = logging.getLogger(__name__)


@dataclass 
class ExecutionResult:
    """Result from RL-enhanced trade execution."""
    
    # Decision
    decision: TradeDecision = None
    
    # Execution
    order_placed: bool = False
    fill_price: float = 0.0
    position_size_dollars: float = 0.0
    
    # Stop order
    stop_order_placed: bool = False
    stop_price: float = 0.0
    
    # Metrics (for training)
    baseline_price: float = 0.0  # Price at signal time
    wait_time_seconds: float = 0.0
    
    # Status
    status: str = ""
    error: Optional[str] = None


class RLEnhancedExecutor:
    """
    Integrates RL ensemble with trading system execution.
    
    Workflow:
    1. Receive signal from AI signal generator
    2. Get RL decision (entry timing, position size, exit strategy)
    3. Execute with optional waiting
    4. Place position with RL-determined stop
    5. Track for training feedback
    """
    
    def __init__(self, 
                 trading_system=None,  # Reference to TradingSystem instance
                 ensemble: Optional[EnsembleDecisionMaker] = None,
                 config: Optional[EnsembleConfig] = None):
        
        self.trading_system = trading_system
        self.ensemble = ensemble or EnsembleDecisionMaker(config)
        self.reward_calculator = RewardCalculator()
        
        # Track executions for training
        self.pending_executions = {}  # order_id -> ExecutionResult
        self.completed_trades = []
        
        # Mode
        self.enabled = True
        self.training_mode = False  # Collect data but don't train live
    
    def execute_signal(self,
                       symbol: str,
                       signal_type: str,
                       signal_score: float,
                       indicators: Dict,
                       current_price: float,
                       account_equity: float,
                       account_state: Optional[Dict] = None
                      ) -> ExecutionResult:
        """
        Execute trade with RL enhancement.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            signal_type: "BUY_CALL", "BUY_PUT", etc.
            signal_score: AI signal confidence (0-100)
            indicators: Dict with technical indicators
            current_price: Current underlying price
            account_equity: Account value
            account_state: Optional account context
        
        Returns:
            ExecutionResult with decision and execution details
        """
        result = ExecutionResult()
        
        if not self.enabled:
            result.status = "RL executor disabled"
            return result
        
        # Record baseline price (for reward calculation)
        result.baseline_price = current_price
        
        # Get RL decision
        decision = self.ensemble.make_decision(
            indicators=indicators,
            signal_score=signal_score,
            signal_type=signal_type,
            account_state=account_state,
            use_rl=self.enabled
        )
        result.decision = decision
        
        # Check if should trade
        if not decision.should_trade:
            result.status = f"Signal filtered: {decision.decision_source}"
            logger.info(f"RL filtered signal for {symbol}: {decision.decision_source}")
            return result
        
        # Apply entry timing
        if decision.wait_seconds > 0:
            logger.info(f"RL waiting {decision.wait_seconds}s for better entry on {symbol}")
            result.wait_time_seconds = decision.wait_seconds
            time.sleep(decision.wait_seconds)
        
        # Calculate position size
        position_size_frac = decision.position_size_pct / 100
        position_size_dollars = account_equity * position_size_frac
        result.position_size_dollars = position_size_dollars
        
        # Calculate stop price
        stop_pct = decision.stop_loss_pct / 100
        stop_price = current_price * (1 - stop_pct)
        result.stop_price = stop_price
        
        # Execute trade (if trading_system is available)
        # Execute trade (if trading_system is available)
        if self.trading_system is not None:
            try:
                # Place entry order via trading system
                # We interpret position_size_pct to calculate quantity
                # Assuming simple quantity=1 for now unless trading system supports dynamic sizing
                
                # Get beta for symbol if available (usually in watchlist)
                beta = 1.0
                if hasattr(self.trading_system, 'watchlist'):
                    for item in self.trading_system.watchlist:
                        if item['symbol'] == symbol:
                            beta = item['beta']
                            break
                
                # Call trading system
                self.trading_system.place_option_order(
                    symbol=symbol,
                    underlying_price=current_price,
                    beta=beta,
                    ai_score=signal_score,
                    quantity=max(1, int(result.position_size_dollars / 500)), # Estimate quantity ($500 per contract approx)
                    stop_pct=decision.stop_loss_pct / 100
                )
                
                result.order_placed = True
                result.fill_price = current_price  # Simulated for now
                result.stop_order_placed = True
                result.status = "Order placed with RL optimization"
                
                logger.info(
                    f"RL executed {signal_type} on {symbol}: "
                    f"size={decision.position_size_pct:.1f}%, "
                    f"stop={decision.stop_loss_pct:.0f}%, "
                    f"entry={decision.entry_action.name}"
                )
                
            except Exception as e:
                result.error = str(e)
                result.status = f"Execution error: {e}"
                logger.error(f"RL execution error for {symbol}: {e}")
        else:
            # Simulation mode
            result.order_placed = True
            result.fill_price = current_price
            result.stop_order_placed = True
            result.status = "Simulated (no trading system)"
        
        return result
    
    def get_exit_update(self,
                        symbol: str,
                        entry_price: float,
                        current_price: float,
                        time_in_trade: float,
                        indicators: Dict,
                        account_state: Optional[Dict] = None
                       ) -> Tuple[ExitAction, float]:
        """
        Get updated exit decision for active position.
        
        Called periodically to check if stop should be adjusted.
        
        Returns:
            exit_action: Updated exit action
            new_stop_price: Updated stop price
        """
        exit_action, vote = self.ensemble.update_exit_decision(
            entry_price=entry_price,
            current_price=current_price,
            time_in_trade=time_in_trade,
            indicators=indicators,
            account_state=account_state
        )
        
        # Calculate new stop price
        if exit_action == ExitAction.TAKE_PROFIT:
            new_stop_price = current_price * 0.995  # Tight stop for exit
        else:
            stop_pct = {
                ExitAction.HOLD_TIGHT_2: 0.02,
                ExitAction.HOLD_NORMAL_5: 0.05,
                ExitAction.HOLD_LOOSE_8: 0.08,
            }.get(exit_action, 0.05)
            new_stop_price = entry_price * (1 - stop_pct)
        
        return exit_action, new_stop_price
    
    def record_trade_result(self,
                            symbol: str,
                            entry_price: float,
                            exit_price: float,
                            position_size: float,
                            decision: TradeDecision,
                            max_favorable: float = None):
        """
        Record trade result for RL training feedback.
        
        Called when position is closed.
        """
        if max_favorable is None:
            max_favorable = max(entry_price, exit_price)
        
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # Calculate rewards for each agent
        entry_reward = self.reward_calculator.calculate_entry_reward(
            fill_price=entry_price,
            baseline_price=decision.entry_vote.get('baseline_price', entry_price),
            action=decision.entry_action.name
        )
        
        sizing_reward = self.reward_calculator.calculate_sizing_reward(
            pnl_pct=pnl_pct,
            position_size_pct=decision.position_size_pct / 100,
            max_drawdown=0  # Would need actual drawdown tracking
        )
        
        exit_reward = self.reward_calculator.calculate_exit_reward(
            entry_price=entry_price,
            exit_price=exit_price,
            max_favorable=max_favorable,
            action=decision.exit_action.name
        )
        
        # Store for training
        self.completed_trades.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'position_size': position_size,
            'decision': decision,
            'rewards': {
                'entry': entry_reward,
                'sizing': sizing_reward,
                'exit': exit_reward,
            },
            'timestamp': datetime.now().isoformat(),
        })
        
        logger.info(
            f"Trade recorded: {symbol} P&L={pnl_pct*100:.2f}%, "
            f"rewards=[entry={entry_reward:.2f}, size={sizing_reward:.2f}, exit={exit_reward:.2f}]"
        )
    
    def get_training_data(self) -> list:
        """Get completed trades for training."""
        return self.completed_trades
    
    def clear_training_data(self):
        """Clear completed trades after training."""
        self.completed_trades = []
    
    def save(self, directory: str):
        """Save ensemble weights."""
        self.ensemble.save(directory)
    
    def load(self, directory: str):
        """Load ensemble weights."""
        self.ensemble.load(directory)
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        stats = self.ensemble.get_stats()
        stats['completed_trades'] = len(self.completed_trades)
        
        if self.completed_trades:
            pnls = [t['pnl_pct'] for t in self.completed_trades]
            stats['avg_pnl_pct'] = np.mean(pnls) * 100
            stats['win_rate'] = sum(1 for p in pnls if p > 0) / len(pnls)
        
        return stats


def integrate_rl_with_trading_system(trading_system, 
                                     ensemble_config: Optional[EnsembleConfig] = None
                                    ) -> RLEnhancedExecutor:
    """
    Factory function to create and integrate RL executor.
    
    Usage in trading_system.py:
    ```python
    from src.rl.integration import integrate_rl_with_trading_system
    
    # In TradingSystem.__init__:
    self.rl_executor = integrate_rl_with_trading_system(self)
    
    # In signal handler:
    result = self.rl_executor.execute_signal(
        symbol=symbol,
        signal_type=signal.signal_type.value,
        signal_score=signal.consensus_score,
        indicators=indicator_data,
        current_price=current_price,
        account_equity=self.account_equity,
    )
    ```
    """
    return RLEnhancedExecutor(
        trading_system=trading_system,
        config=ensemble_config
    )
