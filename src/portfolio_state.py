"""
Portfolio State Management
==========================

Manages portfolio-level risk constraints:
- Max positions
- Daily loss limits
- Per-position loss tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
from models import OptionPosition, PositionStatus


@dataclass
class PortfolioConfig:
    """Portfolio risk configuration."""
    
    # Portfolio size
    portfolio_size: float = 100_000
    
    # Position limits
    max_positions: int = 10
    
    # Loss limits per option (based on option cost)
    max_loss_pct_per_day: float = 0.10      # 10% per day
    max_loss_pct_total: float = 0.30        # 30% total
    
    # Daily portfolio loss limit
    max_daily_portfolio_loss: float = 0.025  # 2.5% of portfolio per day
    
    @property
    def max_daily_portfolio_loss_dollars(self) -> float:
        return self.portfolio_size * self.max_daily_portfolio_loss


@dataclass
class PositionRiskState:
    """Risk tracking for a single position."""
    conid: int
    symbol: str
    
    # Entry info
    entry_price: float  # Option price per share
    quantity: int
    entry_time: datetime
    
    # Cost basis
    total_cost: float  # entry_price * 100 * quantity
    
    # Loss limits (dollar amounts)
    max_daily_loss: float      # 10% of cost
    max_total_loss: float      # 30% of cost
    
    # Tracking
    daily_pnl_realized: float = 0.0
    total_pnl_realized: float = 0.0
    current_unrealized_pnl: float = 0.0
    
    # High water marks
    best_unrealized_pnl: float = 0.0
    worst_unrealized_pnl: float = 0.0
    
    # State
    daily_limit_hit: bool = False
    total_limit_hit: bool = False
    
    def update_unrealized(self, current_price: float):
        """Update unrealized P&L from current option price."""
        self.current_unrealized_pnl = (current_price - self.entry_price) * 100 * self.quantity
        self.best_unrealized_pnl = max(self.best_unrealized_pnl, self.current_unrealized_pnl)
        self.worst_unrealized_pnl = min(self.worst_unrealized_pnl, self.current_unrealized_pnl)
    
    def check_limits(self) -> Optional[str]:
        """Check if any loss limits are hit. Returns reason if triggered."""
        # Check daily limit
        if self.current_unrealized_pnl <= -self.max_daily_loss:
            self.daily_limit_hit = True
            return "daily_loss_limit"
        
        # Check total limit
        if self.current_unrealized_pnl <= -self.max_total_loss:
            self.total_limit_hit = True
            return "total_loss_limit"
        
        return None


@dataclass
class PortfolioState:
    """
    Portfolio-level state and risk management.
    
    Tracks:
    - Active positions and their risk
    - Daily P&L
    - Position count limits
    """
    
    config: PortfolioConfig = field(default_factory=PortfolioConfig)
    
    # Active positions
    positions: Dict[int, PositionRiskState] = field(default_factory=dict)
    
    # Closed positions (today)
    closed_today: List[PositionRiskState] = field(default_factory=list)
    
    # Daily tracking
    trading_date: date = field(default_factory=date.today)
    daily_realized_pnl: float = 0.0
    
    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.trading_date = date.today()
        self.daily_realized_pnl = 0.0
        self.closed_today.clear()
        
        # Reset daily counters on positions
        for pos in self.positions.values():
            pos.daily_pnl_realized = 0.0
            pos.daily_limit_hit = False
    
    def can_open_new(self) -> bool:
        """Check if we can open a new position."""
        # Check position count
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Check daily loss limit
        if self.daily_realized_pnl <= -self.config.max_daily_portfolio_loss_dollars:
            return False
        
        return True
    
    def add_position(
        self,
        conid: int,
        symbol: str,
        entry_price: float,
        quantity: int = 1
    ) -> Optional[PositionRiskState]:
        """Add a new position with risk tracking."""
        if not self.can_open_new():
            return None
        
        total_cost = entry_price * 100 * quantity
        
        risk_state = PositionRiskState(
            conid=conid,
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            total_cost=total_cost,
            max_daily_loss=total_cost * self.config.max_loss_pct_per_day,
            max_total_loss=total_cost * self.config.max_loss_pct_total,
        )
        
        self.positions[conid] = risk_state
        return risk_state
    
    def close_position(self, conid: int, exit_price: float) -> Optional[float]:
        """
        Close a position and record P&L.
        Returns realized P&L or None if position not found.
        """
        pos = self.positions.pop(conid, None)
        if not pos:
            return None
        
        realized_pnl = (exit_price - pos.entry_price) * 100 * pos.quantity
        pos.total_pnl_realized = realized_pnl
        pos.daily_pnl_realized = realized_pnl
        
        self.daily_realized_pnl += realized_pnl
        self.closed_today.append(pos)
        
        return realized_pnl
    
    def get_position(self, conid: int) -> Optional[PositionRiskState]:
        """Get risk state for a position."""
        return self.positions.get(conid)
    
    def update_position_price(self, conid: int, current_price: float) -> Optional[str]:
        """
        Update position with current price.
        Returns trigger reason if loss limit hit.
        """
        pos = self.positions.get(conid)
        if not pos:
            return None
        
        pos.update_unrealized(current_price)
        return pos.check_limits()
    
    @property
    def total_positions(self) -> int:
        return len(self.positions)
    
    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.current_unrealized_pnl for p in self.positions.values())
    
    @property
    def total_cost_basis(self) -> float:
        return sum(p.total_cost for p in self.positions.values())
    
    def summary(self) -> str:
        """Generate portfolio summary."""
        lines = [
            "=" * 50,
            "PORTFOLIO STATE",
            "=" * 50,
            f"Date: {self.trading_date}",
            f"Positions: {self.total_positions}/{self.config.max_positions}",
            f"",
            f"Cost Basis: ${self.total_cost_basis:,.2f}",
            f"Unrealized P&L: ${self.total_unrealized_pnl:+,.2f}",
            f"Daily Realized: ${self.daily_realized_pnl:+,.2f}",
            f"",
            f"Daily Loss Limit: ${self.config.max_daily_portfolio_loss_dollars:,.2f}",
            f"Can Open New: {'YES' if self.can_open_new() else 'NO'}",
            "=" * 50,
        ]
        return "\n".join(lines)
