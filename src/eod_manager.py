"""
End-of-Day Position Manager
============================

Implements 4 EOD closing strategies:
1. Close Winners Daily - Close profitable positions by 3 PM
2. Friday Close All - Close all positions Friday 3 PM
3. Intraday Only - Never hold overnight
4. User Discretion - No automated closes

Based on: Risk-Tolerance-Trade-Frequency-EOD-Strategies.md
"""

import logging
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from risk_config import EODStrategy

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position for EOD management."""
    symbol: str
    entry_price: float
    current_price: float
    entry_time: datetime
    quantity: int
    option_type: str  # "CALL" or "PUT"
    expiry: str
    strike: float
    
    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def pnl_dollars(self) -> float:
        """Calculate P&L in dollars."""
        return (self.current_price - self.entry_price) * self.quantity * 100
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.pnl_pct > 0
    
    @property
    def hold_duration_hours(self) -> float:
        """Hours since entry."""
        return (datetime.now() - self.entry_time).total_seconds() / 3600


class EODManager:
    """
    Manages end-of-day position closing based on user strategy.
    """
    
    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EOD_CHECK_TIME = time(15, 0)  # 3:00 PM
    FINAL_EXIT_TIME = time(15, 55)  # 3:55 PM
    
    def __init__(
        self,
        strategy: EODStrategy,
        close_position_callback: Callable[[Position], bool],
        profit_threshold: float = 0.10,  # 10% profit to close winners
        loss_threshold: float = 0.05,    # 5% loss tightens stop
    ):
        self.strategy = strategy
        self.close_position = close_position_callback
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        
        self.positions_closed_today: List[str] = []
        self.last_check_time: Optional[datetime] = None
        
        logger.info(f"EODManager initialized: {strategy.value}")
    
    def check_and_execute(self, positions: List[Position]) -> Dict[str, str]:
        """
        Check if any positions should be closed based on EOD strategy.
        
        Returns: Dict of {symbol: reason} for closed positions
        """
        now = datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        closed = {}
        
        # Only check during market hours
        if not self._is_market_hours(current_time):
            return closed
        
        # Route to appropriate strategy
        if self.strategy == EODStrategy.CLOSE_WINNERS:
            closed = self._close_winners_strategy(positions, current_time)
        
        elif self.strategy == EODStrategy.FRIDAY_CLOSE:
            closed = self._friday_close_strategy(positions, current_time, current_day)
        
        elif self.strategy == EODStrategy.INTRADAY_ONLY:
            closed = self._intraday_only_strategy(positions, current_time)
        
        elif self.strategy == EODStrategy.USER_DISCRETION:
            # No automated closes
            closed = {}
        
        self.last_check_time = now
        return closed
    
    def _is_market_hours(self, current_time: time) -> bool:
        """Check if within market hours."""
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE
    
    # ============= Strategy 1: Close Winners Daily =============
    
    def _close_winners_strategy(
        self,
        positions: List[Position],
        current_time: time
    ) -> Dict[str, str]:
        """
        Close all profitable positions by 3 PM.
        Tighten stops on losing positions.
        Close everything by 3:55 PM.
        """
        closed = {}
        
        # 3:00 PM - Close winners, tighten losers
        if current_time >= self.EOD_CHECK_TIME:
            for pos in positions:
                if pos.symbol in self.positions_closed_today:
                    continue
                
                if pos.pnl_pct >= self.profit_threshold:
                    # Close winner
                    success = self.close_position(pos)
                    if success:
                        closed[pos.symbol] = f"EOD_WINNER (+{pos.pnl_pct*100:.1f}%)"
                        self.positions_closed_today.append(pos.symbol)
                        logger.info(f"✅ EOD Close Winner: {pos.symbol} +{pos.pnl_pct*100:.1f}%")
                
                elif pos.pnl_pct < -self.loss_threshold:
                    # Losing position - log for potential stop tightening
                    logger.warning(f"⚠️ EOD Loser: {pos.symbol} {pos.pnl_pct*100:.1f}% - tighten stop")
        
        # 3:55 PM - Close everything remaining
        if current_time >= self.FINAL_EXIT_TIME:
            for pos in positions:
                if pos.symbol not in self.positions_closed_today:
                    success = self.close_position(pos)
                    if success:
                        closed[pos.symbol] = f"EOD_FINAL ({pos.pnl_pct*100:+.1f}%)"
                        self.positions_closed_today.append(pos.symbol)
                        logger.info(f"📤 EOD Final Close: {pos.symbol} {pos.pnl_pct*100:+.1f}%")
        
        return closed
    
    # ============= Strategy 2: Friday Close All =============
    
    def _friday_close_strategy(
        self,
        positions: List[Position],
        current_time: time,
        current_day: str
    ) -> Dict[str, str]:
        """
        Close all positions Friday at 3 PM.
        Monday-Thursday: Normal trading, hold overnight OK.
        """
        closed = {}
        
        if current_day != "Friday":
            return closed  # Only act on Friday
        
        # Friday 3:00 PM - Close everything
        if current_time >= self.EOD_CHECK_TIME:
            for pos in positions:
                if pos.symbol in self.positions_closed_today:
                    continue
                
                success = self.close_position(pos)
                if success:
                    pnl_str = f"+{pos.pnl_pct*100:.1f}%" if pos.is_profitable else f"{pos.pnl_pct*100:.1f}%"
                    closed[pos.symbol] = f"FRIDAY_CLOSE ({pnl_str})"
                    self.positions_closed_today.append(pos.symbol)
                    logger.info(f"📅 Friday Close: {pos.symbol} {pnl_str}")
        
        return closed
    
    # ============= Strategy 3: Intraday Only =============
    
    def _intraday_only_strategy(
        self,
        positions: List[Position],
        current_time: time
    ) -> Dict[str, str]:
        """
        Never hold overnight.
        Enter within 1 hour of open (9:30-10:30 AM)
        Exit by 3:00 PM latest.
        """
        closed = {}
        
        # 3:00 PM - Close everything
        if current_time >= self.EOD_CHECK_TIME:
            for pos in positions:
                if pos.symbol in self.positions_closed_today:
                    continue
                
                # Only close positions opened today
                if pos.entry_time.date() == datetime.now().date():
                    success = self.close_position(pos)
                    if success:
                        hours_held = pos.hold_duration_hours
                        pnl_str = f"+{pos.pnl_pct*100:.1f}%" if pos.is_profitable else f"{pos.pnl_pct*100:.1f}%"
                        closed[pos.symbol] = f"INTRADAY_EXIT ({pnl_str}, {hours_held:.1f}h)"
                        self.positions_closed_today.append(pos.symbol)
                        logger.info(f"🌅 Intraday Exit: {pos.symbol} {pnl_str} after {hours_held:.1f}h")
        
        return closed
    
    # ============= Utility Methods =============
    
    def reset_daily(self):
        """Reset daily tracking (call at market open)."""
        self.positions_closed_today = []
        logger.info("EODManager: Daily reset complete")
    
    def should_enter_trade(self, current_time: time = None) -> bool:
        """
        Check if we should enter new trades based on EOD strategy.
        
        Intraday strategy: Only enter 9:30-10:30 AM
        Other strategies: Enter anytime before 3:00 PM
        """
        if current_time is None:
            current_time = datetime.now().time()
        
        if self.strategy == EODStrategy.INTRADAY_ONLY:
            # Only enter first hour
            entry_window_end = time(10, 30)
            return self.MARKET_OPEN <= current_time <= entry_window_end
        
        else:
            # Enter anytime before EOD check
            return current_time < self.EOD_CHECK_TIME
    
    def get_status(self) -> Dict:
        """Get current EOD manager status."""
        now = datetime.now()
        
        return {
            "strategy": self.strategy.value,
            "current_time": now.strftime("%H:%M"),
            "positions_closed_today": len(self.positions_closed_today),
            "market_hours": self._is_market_hours(now.time()),
            "should_enter": self.should_enter_trade(now.time()),
            "time_to_eod_check": self._time_until(self.EOD_CHECK_TIME),
            "time_to_close": self._time_until(self.MARKET_CLOSE),
        }
    
    def _time_until(self, target_time: time) -> str:
        """Calculate time until a target time."""
        now = datetime.now()
        target = datetime.combine(now.date(), target_time)
        
        if target < now:
            return "Past"
        
        diff = target - now
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{hours}h {minutes}m"


# ============= IV Crush Avoidance Manager =============

class EarningsManager:
    """
    Manages positions around earnings events to avoid IV crush.
    """
    
    def __init__(self, earnings_calendar: Dict[str, datetime] = None):
        self.earnings_calendar = earnings_calendar or {}
    
    def days_to_earnings(self, symbol: str) -> Optional[int]:
        """Get days until next earnings for a symbol."""
        earnings_date = self.earnings_calendar.get(symbol)
        if not earnings_date:
            return None
        
        days = (earnings_date - datetime.now()).days
        return days
    
    def should_avoid_symbol(self, symbol: str, avoid_earnings: bool) -> bool:
        """Check if we should avoid trading this symbol due to earnings."""
        if not avoid_earnings:
            return False
        
        days = self.days_to_earnings(symbol)
        if days is None:
            return False  # No earnings data, proceed
        
        # Avoid if earnings within 5 days
        return 0 <= days <= 5
    
    def get_earnings_strategy(self, symbol: str) -> str:
        """Get recommended strategy based on earnings timing."""
        days = self.days_to_earnings(symbol)
        
        if days is None:
            return "NORMAL"
        
        if days < 0:
            return "NORMAL"  # Earnings passed
        
        if days == 0:
            return "EXIT_BEFORE_ANNOUNCEMENT"
        
        if days <= 5:
            return "AVOID_OR_REDUCE_SIZE"
        
        if days <= 14:
            return "NORMAL_CAUTION"
        
        return "NORMAL"


# ============= CLI Demo =============

if __name__ == "__main__":
    print("EOD Manager Demo")
    print("=" * 50)
    
    # Mock close function
    def mock_close(pos: Position) -> bool:
        print(f"  [MOCK] Closing {pos.symbol}")
        return True
    
    # Create manager
    manager = EODManager(
        strategy=EODStrategy.CLOSE_WINNERS,
        close_position_callback=mock_close
    )
    
    # Mock positions
    positions = [
        Position(
            symbol="AAPL",
            entry_price=5.00,
            current_price=5.75,  # +15%
            entry_time=datetime.now() - timedelta(hours=3),
            quantity=2,
            option_type="CALL",
            expiry="20260123",
            strike=185.0
        ),
        Position(
            symbol="NVDA",
            entry_price=8.00,
            current_price=7.20,  # -10%
            entry_time=datetime.now() - timedelta(hours=2),
            quantity=1,
            option_type="CALL",
            expiry="20260123",
            strike=145.0
        ),
    ]
    
    print(f"\nStrategy: {manager.strategy.value}")
    print(f"Current status: {manager.get_status()}")
    print(f"\nPositions:")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.pnl_pct*100:+.1f}% (${pos.pnl_dollars:+.2f})")
    
    print("\nChecking EOD rules...")
    closed = manager.check_and_execute(positions)
    
    if closed:
        print(f"\nClosed positions: {closed}")
    else:
        print("\nNo positions closed (not yet 3 PM)")
