"""
Data Models for Volatility-Aware Options Stop-Loss System
==========================================================

Contains:
- PositionStatus: Enum for tracking position lifecycle
- OptionPosition: Complete position state with Greeks, underlying data, stop state
- VolatilityTracker: VIX/ATR-based index volatility tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import math


class PositionStatus(Enum):
    """Lifecycle status of an option position."""
    TRACKING = "tracking"              # Normal monitoring
    EXIT_TRIGGERED = "exit_triggered"  # Underlying hit stop level
    EXIT_ORDER_PLACED = "exit_order_placed"  # Limit sell order placed
    EXIT_FILLED = "exit_filled"        # Order filled, position closed
    CLOSED = "closed"                  # Fully closed and archived


@dataclass
class OptionPosition:
    """
    Represents an open option position with full state tracking.
    
    Includes:
    - Contract identification
    - Entry information
    - Current market data (bid/ask)
    - Greeks from IBKR (tick type 13)
    - Underlying reference data (price, beta)
    - Stop-loss state (underlying-based)
    - Exit order state (smart limit execution)
    - P&L tracking
    """
    
    # ===== Identity =====
    conid: int                          # IB Contract ID
    symbol: str                         # Ticker (e.g., "SPY")
    sectype: str = "OPT"
    exchange: str = "SMART"
    currency: str = "USD"
    
    # ===== Contract Specification =====
    expiry: str = ""                    # YYYYMMDD format
    strike: float = 0.0
    right: str = "C"                    # "C" for call, "P" for put
    
    # ===== Entry Information =====
    quantity: int = 0
    avg_entry_price: float = 0.0       # $ per contract
    entry_time: datetime = field(default_factory=datetime.now)
    
    # ===== Underlying Reference =====
    underlying_symbol: str = ""         # e.g., "SPY" for SPY options
    underlying_beta: float = 1.0        # Beta vs SPX (default=1.0)
    underlying_entry_price: float = 0.0 # Underlying price at option entry
    
    # ===== Current Market Data =====
    current_bid: Optional[float] = None
    current_ask: Optional[float] = None
    current_last: Optional[float] = None
    last_update: Optional[datetime] = None
    
    # ===== Greeks (from tick type 13 - Model) =====
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    implied_vol: Optional[float] = None  # IV as decimal (0.25 = 25%)
    model_price: Optional[float] = None  # Theoretical price from IB model
    underlying_price: Optional[float] = None  # Current underlying spot
    
    # ===== Stop-Loss State (Underlying-Driven) =====
    underlying_high: float = 0.0        # Highest underlying since entry
    underlying_stop_level: Optional[float] = None  # Stop trigger price
    trail_distance_dollars: Optional[float] = None
    trail_distance_pct: Optional[float] = None
    
    # ===== Order State =====
    stop_order_id: Optional[int] = None
    stop_order_price: Optional[float] = None
    stop_order_time: Optional[datetime] = None
    
    # ===== Exit State (Smart Execution) =====
    status: PositionStatus = PositionStatus.TRACKING
    exit_triggered: bool = False
    exit_triggered_time: Optional[datetime] = None
    exit_order_id: Optional[int] = None
    exit_limit_price: Optional[float] = None
    exit_reprices: int = 0
    exit_last_reprice_time: Optional[datetime] = None
    
    # ===== P&L =====
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_price: Optional[float] = None
    closed_time: Optional[datetime] = None
    
    # ===== Helper Methods =====
    
    def days_to_expiry(self) -> int:
        """Calculate days remaining until expiration."""
        if not self.expiry:
            return 30  # Default fallback
        try:
            exp_date = datetime.strptime(self.expiry, "%Y%m%d")
            return max((exp_date - datetime.now()).days, 0)
        except ValueError:
            return 30
    
    def is_deep_itm(self) -> bool:
        """Check if option is deep in the money (>10%)."""
        if self.underlying_price is None:
            return False
        if self.right == "C":
            return self.underlying_price > self.strike * 1.10
        else:  # Put
            return self.underlying_price < self.strike * 0.90
    
    def update_underlying_high(self) -> None:
        """Track highest underlying price since entry (for trailing)."""
        if self.underlying_price is not None:
            if self.underlying_high == 0:
                self.underlying_high = self.underlying_entry_price or self.underlying_price
            self.underlying_high = max(self.underlying_high, self.underlying_price)
    
    def is_ready_for_trading(self) -> bool:
        """Check if position has all required data for stop management."""
        return all([
            self.current_bid is not None,
            self.current_ask is not None,
            self.underlying_price is not None,
            self.delta is not None,
        ])
    
    def __str__(self) -> str:
        right_str = "CALL" if self.right == "C" else "PUT"
        dte = self.days_to_expiry()
        if self.current_bid and self.current_ask:
            bid_ask = f"${self.current_bid:.2f}/${self.current_ask:.2f}"
        else:
            bid_ask = "N/A"
        return f"{self.symbol} {self.expiry} {self.strike} {right_str} x{self.quantity} | {bid_ask} | DTE:{dte}"


@dataclass
class VolatilityTracker:
    """
    Tracks index (SPX) volatility via VIX or ATR.
    
    Used to dynamically size stop distances based on current market conditions.
    """
    
    # VIX-based (preferred)
    vix_level: Optional[float] = None
    vix_update_time: Optional[datetime] = None
    
    # ATR-based (alternative)
    atr_14_dollars: Optional[float] = None
    atr_update_time: Optional[datetime] = None
    
    # SPX reference
    spx_price: Optional[float] = None
    
    def get_daily_vol_pct(self) -> float:
        """
        Return daily volatility as a decimal (e.g., 0.0126 = 1.26%).
        
        VIX is annualized, so divide by sqrt(252 trading days).
        Example: VIX 20 → 0.20 / sqrt(252) ≈ 0.0126
        """
        if self.vix_level is not None:
            return self.vix_level / 100.0 / math.sqrt(252)
        elif self.atr_14_dollars is not None and self.spx_price is not None:
            return self.atr_14_dollars / self.spx_price
        else:
            return 0.012  # Default ~1.2% daily (VIX ~19)
    
    def update_vix(self, vix: float) -> None:
        """Update VIX level."""
        self.vix_level = vix
        self.vix_update_time = datetime.now()
    
    def update_atr(self, atr: float) -> None:
        """Update ATR."""
        self.atr_14_dollars = atr
        self.atr_update_time = datetime.now()
    
    def update_spx_price(self, price: float) -> None:
        """Update SPX reference price."""
        self.spx_price = price
    
    def __str__(self) -> str:
        vol_pct = self.get_daily_vol_pct() * 100
        return f"VIX: {self.vix_level or 'N/A'}, Daily Vol: {vol_pct:.2f}%"


# Beta lookup table for common symbols
# Source: Historical betas vs S&P 500
BETA_TABLE = {
    # Index ETFs
    "SPY": 1.00,
    "QQQ": 1.20,
    "IWM": 1.15,
    "DIA": 0.95,
    "VTI": 1.00,
    
    # Tech (high beta)
    "AAPL": 1.25,
    "MSFT": 1.10,
    "GOOGL": 1.15,
    "AMZN": 1.30,
    "META": 1.35,
    "NVDA": 1.80,
    "TSLA": 2.00,
    "AMD": 1.70,
    
    # Financials
    "JPM": 1.10,
    "BAC": 1.30,
    "GS": 1.35,
    
    # Healthcare (lower beta)
    "JNJ": 0.65,
    "UNH": 0.85,
    "PFE": 0.70,
    
    # Consumer staples (defensive)
    "KO": 0.55,
    "PG": 0.45,
    "WMT": 0.50,
    
    # Energy (volatile)
    "XOM": 0.90,
    "CVX": 0.95,
    
    # Commodities
    "GLD": 0.05,
    "SLV": 0.20,
    "USO": 0.60,
}


def get_beta(symbol: str) -> float:
    """
    Get beta for a symbol from lookup table.
    Returns 1.0 (market beta) if symbol not found.
    """
    return BETA_TABLE.get(symbol.upper(), 1.0)
