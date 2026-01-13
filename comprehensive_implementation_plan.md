# COMPREHENSIVE IMPLEMENTATION PLAN
## Advanced Volatility-Aware Options Stop-Loss Management System
### Interactive Brokers API with Beta + Index Volatility Sizing

**Document Version**: 2.0  
**Date**: January 9, 2026  
**Target**: Claude 3.5 Opus Code Generation  
**Status**: Production-Ready Specification  

---

## EXECUTIVE SUMMARY

This document specifies a production-grade Python system that automates trailing stop-loss management for options positions on Interactive Brokers. The system improves upon basic stop sizing by:

1. **Risk-aware stops**: Dynamically sizes stop distances using underlying beta and current index volatility
2. **Underlying-driven triggers**: Initiates stops based on underlying price movement, not noisy option quotes
3. **Smart execution**: Uses theoretical pricing and bid/ask context to execute exits in illiquid options
4. **Adaptive re-pricing**: Automatically adjusts limit prices if execution doesn't fill

**Key metrics**:
- Stops sized as: `stop_distance = k × beta × index_volatility`
- Execution uses theoretical option pricing with bid/ask guardrails
- Fully integrated with IBKR TWS API via Python `ibapi`
- Paper trading ready, cloud-deployable, production-safe

---

## TABLE OF CONTENTS

1. [System Architecture](#system-architecture)
2. [Data Requirements & Collection](#data-requirements--collection)
3. [Mathematical Framework](#mathematical-framework)
4. [Detailed Class Specifications](#detailed-class-specifications)
5. [API Integration Details](#api-integration-details)
6. [Stop-Loss Logic Implementation](#stop-loss-logic-implementation)
7. [Order Execution Strategy](#order-execution-strategy)
8. [Error Handling & Recovery](#error-handling--recovery)
9. [Deployment & Configuration](#deployment--configuration)
10. [Testing Protocol](#testing-protocol)
11. [Monitoring & Observability](#monitoring--observability)
12. [Code Examples](#code-examples)

---

## SYSTEM ARCHITECTURE

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   Interactive Brokers Account                    │
│              (Paper Trading or Live)                             │
│  • Open Option Positions                                         │
│  • Real-time Market Data                                         │
│  • Portfolio Data                                                │
└────────────────────┬────────────────────────────────────────────┘
                     │ TWS Socket API (port 7497)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            Python Application Layer                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Connection Manager (EClient + EWrapper)                 │   │
│  │  • Socket lifecycle                                      │   │
│  │  • Message routing                                       │   │
│  │  • Error callbacks                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Data Collection Layer                                    │   │
│  │  • Position loading (reqPositions)                       │   │
│  │  • Market data subscription (reqMktData)                 │   │
│  │  • Greeks reception (tickOptionComputation)              │   │
│  │  • Fundamental data (beta) caching                       │   │
│  │  • Index volatility tracking                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Risk Calculation Engine                                  │   │
│  │  • Beta + index vol stop sizing                          │   │
│  │  • Trailing stop (underlying) logic                      │   │
│  │  • Days-to-expiry adjustments                            │   │
│  │  • Theoretical pricing                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Order Execution Engine                                   │   │
│  │  • Limit order generation                                │   │
│  │  • Spread-aware re-pricing                               │   │
│  │  • Adaptive fill logic                                   │   │
│  │  • Order tracking & confirmation                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ State Management & Persistence                           │   │
│  │  • Position tracking                                     │   │
│  │  • Order history                                         │   │
│  │  • P&L tracking                                          │   │
│  │  • Configuration management                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Monitoring & Logging                                     │   │
│  │  • Real-time event logging                               │   │
│  │  • Performance metrics                                   │   │
│  │  • Audit trail                                           │   │
│  │  • Alert triggers                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Market Open (9:30 AM ET)
    ↓
1. INITIALIZATION PHASE
    ├─ Connect to IB (EClient.connect)
    ├─ Request positions (reqPositions)
    ├─ For each position:
    │   ├─ Fetch beta data
    │   ├─ Request market data (reqMktData)
    │   ├─ Setup Greeks subscriptions
    │   └─ Create OptionPosition object
    └─ Get index volatility baseline
    ↓
2. CONTINUOUS MONITORING PHASE (every tick)
    ├─ Receive tickPrice (bid/ask/last)
    ├─ Receive tickOptionComputation (Greeks)
    ├─ Update index volatility
    ├─ For each position:
    │   ├─ Update current bid/ask/Greeks
    │   ├─ Compute underlying stop distance
    │   ├─ Check: is underlying below stop threshold?
    │   │   └─ YES: Enter EXECUTION phase
    │   │   └─ NO: Continue monitoring
    │   └─ Else: update trailing stop if bid rose
    └─ Log state periodically
    ↓
3. EXECUTION PHASE (triggered by underlying stop)
    ├─ Compute option theoretical price at stop level
    ├─ Read current bid/ask
    ├─ Compute smart limit price (bid/theo/spread function)
    ├─ Place SELL limit order
    ├─ Wait N seconds
    ├─ If unfilled:
    │   ├─ Re-read quotes
    │   ├─ Recompute theoretical
    │   ├─ Adjust limit price down (if still need exit)
    │   └─ Reprice order (repeat)
    ├─ Order fills → Position closed
    └─ Log execution details
    ↓
Market Close (4:00 PM ET)
    └─ Log daily summary
```

---

## DATA REQUIREMENTS & COLLECTION

### 1. Core Position Data

For each open option position, collect and maintain:

```python
@dataclass
class OptionPositionData:
    # Identity
    conid: int                          # Contract ID from IB
    symbol: str                         # Ticker (e.g. "SPY")
    sectype: str                        # Always "OPT"
    exchange: str                       # Exchange (typically "SMART")
    
    # Contract specification
    expiry: str                         # YYYYMMDD format
    strike: float                       # Strike price
    right: str                          # "C" or "P"
    
    # Position info
    quantity: int                       # Contracts held
    avg_entry_price: float              # Entry cost
    entry_time: datetime               # When position opened
    
    # Current market data
    current_bid: Optional[float]        # Latest bid
    current_ask: Optional[float]        # Latest ask
    current_last: Optional[float]       # Latest traded price
    
    # Greeks (from API tickOptionComputation, tick type 13)
    delta: Optional[float]              # Sensitivity to underlying [-1, 1]
    gamma: Optional[float]              # Delta sensitivity
    vega: Optional[float]               # IV sensitivity
    theta: Optional[float]              # Time decay
    implied_vol: Optional[float]        # IV as decimal (e.g. 0.35 = 35%)
    model_price: Optional[float]        # Theoretical price from model
    underlying_price: Optional[float]   # Latest underlying spot
    
    # Underlying risk data
    underlying_beta: Optional[float]    # Beta vs SPX
    underlying_atr: Optional[float]     # 14-day ATR in dollars
    
    # Stop-loss state
    underlying_entry_price: float       # Underlying price at entry
    underlying_stop_level: Optional[float]  # Calculated stop price (underlying)
    underlying_high_since_entry: float  # Highest since entry (trailing)
    trail_distance_dollars: Optional[float]  # Dollar move allowed
    trail_distance_pct: Optional[float]     # % move allowed
    
    # Option order state
    stop_order_id: Optional[int]        # Active STP order ID (if any)
    stop_order_price: Optional[float]   # Limit price of current stop
    stop_order_created_time: Optional[datetime]
    
    # Exit execution state
    exit_triggered: bool                # Has stop condition been met?
    exit_order_id: Optional[int]        # Smart exit order ID
    exit_limit_price: Optional[float]   # Current limit price
    exit_reprices: int                  # Count of reprices
    exit_last_reprice_time: Optional[datetime]
    
    # P&L tracking
    current_mark: Optional[float]       # Mark price for P&L
    unrealized_pnl: Optional[float]     # Current position P&L
    realized_pnl: Optional[float]       # If closed
    closed_at_price: Optional[float]    # Exit price
    closed_at_time: Optional[datetime]
```

### 2. Market Data Subscriptions

Use `reqMktData` with specific tick types:

```python
def subscribe_to_option_data(contract: Contract, req_id: int):
    """
    Subscribe to both bid/ask and Greeks for an option.
    
    Tick types received:
    - 1: BID price
    - 2: ASK price
    - 4: LAST traded price
    - 9: CLOSE price
    - 10: Bid option computation (Greeks based on bid)
    - 11: Ask option computation (Greeks based on ask)
    - 12: Last option computation (Greeks based on last trade)
    - 13: Model option computation (IB's model Greeks, most stable)
    - 24: Implied vol (at-the-money)
    """
    self.client.reqMktData(
        req_id,
        contract,
        "",        # snapshot only for fundamentals
        False,     # regularMktDataOnly
        False,     # manualOrderCancelOrder
        []         # mktDataOptions - empty list = standard
    )
    # Greeks arrive via tickOptionComputation callback (tick type 13 preferred)
    # Bid/Ask prices arrive via tickPrice callback
```

**Critical detail**: Greeks are only available during/shortly after market hours. During after-hours, use delayed data by setting `reqMarketDataType(2)`.

### 3. Index Volatility Tracking

Maintain real-time SPX volatility index via two methods:

#### Method A: VIX-based estimate

```python
def get_index_vol_from_vix(vix_level: float) -> float:
    """
    Convert VIX to daily % volatility.
    VIX is annualized, so divide by sqrt(252) trading days.
    """
    return vix_level / 100.0 / math.sqrt(252)

# Example: VIX 20 → 0.20 / sqrt(252) ≈ 0.0126 (1.26% daily)
# In dollars for SPX at 5800: 5800 * 0.0126 ≈ 73 points ATR equivalent
```

**Implementation**:
- Subscribe to VIX (`MKTINDEX:VIX`) as a market data instrument.
- Receive via `tickPrice` when type = 4 (last).
- Update daily or every 30 seconds.

#### Method B: SPX ATR (more direct)

```python
def compute_atr(prices: List[float], period: int = 14) -> float:
    """
    Compute Average True Range over last N periods.
    Input: list of high-low-close tuples over past 14 days.
    Output: ATR in dollars.
    """
    true_ranges = []
    for i in range(1, len(prices)):
        tr = max(
            prices[i]['high'] - prices[i]['low'],
            abs(prices[i]['high'] - prices[i-1]['close']),
            abs(prices[i]['low'] - prices[i-1]['close'])
        )
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period
```

**Implementation**:
- Fetch 14-20 days of SPX historical bar data (1-day bars).
- Compute ATR daily or on startup.
- Cache and refresh once per market day (or hourly for safety).

### 4. Beta Data Collection

Beta is fundamental to the strategy; source it carefully.

#### Method A: IBKR Fundamental Data (most reliable)

```python
def request_beta(symbol: str) -> float:
    """
    Request fundamental data for a symbol via IBKR API.
    Returns beta vs SPX.
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    
    # Use reportType = "ReportsFinSummary" to get beta
    self.client.reqFundamentalData(
        req_id=self.get_next_req_id(),
        contract=contract,
        reportType="ReportsFinSummary",
        options=[]
    )
    # Result arrives via fundamentalData callback (XML parsing required)
```

**Parsing XML response**:
- Response is raw XML from FactSet.
- Extract tag: `<Beta>1.25</Beta>`
- Cache result (beta changes slowly, refresh weekly).

#### Method B: Preconfigured lookup table

For production: maintain a static dict or database of symbol → beta:

```python
BETA_TABLE = {
    "AAPL": 1.23,
    "MSFT": 0.95,
    "TSLA": 2.10,
    "GLD": 0.10,    # Low beta defensive
    "SPY": 1.00,    # By definition
    "QQQ": 1.45,
    "VTI": 0.98,
    # ... add more as needed
}
```

Use this for fast startup and fallback. Refresh from API once daily.

#### Method C: External data source

Use a lightweight API (Polygon, Yahoo Finance) if IBKR fundamental data is unavailable:

```python
import requests

def get_beta_from_polygon(symbol: str, api_key: str) -> float:
    """Fetch beta from Polygon.io."""
    url = f"https://api.polygon.io/v1/reference/financials?ticker={symbol}&apikey={api_key}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()['results'][0]['beta']
    return 1.0  # Default fallback
```

---

## MATHEMATICAL FRAMEWORK

### 1. Stop Distance Computation

#### Core Formula: Beta × Index Volatility

For each option position, define underlying stop distance:

```
M_stock = k × β × σ_index
```

Where:
- `k` = aggression factor (0.7–1.5, tunable)
- `β` = underlying beta vs SPX
- `σ_index` = index volatility (choose one):
  - Daily % vol from VIX: `VIX / sqrt(252)`
  - Daily dollar vol from SPX ATR: `ATR_SPX_dollars`

**Examples**:

1. **High-beta tech in calm market**:
   - β = 2.0, VIX = 15, k = 1.0
   - σ_index = 0.15 / sqrt(252) ≈ 0.0095 (0.95% daily)
   - M_stock = 1.0 × 2.0 × 0.0095 = 0.019 (1.9% of underlying)

2. **Low-beta defensive in volatile market**:
   - β = 0.6, VIX = 35, k = 1.0
   - σ_index = 0.35 / sqrt(252) ≈ 0.022 (2.2% daily)
   - M_stock = 1.0 × 0.6 × 0.022 = 0.0132 (1.32% of underlying)

#### Adjustment for Days-to-Expiry

Options with short DTE have higher gamma (acceleration). Widen stops:

```python
def adjust_for_expiry(base_trail_pct: float, days_to_expiry: int) -> float:
    """
    Wider stops for short-dated, gamma-heavy options.
    """
    if days_to_expiry > 30:
        return base_trail_pct
    elif days_to_expiry >= 7:
        return base_trail_pct * 1.5      # 50% wider
    else:
        return base_trail_pct * 2.0      # 100% wider (very short)
```

### 2. Underlying Stop to Option Stop (Delta Conversion)

Once underlying stop is set, convert to expected option price at that level:

```
P_opt_at_stop ≈ P_opt_now - Δ × (S_now - S_stop)
```

More precisely, use the model:

```python
def compute_option_price_at_stop(
    current_price: float,
    current_delta: float,
    current_iv: float,
    underlying_stop: float,
    days_to_expiry: int,
    strike: float,
    right: str,
    rate: float = 0.05
) -> float:
    """
    Compute fair option value at underlying stop level.
    
    Simple approach: use delta.
    Advanced approach: call Black-Scholes with (S_stop, IV, DTE, strike).
    """
    # Simple delta approximation
    underlying_move = underlying_stop - underlying_now
    option_move = current_delta * underlying_move
    theo_price = current_price + option_move
    
    return max(theo_price, 0.01)  # Option value is always > 0
```

### 3. Theoretical Price as Anchor for Execution

Use Black-Scholes (or IB's model) to bound execution prices:

```python
def black_scholes_call(
    S: float,      # Underlying price
    K: float,      # Strike
    T: float,      # Time to expiry (years)
    r: float,      # Risk-free rate
    sigma: float   # IV (annualized)
) -> float:
    """Black-Scholes call price."""
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    call = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    call = black_scholes_call(S, K, T, r, sigma)
    put = call - S + K*math.exp(-r*T)
    return put
```

### 4. Smart Limit Price Computation

When exiting, compute a limit that respects bid/ask but doesn't overshoot:

```python
def compute_smart_limit_price(
    current_bid: float,
    current_ask: float,
    theoretical_price: float,
    allowed_slippage_pct: float = 0.03,  # 3% below theo
    spread_participation: float = 0.5
) -> float:
    """
    For a SELL order, set limit between bid and mid.
    
    Never below: bid
    Never above: theoretical price
    Default: somewhere in middle, respecting spread
    """
    mid = (current_bid + current_ask) / 2
    
    # Compute theoretical with slippage allowance
    theo_min = theoretical_price * (1 - allowed_slippage_pct)
    
    # Spread participation: how deep into bid-ask to go
    spread = current_ask - current_bid
    limit = current_bid + spread * spread_participation
    
    # Take the maximum of: bid + spread_part, or theo_min
    # But cap at theoretical price
    limit = max(limit, theo_min)
    limit = min(limit, theoretical_price)
    
    return limit
```

---

## DETAILED CLASS SPECIFICATIONS

### Class 1: OptionPosition

Complete specification for state container:

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

class PositionStatus(Enum):
    TRACKING = "tracking"          # Normal monitoring
    EXIT_TRIGGERED = "exit_triggered"  # Underlying hit stop
    EXIT_ORDER_PLACED = "exit_order_placed"
    EXIT_FILLED = "exit_filled"
    CLOSED = "closed"

@dataclass
class OptionPosition:
    """Represents one open option position."""
    
    # ===== Identity =====
    conid: int
    symbol: str
    sectype: str = "OPT"
    exchange: str = "SMART"
    currency: str = "USD"
    
    # ===== Contract Specification =====
    expiry: str                    # YYYYMMDD
    strike: float
    right: str                     # "C" or "P"
    
    # ===== Entry Information =====
    quantity: int
    avg_entry_price: float         # $ per contract
    entry_time: datetime = field(default_factory=datetime.now)
    
    # ===== Underlying Reference =====
    underlying_symbol: str = ""    # e.g., "SPY" for SPY options
    underlying_beta: float = 1.0   # Default to market if unknown
    underlying_entry_price: float = 0.0
    
    # ===== Current Market Data =====
    current_bid: Optional[float] = None
    current_ask: Optional[float] = None
    current_last: Optional[float] = None
    last_update: Optional[datetime] = None
    
    # ===== Greeks (from tick type 13) =====
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    implied_vol: Optional[float] = None
    model_price: Optional[float] = None
    underlying_price: Optional[float] = None
    
    # ===== Stop-Loss State =====
    underlying_high: float = field(default_factory=lambda: 0.0)
    underlying_stop_level: Optional[float] = None
    trail_distance_dollars: Optional[float] = None
    trail_distance_pct: Optional[float] = None
    
    # ===== Order State =====
    stop_order_id: Optional[int] = None
    stop_order_price: Optional[float] = None
    stop_order_time: Optional[datetime] = None
    
    # ===== Exit State =====
    status: PositionStatus = PositionStatus.TRACKING
    exit_triggered: bool = False
    exit_triggered_time: Optional[datetime] = None
    exit_order_id: Optional[int] = None
    exit_limit_price: Optional[float] = None
    exit_reprices: int = 0
    
    # ===== P&L =====
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_price: Optional[float] = None
    closed_time: Optional[datetime] = None
    
    # ===== Helpers =====
    def days_to_expiry(self) -> int:
        """Days until option expires."""
        exp_date = datetime.strptime(self.expiry, "%Y%m%d")
        return (exp_date - datetime.now()).days
    
    def is_deep_itm(self) -> bool:
        """Is option deep in the money?"""
        if self.underlying_price is None:
            return False
        if self.right == "C":
            return self.underlying_price > self.strike * 1.10
        else:
            return self.underlying_price < self.strike * 0.90
    
    def update_high_since_entry(self) -> None:
        """Track highest underlying price."""
        if self.underlying_price is not None:
            self.underlying_high = max(
                self.underlying_high or self.underlying_entry_price,
                self.underlying_price
            )
    
    def __str__(self) -> str:
        right_str = "CALL" if self.right == "C" else "PUT"
        dte = self.days_to_expiry()
        bid_ask = f"${self.current_bid:.2f}/${self.current_ask:.2f}" if self.current_bid else "N/A"
        return f"{self.symbol} {self.expiry} {self.strike} {right_str} x{self.quantity} | bid/ask: {bid_ask} | DTE: {dte}"
```

### Class 2: VolatilityTracker

Maintains index volatility index:

```python
@dataclass
class VolatilityTracker:
    """Tracks SPX volatility via VIX or ATR."""
    
    vix_level: Optional[float] = None
    vix_update_time: Optional[datetime] = None
    
    atr_14_dollars: Optional[float] = None
    atr_update_time: Optional[datetime] = None
    
    spx_price: Optional[float] = None
    
    def get_daily_vol_pct(self) -> float:
        """Return daily volatility as decimal."""
        if self.vix_level is not None:
            # VIX to daily %
            return self.vix_level / 100.0 / math.sqrt(252)
        elif self.atr_14_dollars is not None and self.spx_price is not None:
            # ATR to %
            return self.atr_14_dollars / self.spx_price
        else:
            return 0.01  # 1% fallback
    
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
```

### Class 3: StopCalculator

Centralized stop distance calculation:

```python
@dataclass
class StopCalculator:
    """Computes risk-sized stops."""
    
    k_aggression: float = 1.0       # Aggression factor
    min_trail_pct: float = 0.04     # Don't go below 4%
    max_trail_pct: float = 0.40     # Don't go above 40%
    
    def compute_underlying_stop(
        self,
        entry_price: float,
        beta: float,
        index_vol_pct: float,
        days_to_expiry: int
    ) -> float:
        """
        Compute underlying stop level.
        
        Returns: stop price in dollars.
        """
        # Base trail: k * beta * vol
        base_trail = self.k_aggression * beta * index_vol_pct
        
        # Adjust for short DTE
        if days_to_expiry < 7:
            base_trail *= 2.0
        elif days_to_expiry < 30:
            base_trail *= 1.5
        
        # Clamp
        trail = max(self.min_trail_pct, min(base_trail, self.max_trail_pct))
        
        # Convert to dollar move
        stop_distance = entry_price * trail
        stop_level = entry_price - stop_distance
        
        return max(stop_level, 0.01)  # Don't go negative
    
    def compute_option_trail_pct(
        self,
        underlying_stop_distance_pct: float,
        delta: float
    ) -> float:
        """
        Given the underlying trail %, convert to option trail %.
        
        Wider stops for options with lower delta (more leverage).
        """
        # For low-delta options, a 1% underlying move = big % option move
        # So keep the dollar stop but express as %.
        # For calls: delta ≈ 0.3 means 1% underlying = 3% option
        # So option trail % = underlying trail % / delta
        
        if delta < 0.05:
            # Very far OTM; use minimum
            return self.min_trail_pct
        
        option_trail = underlying_stop_distance_pct / max(delta, 0.1)
        return max(self.min_trail_pct, min(option_trail, self.max_trail_pct))
```

---

## API INTEGRATION DETAILS

### 1. Connection Lifecycle

```python
class VolatilityAwareStopManager(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        # State
        self.positions: Dict[int, OptionPosition] = {}  # conid -> position
        self.next_req_id = 1000
        self.vol_tracker = VolatilityTracker()
        self.stop_calc = StopCalculator()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Monitoring
        self.monitoring_active = False
        self.market_open = False
    
    def nextValidId(self, orderId: int) -> None:
        """Callback: connection established."""
        logger.info(f"Connected. Next order ID: {orderId}")
        self.next_req_id = orderId
    
    def connect_and_start(self, host: str, port: int, client_id: int) -> bool:
        """
        Connect to TWS/IB Gateway and start monitoring.
        
        Args:
            host: "127.0.0.1"
            port: 7497 (TWS) or 4002 (IB Gateway)
            client_id: unique ID for this connection
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to {host}:{port}...")
            self.connect(host, port, client_id)
            
            # Start message loop in background thread
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            import time
            time.sleep(2)
            
            if not self.isConnected():
                logger.error("Connection failed")
                return False
            
            logger.info("Connected successfully!")
            return True
        
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            return False
```

### 2. Position Loading

```python
def load_positions(self) -> None:
    """Request all positions from account."""
    logger.info("Requesting positions...")
    self.reqPositions()  # Triggers position() callbacks

def position(
    self,
    account: str,
    contract: Contract,
    quantity: Decimal,
    avgCost: float
) -> None:
    """
    Callback: receive one position.
    Called multiple times, once per position.
    """
    # Only track options
    if contract.secType != "OPT":
        return
    
    if quantity == 0:
        return
    
    with self.lock:
        pos = OptionPosition(
            conid=contract.conId,
            symbol=contract.symbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=contract.strike,
            right=contract.right,
            quantity=int(quantity),
            avg_entry_price=float(avgCost),
            underlying_symbol=contract.symbol,  # Simplified; could parse
        )
        
        self.positions[contract.conId] = pos
        logger.info(f"Loaded position: {pos}")

def positionEnd(self) -> None:
    """Callback: all positions received."""
    with self.lock:
        num = len(self.positions)
    logger.info(f"Position loading complete. Found {num} option positions")
    self._subscribe_to_market_data()
    self._fetch_beta_data()
    self._subscribe_to_index_vol()
    self._start_monitoring_loop()
```

### 3. Market Data Subscription

```python
def _subscribe_to_market_data(self) -> None:
    """Subscribe to Greeks and bid/ask for all positions."""
    with self.lock:
        for conid, pos in self.positions.items():
            self._subscribe_one_option(conid, pos)

def _subscribe_one_option(self, conid: int, pos: OptionPosition) -> None:
    """Subscribe to market data and Greeks for one option."""
    contract = Contract()
    contract.conId = conid
    contract.symbol = pos.symbol
    contract.secType = "OPT"
    contract.exchange = pos.exchange
    contract.currency = pos.currency
    contract.lastTradeDateOrContractMonth = pos.expiry
    contract.strike = pos.strike
    contract.right = pos.right
    
    req_id = conid  # Use conid as request ID for easy mapping
    
    # Subscribe to market data
    # Generic tick list empty = default (bid, ask, Greeks, etc.)
    self.reqMktData(req_id, contract, "", False, False, [])
    
    logger.info(f"Subscribed to market data: {pos}")

def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib) -> None:
    """
    Callback: price update received.
    
    Tick types:
    1 = BID
    2 = ASK
    4 = LAST
    """
    with self.lock:
        pos = self.positions.get(reqId)
        if not pos:
            return
        
        if tickType == 1:  # BID
            pos.current_bid = price
            if pos.underlying_high == 0:
                pos.underlying_high = pos.underlying_entry_price or 0
        
        elif tickType == 2:  # ASK
            pos.current_ask = price
        
        elif tickType == 4:  # LAST
            pos.current_last = price
        
        pos.last_update = datetime.now()

def tickOptionComputation(
    self,
    reqId: int,
    tickType: int,
    tickAttrib: TickAttrib,
    impliedVol: float,
    delta: float,
    optPrice: float,
    pvDividend: float,
    gamma: float,
    vega: float,
    theta: float,
    undPrice: float
) -> None:
    """
    Callback: Greek values received.
    
    tickType:
    10 = Bid option computation
    11 = Ask option computation
    12 = Last option computation
    13 = Model option computation (most stable, use this)
    """
    if tickType != 13:  # Only use model (tick 13)
        return
    
    with self.lock:
        pos = self.positions.get(reqId)
        if not pos:
            return
        
        pos.delta = delta
        pos.gamma = gamma
        pos.vega = vega
        pos.theta = theta
        pos.implied_vol = impliedVol
        pos.model_price = optPrice
        pos.underlying_price = undPrice
        
        # Update underlying high for trailing logic
        if undPrice > (pos.underlying_high or 0):
            pos.underlying_high = undPrice
```

### 4. Index Volatility Subscription

```python
def _subscribe_to_index_vol(self) -> None:
    """Subscribe to VIX for volatility tracking."""
    vix_contract = Contract()
    vix_contract.symbol = "VIX"
    vix_contract.secType = "IND"  # Index
    vix_contract.exchange = "CBOE"
    vix_contract.currency = "USD"
    
    req_id = 9999  # Reserve ID for VIX
    self.reqMktData(req_id, vix_contract, "", False, False, [])
    logger.info("Subscribed to VIX for volatility tracking")

def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib) -> None:
    """Handle VIX updates."""
    if reqId == 9999 and tickType == 4:  # LAST price for VIX
        self.vol_tracker.update_vix(price)
        logger.debug(f"VIX updated: {price:.2f}")
```

### 5. Beta Data Fetching

```python
def _fetch_beta_data(self) -> None:
    """Fetch beta for all unique underlyings."""
    underlyings = set()
    with self.lock:
        for pos in self.positions.values():
            underlyings.add(pos.underlying_symbol)
    
    for symbol in underlyings:
        beta = self._get_or_fetch_beta(symbol)
        with self.lock:
            for pos in self.positions.values():
                if pos.underlying_symbol == symbol:
                    pos.underlying_beta = beta
        
        logger.info(f"Beta for {symbol}: {beta:.2f}")

def _get_or_fetch_beta(self, symbol: str) -> float:
    """Get beta from cache or fetch from API."""
    # First, try lookup table
    if symbol in BETA_TABLE:
        return BETA_TABLE[symbol]
    
    # Could add API fetch here (reqFundamentalData)
    # For now, default to 1.0
    return 1.0
```

---

## STOP-LOSS LOGIC IMPLEMENTATION

### 1. Trailing Stop on Underlying

```python
def update_underlying_stops(self) -> None:
    """
    For each position, compute and track underlying-level stops.
    This is the core trigger mechanism.
    """
    with self.lock:
        for conid, pos in self.positions.items():
            if pos.status in [PositionStatus.CLOSED, PositionStatus.EXIT_FILLED]:
                continue
            
            if pos.underlying_price is None:
                continue  # No data yet
            
            # Update high for trailing
            pos.update_high_since_entry()
            
            # Compute stop level
            index_vol = self.vol_tracker.get_daily_vol_pct()
            dte = pos.days_to_expiry()
            
            stop_level = self.stop_calc.compute_underlying_stop(
                entry_price=pos.underlying_entry_price,
                beta=pos.underlying_beta,
                index_vol_pct=index_vol,
                days_to_expiry=dte
            )
            
            # Trailing: allow stop to move up, but not down
            if pos.underlying_stop_level is None:
                pos.underlying_stop_level = stop_level
            else:
                pos.underlying_stop_level = max(pos.underlying_stop_level, stop_level)
            
            # Compute distance for logging
            pos.trail_distance_dollars = pos.underlying_entry_price - pos.underlying_stop_level
            pos.trail_distance_pct = pos.trail_distance_dollars / pos.underlying_entry_price
            
            # Check trigger condition
            if pos.underlying_price <= pos.underlying_stop_level and not pos.exit_triggered:
                logger.warning(
                    f"STOP TRIGGERED: {pos.symbol} underlying {pos.underlying_price:.2f} "
                    f"<= stop {pos.underlying_stop_level:.2f}"
                )
                pos.exit_triggered = True
                pos.exit_triggered_time = datetime.now()
                pos.status = PositionStatus.EXIT_TRIGGERED

def handle_exit_triggered_positions(self) -> None:
    """
    For positions where underlying hit stop, place smart exit orders.
    """
    with self.lock:
        triggered = [
            (conid, pos) for conid, pos in self.positions.items()
            if pos.status == PositionStatus.EXIT_TRIGGERED
        ]
    
    for conid, pos in triggered:
        self._execute_smart_exit(conid, pos)

def _execute_smart_exit(self, conid: int, pos: OptionPosition) -> None:
    """
    Place a smart limit order to exit the position.
    Uses bid/ask/theoretical price to set limit.
    """
    if pos.current_bid is None or pos.current_ask is None:
        logger.warning(f"{pos.symbol}: No bid/ask available yet, skipping exit")
        return
    
    # Compute theoretical price at stop level
    theo = self._compute_theoretical_at_stop(pos)
    
    # Compute smart limit price
    limit = self._compute_smart_limit_price(
        bid=pos.current_bid,
        ask=pos.current_ask,
        theoretical=theo,
        spread_participation=0.5
    )
    
    # Place order
    order = Order()
    order.orderId = self.next_req_id
    order.clientId = 0
    order.action = "SELL"  # Always sell to close
    order.orderType = "LMT"  # Limit order
    order.totalQuantity = pos.quantity
    order.lmtPrice = limit
    order.tif = "GTC"  # Good till canceled
    order.transmit = True
    
    contract = self._build_contract_from_position(pos)
    
    self.placeOrder(order.orderId, contract, order)
    
    with self.lock:
        pos.exit_order_id = order.orderId
        pos.exit_limit_price = limit
        pos.status = PositionStatus.EXIT_ORDER_PLACED
    
    self.next_req_id += 1
    
    logger.info(
        f"PLACED EXIT ORDER: {pos.symbol} SELL {pos.quantity} @ ${limit:.2f} "
        f"(bid ${pos.current_bid:.2f}, ask ${pos.current_ask:.2f}, theo ${theo:.2f})"
    )

def _compute_theoretical_at_stop(self, pos: OptionPosition) -> float:
    """
    Compute theoretical option price at the underlying stop level.
    
    Use Black-Scholes with:
    - S = underlying_stop_level
    - K = strike
    - T = days_to_expiry / 365
    - σ = implied_vol
    - r = 0.05 (risk-free rate, simplified)
    """
    if pos.implied_vol is None or pos.underlying_stop_level is None:
        # Fallback to delta approximation
        if pos.current_bid and pos.delta:
            move = (pos.underlying_price or 0) - pos.underlying_stop_level
            return max(pos.current_bid - (pos.delta * move), 0.01)
        return pos.current_bid or 0.01
    
    T = pos.days_to_expiry() / 365.0
    if T <= 0:
        return 0.01  # Expired
    
    if pos.right == "C":
        theo = black_scholes_call(
            S=pos.underlying_stop_level,
            K=pos.strike,
            T=T,
            r=0.05,
            sigma=pos.implied_vol
        )
    else:
        theo = black_scholes_put(
            S=pos.underlying_stop_level,
            K=pos.strike,
            T=T,
            r=0.05,
            sigma=pos.implied_vol
        )
    
    return max(theo, 0.01)

def _compute_smart_limit_price(
    self,
    bid: float,
    ask: float,
    theoretical: float,
    spread_participation: float = 0.5
) -> float:
    """
    Compute execution limit price for a sell order.
    
    Constraints:
    - Never below bid
    - Don't expect more than theoretical
    - Account for spread width
    """
    spread = ask - bid
    mid = bid + spread / 2
    
    # How deep into the spread are we willing to go?
    limit = bid + spread * spread_participation
    
    # But don't exceed theoretical
    limit = min(limit, theoretical)
    
    # Always at least at bid
    limit = max(limit, bid)
    
    return limit
```

---

## ORDER EXECUTION STRATEGY

### 1. Smart Limit Ordering with Re-pricing

```python
def orderStatus(
    self,
    orderId: int,
    status: str,
    filled: Decimal,
    remaining: Decimal,
    avgFillPrice: float,
    ...
) -> None:
    """Callback: order status update."""
    if status == "Filled":
        with self.lock:
            for conid, pos in self.positions.items():
                if pos.exit_order_id == orderId:
                    pos.status = PositionStatus.EXIT_FILLED
                    pos.closed_price = avgFillPrice
                    pos.closed_time = datetime.now()
                    pos.realized_pnl = (avgFillPrice - pos.avg_entry_price) * pos.quantity * 100
                    logger.warning(
                        f"*** POSITION CLOSED *** {pos.symbol} filled at ${avgFillPrice:.2f}, "
                        f"P&L: ${pos.realized_pnl:.2f}"
                    )
    
    elif status == "Cancelled":
        logger.info(f"Order {orderId} cancelled")

def reprice_unfilled_exit_orders(self) -> None:
    """
    If exit order hasn't filled and underlying still below stop,
    walk the limit price down (adaptive re-pricing).
    """
    with self.lock:
        unfilled = [
            (conid, pos) for conid, pos in self.positions.items()
            if pos.status == PositionStatus.EXIT_ORDER_PLACED
            and pos.exit_order_id is not None
        ]
    
    for conid, pos in unfilled:
        # Check: has enough time passed?
        if pos.exit_reprices >= 5:  # Max 5 reprices
            logger.info(f"{pos.symbol}: Max reprices reached, stopping")
            continue
        
        time_since_order = (datetime.now() - (pos.exit_triggered_time or datetime.now())).total_seconds()
        if time_since_order < 5:  # Wait at least 5 seconds
            continue
        
        # Check: is underlying still below stop?
        if pos.underlying_price is None or pos.underlying_price > pos.underlying_stop_level:
            logger.info(f"{pos.symbol}: Underlying recovered, cancelling exit order")
            self.cancelOrder(pos.exit_order_id)
            continue
        
        # Recompute limit price
        theo = self._compute_theoretical_at_stop(pos)
        new_limit = self._compute_smart_limit_price(
            bid=pos.current_bid,
            ask=pos.current_ask,
            theoretical=theo,
            spread_participation=0.7  # Go deeper into spread
        )
        
        # Only reprice if lower (walking down)
        if new_limit < (pos.exit_limit_price or float('inf')):
            # Cancel old order
            self.cancelOrder(pos.exit_order_id)
            
            # Place new order
            order = Order()
            order.orderId = self.next_req_id
            order.action = "SELL"
            order.orderType = "LMT"
            order.totalQuantity = pos.quantity
            order.lmtPrice = new_limit
            order.tif = "GTC"
            order.transmit = True
            
            contract = self._build_contract_from_position(pos)
            self.placeOrder(order.orderId, contract, order)
            
            with self.lock:
                pos.exit_order_id = order.orderId
                pos.exit_limit_price = new_limit
                pos.exit_reprices += 1
            
            self.next_req_id += 1
            
            logger.info(
                f"REPRICED EXIT: {pos.symbol} reprices #{pos.exit_reprices} "
                f"new limit ${new_limit:.2f} (was ${pos.exit_limit_price:.2f})"
            )
```

---

## ERROR HANDLING & RECOVERY

### 1. Connection Loss Handling

```python
def connectionClosed(self) -> None:
    """Callback: connection lost."""
    logger.error("Connection to TWS closed!")
    self.monitoring_active = False
    self._attempt_reconnect()

def _attempt_reconnect(self, max_attempts: int = 5, delay: float = 10.0) -> bool:
    """Attempt to reconnect after connection loss."""
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Reconnect attempt {attempt}/{max_attempts} in {delay}s...")
        time.sleep(delay)
        
        try:
            if self.connect_and_start("127.0.0.1", 7497, 100):
                logger.info("Reconnected successfully!")
                return True
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
    
    logger.error(f"Failed to reconnect after {max_attempts} attempts")
    return False

def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = None) -> None:
    """Callback: error received."""
    # Ignore non-fatal errors
    if errorCode in [2104, 2158]:  # Market data subscriptions
        logger.debug(f"Market data: {errorString}")
        return
    
    logger.error(f"Error {errorCode}: {errorString}")
    
    # Fatal errors: stop monitoring
    if errorCode in [10061]:  # Can't connect
        self.monitoring_active = False
        self._attempt_reconnect()
```

### 2. Data Validation

```python
def _validate_position_ready(self, pos: OptionPosition) -> bool:
    """Check if position has all required data before trading."""
    checks = [
        (pos.current_bid is not None, "bid price"),
        (pos.current_ask is not None, "ask price"),
        (pos.delta is not None, "delta"),
        (pos.implied_vol is not None, "implied volatility"),
        (pos.underlying_price is not None, "underlying price"),
        (pos.underlying_stop_level is not None, "underlying stop"),
    ]
    
    for check, name in checks:
        if not check:
            logger.debug(f"{pos.symbol}: Waiting for {name}")
            return False
    
    return True
```

---

## DEPLOYMENT & CONFIGURATION

### config.py (Enhanced)

```python
# ===== Connection =====
IB_HOST = "127.0.0.1"
IB_PORT = 7497                  # 7497 for TWS, 4002 for IB Gateway
IB_CLIENT_ID = 100

# ===== Strategy Parameters =====
STRATEGY_K_AGGRESSION = 1.0     # Beta × Vol scaling factor
STRATEGY_MIN_TRAIL_PCT = 0.04   # 4% minimum
STRATEGY_MAX_TRAIL_PCT = 0.40   # 40% maximum

# ===== Volatility Settings =====
USE_VIX = True                  # Use VIX vs ATR
VIX_UPDATE_INTERVAL = 30        # seconds

# ===== Exit Execution =====
EXIT_SPREAD_PARTICIPATION = 0.5 # 50% into bid-ask spread
EXIT_ALLOWED_SLIPPAGE_PCT = 0.03  # 3% below theoretical
EXIT_MAX_REPRICES = 5
EXIT_REPRICE_INTERVAL = 10      # seconds between reprices

# ===== Stop-Loss Adjustment =====
ADJUST_FOR_DTE = True
DTE_30_PLUS_MULTIPLIER = 1.0    # Normal
DTE_7_30_MULTIPLIER = 1.5       # 50% wider
DTE_UNDER_7_MULTIPLIER = 2.0    # 100% wider

# ===== Market Hours =====
MARKET_OPEN_TIME = "09:30"      # ET
MARKET_CLOSE_TIME = "16:00"     # ET

# ===== Logging & Monitoring =====
LOG_LEVEL = "INFO"              # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "volatility_stops.log"

# ===== Paper Trading =====
PAPER_TRADING = True            # Set to False for live

# ===== Filtering =====
ALLOWED_SYMBOLS = []            # [] = all
EXCLUDED_SYMBOLS = []
MIN_POSITION_QUANTITY = 1
```

### Running the Application

```bash
#!/bin/bash
# run_volatility_stops.sh

# Ensure TWS is running
echo "Ensure TWS is running on port 7497..."
sleep 1

# Install dependencies
pip install ibapi --upgrade

# Run application
python advanced_volatility_stops.py

# Monitor logs in another terminal:
# tail -f volatility_stops.log
```

---

## TESTING PROTOCOL

### Phase 1: Unit Testing

```python
# test_stop_calculator.py
import pytest
from stop_calculator import StopCalculator

def test_underlying_stop_basic():
    """Test basic stop computation."""
    calc = StopCalculator(k_aggression=1.0)
    
    # Entry at 100, beta 1.0, vol 1% → 1% trail
    stop = calc.compute_underlying_stop(
        entry_price=100.0,
        beta=1.0,
        index_vol_pct=0.01,
        days_to_expiry=30
    )
    
    assert stop == 99.0  # 1% below entry

def test_short_dte_multiplier():
    """Test DTE multiplier."""
    calc = StopCalculator(k_aggression=1.0)
    
    # Same inputs, but 5 DTE (short)
    stop = calc.compute_underlying_stop(
        entry_price=100.0,
        beta=1.0,
        index_vol_pct=0.01,
        days_to_expiry=5
    )
    
    # Should be 2x wider: 2%
    assert stop == 98.0  # 2% below entry

def test_high_beta_high_vol():
    """Test high risk scenario."""
    calc = StopCalculator(k_aggression=1.0)
    
    # High beta, high vol
    stop = calc.compute_underlying_stop(
        entry_price=100.0,
        beta=2.0,
        index_vol_pct=0.03,
        days_to_expiry=30
    )
    
    # 2.0 * 0.03 = 6%
    assert stop == 94.0

def test_clamping():
    """Test min/max clamping."""
    calc = StopCalculator(min_trail_pct=0.04, max_trail_pct=0.40)
    
    # Compute extreme trail
    stop = calc.compute_underlying_stop(
        entry_price=100.0,
        beta=0.1,  # Very low beta
        index_vol_pct=0.01,
        days_to_expiry=30
    )
    
    # Should be clamped to min 4%
    assert stop == 96.0  # 4% below
```

### Phase 2: Paper Trading Testing (1 Week)

```
Monday (Day 1):
- Start application at 9:30 AM ET
- Manually buy 1 SPY option (far OTM, e.g., 550 call if SPY at 500)
- Watch logs for: position loading, Greeks arrival, stop calculation
- Verify: stop is placed at reasonable level (≈10-15% below bid)
- Monitor for 2 hours minimum

Tuesday-Thursday (Days 2-4):
- Let bot run for full days
- Check daily P&L in Portfolio
- Verify: stops move up when bid rises, stay fixed when falls
- Monitor: any API errors, reconnections

Friday (Day 5):
- Add 2-3 more option contracts in different underlyings
- Test multi-position management
- Verify: each gets appropriate risk-sized stop
- Check logs for any issues

Week 2:
- Stress test: add 10+ positions
- Monitor CPU/memory usage
- Check: does bot handle order fills correctly?
- Manually trigger a stop by moving market simulator (if available)
- Verify: exit order is placed, repriced, and fills
```

### Phase 3: Paper Exit Testing

```
Create a test scenario:
1. Buy 1 SPY 550 call at any price
2. Let bot place initial stop
3. Manually move market price down toward stop in TWS simulator
4. Watch logs for: "STOP TRIGGERED", "PLACED EXIT ORDER"
5. Continue moving price down
6. Verify: order reprices as needed
7. Verify: order fills at reasonable price
8. Check logs for: execution price vs theoretical

Expected outcome:
- Stop triggered when underlying hits limit
- Exit order placed at smart limit price (between bid and theoretical)
- Reprice happens at expected intervals
- Fill price is within expected slippage tolerance
```

---

## MONITORING & OBSERVABILITY

### Comprehensive Logging

```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler (rotating)
file_handler = RotatingFileHandler(
    "volatility_stops.log",
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

### Key Log Events to Monitor

```
# Startup
"Connected. Next order ID: 1001"
"Loaded position: SPY 2025-02-21 550.0 CALL x5"
"Position loading complete. Found 5 option positions"
"Subscribed to market data: SPY ..."

# Data arrival
"VIX updated: 18.50"
"Greeks for SPY: Delta=0.45, Gamma=0.02, Vega=5.2, IV=0.25"

# Stop calculation
"[STOP CALC] SPY: Entry=$500, Beta=1.2, VolIdx=1.5%, Stop=$480.0 (4.0% trail)"
"[STOP CALC] SPY underlying high=$505, stop remains=$480"

# Trigger
"[ALERT] SPY underlying $479.5 <= stop $480 - TRIGGERED"

# Execution
"[EXEC] EXIT ORDER placed: SPY SELL 5 @ $9.50 lim (bid $9.40, ask $10.00, theo $9.75)"
"[EXEC] EXIT ORDER repriced #1: new limit $9.35 (was $9.50)"
"[EXEC] POSITION CLOSED: SPY filled at $9.48, P&L $+245"

# Errors
"[ERROR] Connection lost - attempting reconnect"
"[WARN] No bid price available for SPY yet"
```

### Performance Metrics

```python
class PerformanceMetrics:
    def __init__(self):
        self.positions_tracked = 0
        self.stops_placed = 0
        self.stops_triggered = 0
        self.positions_closed = 0
        self.avg_slippage_pct = 0.0
        self.api_errors = 0
        self.reconnections = 0
    
    def report_daily(self) -> str:
        """Generate daily summary."""
        return f"""
        === DAILY SUMMARY ===
        Positions Tracked: {self.positions_tracked}
        Stops Placed: {self.stops_placed}
        Stops Triggered: {self.stops_triggered}
        Positions Closed: {self.positions_closed}
        Avg Slippage: {self.avg_slippage_pct:.2%}
        API Errors: {self.api_errors}
        Reconnections: {self.reconnections}
        """
```

---

## CODE EXAMPLES

### Example 1: Computing a stop for a high-beta tech stock

```python
# Inputs
entry_underlying_price = 180.0  # TSLA entry
beta = 2.1                       # TSLA vs SPX
vix = 22.0                       # Current VIX
k_aggression = 1.0

# Compute index vol from VIX
index_vol_pct = vix / 100.0 / math.sqrt(252)  # ≈ 0.0139 (1.39% daily)

# Stop distance
stop_distance = k_aggression * beta * index_vol_pct
# = 1.0 × 2.1 × 0.0139 = 0.0292 (2.92%)

# Dollar move
stop_distance_dollars = entry_underlying_price * stop_distance
# = 180 × 0.0292 = $5.26

# Stop level
underlying_stop = entry_underlying_price - stop_distance_dollars
# = 180 - 5.26 = $174.74

# Result: SELL call if TSLA drops below $174.74 (2.92% below entry)
```

### Example 2: Computing exit limit price

```python
# Market data
current_bid = 3.50
current_ask = 3.80
current_spread = 0.30

# Theoretical at stop level (computed via Black-Scholes)
theoretical_price = 2.95

# Parameters
spread_participation = 0.5
allowed_slippage = 0.03  # 3%

# Compute limit
limit = current_bid + current_spread * spread_participation
# = 3.50 + 0.30 * 0.5 = 3.65

# Apply slippage floor
theo_min = theoretical_price * (1 - allowed_slippage)
# = 2.95 * 0.97 = 2.87

# Take max of spread-based and theo-based
limit = max(limit, theo_min)
# = max(3.65, 2.87) = 3.65

# Cap at theoretical
limit = min(limit, theoretical_price)
# = min(3.65, 2.95) = 2.95

# Result: sell limit order at $2.95
```

### Example 3: Adaptive re-pricing sequence

```
Time 0: SPY underlying = $500, bid = $10.00, ask = $10.50
  → Order placed: SELL 1 @ $10.15 (mid)

Time 10s: Underlying drops to $498
  → Bid drops to $9.50, ask to $10.00
  → Reprice #1: new limit = $9.65

Time 20s: Underlying drops to $496
  → Bid drops to $8.90, ask to $9.40
  → Reprice #2: new limit = $9.05

Time 30s: Underlying rallies to $501
  → Stop condition no longer met, cancel order

Result: No forced loss, adjusted for market reality
```

---

## DEPLOYMENT INSTRUCTIONS

### 1. Local Development

```bash
# Clone/setup
git clone <repo>
cd volatility_options_stops
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
python advanced_volatility_stops.py
```

### 2. Cloud Deployment (AWS EC2)

```bash
# Launch t2.micro (free tier)
# SSH in
ssh -i key.pem ubuntu@instance-ip

# Install Python
sudo apt update && sudo apt install python3-pip -y
pip3 install ibapi

# Upload code
scp -i key.pem -r . ubuntu@instance:/home/ubuntu/volatility_stops
cd /home/ubuntu/volatility_stops

# Create systemd service
sudo tee /etc/systemd/system/volatility-stops.service > /dev/null <<EOF
[Unit]
Description=Volatility-Aware Options Stop Loss Manager
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/volatility_stops
ExecStart=/usr/bin/python3 advanced_volatility_stops.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable volatility-stops
sudo systemctl start volatility-stops
sudo systemctl status volatility-stops

# Monitor
sudo journalctl -f -u volatility-stops
```

### 3. Configuration for Different Environments

```python
# config_paper.py (Paper trading)
PAPER_TRADING = True
STRATEGY_K_AGGRESSION = 1.0
STRATEGY_MIN_TRAIL_PCT = 0.04
EXIT_SPREAD_PARTICIPATION = 0.5

# config_live_conservative.py (Live, risk-averse)
PAPER_TRADING = False
STRATEGY_K_AGGRESSION = 0.7  # Tighter stops
STRATEGY_MIN_TRAIL_PCT = 0.06
EXIT_SPREAD_PARTICIPATION = 0.3  # Deeper into spread for fills

# config_live_aggressive.py (Live, wider stops)
PAPER_TRADING = False
STRATEGY_K_AGGRESSION = 1.5  # Wider stops
STRATEGY_MIN_TRAIL_PCT = 0.08
EXIT_SPREAD_PARTICIPATION = 0.7
```

---

## PRODUCTION SAFEGUARDS

### Pre-Live Checklist

- [ ] Paper traded successfully for 5+ business days
- [ ] Tested with 10+ concurrent positions
- [ ] Simulated stop trigger and exit fill
- [ ] Verified Greeks arrival during market hours
- [ ] Confirmed beta data accuracy
- [ ] Tested reconnection after simulated network loss
- [ ] Checked log rotation and disk space
- [ ] Documented all parameters in config
- [ ] Have manual override plan (can kill TWS instantly)
- [ ] Have support contact info (IBKR, your broker)

### Kill Switches

```python
def emergency_stop(self) -> None:
    """
    Emergency shutdown.
    - Cancel all pending orders immediately
    - Close connections
    - Exit gracefully
    """
    logger.critical("EMERGENCY STOP INITIATED")
    
    with self.lock:
        for orderId in list(self.stop_orders.keys()):
            self.cancelOrder(orderId)
    
    time.sleep(2)
    self.disconnect()
    logger.critical("Emergency stop complete")
```

---

## SUMMARY

This document provides a complete specification for a **production-grade, volatility-aware options stop-loss system** on Interactive Brokers using Python and the TWS API.

**Key innovations**:
1. **Risk-sized stops**: Beta × Index Volatility (not fixed 10%)
2. **Underlying-driven triggers**: Avoids option quote noise
3. **Smart execution**: Theoretical pricing + bid/ask guardrails
4. **Adaptive re-pricing**: Walks down limit if needed

**Ready for Claude 3.5 Opus code generation**. All specifications are precise, reference actual IBKR API callbacks and tick types (2025 standard), and include error handling, testing, and deployment guidance.

---

**Document prepared for**: Antigravity  
**Target coder**: Claude 3.5 Opus  
**Status**: Production specification, ready for implementation
