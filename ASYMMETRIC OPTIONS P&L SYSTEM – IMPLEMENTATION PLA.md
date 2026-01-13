<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ASYMMETRIC OPTIONS P\&L SYSTEM – IMPLEMENTATION PLAN

## Infinite Upside, Refined Downside Capping

### Target: Claude 3.5 Opus / Antigravity – Python + IBKR TWS API


***

## 1. Executive Summary

Goal: build a Python system on Interactive Brokers that:

- Caps **per‑trade loss** at a pre‑defined dollar amount (e.g. 0.5–1% of equity).
- Lets **profits run without a target** (no cap on upside).
- Uses **refined, volatility‑aware stops** based on beta and index volatility.
- Trails stops **upward only** as underlying rises; never loosens stops.
- Exits with **smart limit orders** that respect bid/ask and theoretical value.
- Tracks P\&L distribution to confirm **right‑skewed (asymmetric)** outcomes.

Key ideas:

- Position size is constrained by **max acceptable loss**, not “capital per trade”.
- Stop distance is computed from **β × σ_index** (beta × index volatility), adjusted for DTE.
- As underlying rises, **stop ratchets up** but never down.
- System is **event‑driven** (IBKR callbacks) with a dedicated risk/stop manager.

***

## 2. Asymmetric Philosophy

Objective: create a P\&L distribution with:

- **Hard loss cap** per position: $-L_{\max}$.
- **Unbounded profit** per position: $+∞$ (no profit targets).
- Many **small, controlled losses**; fewer **large winners**.

Target statistics (over many trades):

- Win rate: 40–60%.
- Average winner: 2–5× average loser.
- Profit factor (gross wins / gross losses): 1.5–3.0+.
- P\&L distribution **positively skewed**.

***

## 3. Loss‑Capping \& Position Sizing

### 3.1 Portfolio‑Level Risk Parameters

Configure once:

```python
PORTFOLIO_SIZE = 100_000          # total account equity
MAX_POSITIONS = 10                # concurrent positions
MAX_DAILY_LOSS = 0.025 * PORTFOLIO_SIZE  # e.g. 2.5% per day cap

MAX_LOSS_PER_POSITION = 0.005 * PORTFOLIO_SIZE  # e.g. 0.5% per position
K_AGGRESSION = 1.0                # stop-width aggression factor
```


### 3.2 Refined Stop Distance (Underlying)

For a position in underlying $S$:

- $S_0$ = underlying entry price
- $\beta$ = beta vs SPX
- $\text{VIX}$ = current VIX level
- $\sigma_{\text{idx}} = \frac{\text{VIX}}{100\sqrt{252}}$ (daily index vol)
- Base stop distance (fraction of $S_0$):

$$
d_{\text{base}} = K_{\text{aggr}} \times \beta \times \sigma_{\text{idx}}
$$

DTE adjustment (gamma‑aware):

```python
def adjust_for_dte(base_pct: float, dte: int) -> float:
    if dte > 30:
        m = 1.0
    elif dte >= 14:
        m = 1.2
    elif dte >= 7:
        m = 1.5
    else:
        m = 2.0
    return base_pct * m
```

Clamp to reasonable band:

```python
STOP_MIN = 0.004   # 0.4% min
STOP_MAX = 0.20    # 20% max
```

Final stop distance and level:

```python
d_stop = clamp(adjust_for_dte(d_base, dte), STOP_MIN, STOP_MAX)
S_stop = S_0 * (1 - d_stop)
```


### 3.3 Position Size from Max Loss Constraint

Given:

- Option premium $C$ (per share).
- Delta $\Delta$.
- Stop distance fraction $d_{\text{stop}}$.
- Max dollar loss for this position $L_{\max}$.

Approximate loss per contract if stop hit:

$$
L_{\text{per\,contract}} \approx C \times 100 \times \Delta \times d_{\text{stop}}
$$

Contracts:

```python
loss_per_contract = premium * 100 * delta * d_stop
contracts = int(MAX_LOSS_PER_POSITION / loss_per_contract)
contracts = max(1, contracts)
```

Result: **if stop hits**, realized loss ≈ $L_{\max}$.

Example (WDC, approx):

- $C = 8.35$, $\Delta ≈ 0.5$, $d_{\text{stop}} ≈ 0.015$, $L_{\max}=500$:

$$
L_{\text{per}} ≈ 8.35 \times 100 \times 0.5 \times 0.015 ≈ \$6.3
$$

$$
N ≈ 500 / 6.3 ≈ 79\ \text{contracts}
$$

***

## 4. Up‑Only Trailing Logic

### 4.1 Concept

- Track **entry price** $S_0$, **stop** $S_{\text{stop}}$, and **high water mark** $S_{\text{high}}$.
- On each new underlying price $S_t$:
    - If $S_t > S_{\text{high}}$:
update $S_{\text{high}} = S_t$ and move stop **upwards**:

$$
S_{\text{stop,new}} = \max\left(S_{\text{stop,old}},\ S_{\text{high}} - S_0 \times d_{\text{stop}}\right)
$$
    - If $S_t \leq S_{\text{stop,current}}$: **trigger exit**.
    - **Never** move stop down.


### 4.2 Pseudocode

```python
def update_trailing_stop(pos, current_underlying: float):
    # pos: AsymmetricPosition
    if current_underlying > pos.underlying_high:
        pos.underlying_high = current_underlying
        candidate = pos.underlying_high - pos.underlying_entry_price * pos.stop_distance_pct
        pos.stop_level_current = max(pos.stop_level_current, candidate)

    if current_underlying <= pos.stop_level_current and not pos.exit_triggered:
        pos.exit_triggered = True
        pos.exit_trigger_time = now()
        return True  # triggered
    return False
```

Effect:

- If trade fails early → small, predefined loss.
- If trade trends strongly → stop ratchets up, **locking in profit** while leaving room.
- No profit target: upside is **unbounded**.

***

## 5. Profit‑Running Rules

- No fixed profit target.
- Exit only on:
    - Stop hit.
    - Near expiry (e.g. DTE ≤ 1) to avoid last‑day gamma blowup.
    - Manual override.

Optional conservative feature: partial scale‑out once unrealized gain exceeds threshold (e.g. +200% of risk), but do not close entire position.

***

## 6. P\&L Asymmetry Metrics

Maintain a simple analytics module to verify the strategy’s intended behavior over time.

```python
@dataclass
class AsymmetryStats:
    closed_pnls: list[float] = field(default_factory=list)

    def add(self, pnl: float):
        self.closed_pnls.append(pnl)

    def win_rate(self) -> float:
        n = len(self.closed_pnls)
        if n == 0:
            return 0.0
        wins = sum(1 for p in self.closed_pnls if p > 0)
        return wins / n

    def avg_winner(self) -> float:
        wins = [p for p in self.closed_pnls if p > 0]
        return sum(wins) / len(wins) if wins else 0.0

    def avg_loser(self) -> float:
        losses = [p for p in self.closed_pnls if p < 0]
        return abs(sum(losses) / len(losses)) if losses else 0.0

    def profit_factor(self) -> float:
        wins = sum(p for p in self.closed_pnls if p > 0)
        losses = abs(sum(p for p in self.closed_pnls if p < 0))
        return wins / losses if losses > 0 else 0.0

    def expected_value(self) -> float:
        wr = self.win_rate()
        aw = self.avg_winner()
        al = self.avg_loser()
        return wr * aw - (1 - wr) * al
```

Daily log:

- Number of trades.
- Win rate.
- Avg winner / loser.
- Profit factor.
- Expected value per trade.

Target: **avg winner ≥ 2 × avg loser**; **profit factor ≥ 1.5**.

***

## 7. System Architecture (High Level)

### 7.1 Components

- **IBKR Connector** (`ibapi.EClient` + `EWrapper` subclass)
    - Handles connect/disconnect, market data, orders.
- **Data Layer**
    - Positions (`reqPositions`).
    - Market data (`reqMktData` for options and underlyings).
    - Greeks (`tickOptionComputation`, tick type 13).
    - VIX / SPX for volatility.
- **Risk \& Stop Manager** (this spec)
    - Computes refined stops and position sizes.
    - Maintains `AsymmetricPosition` objects.
    - Updates trailing stops and triggers exits.
- **Execution Engine**
    - Builds IBKR `Contract` and `Order` objects.
    - Places smart LIMIT exits.
    - Handles re‑pricing if unfilled.
- **Analytics \& Logging**
    - Records realized/unrealized P\&L.
    - Computes asymmetry stats and daily summaries.


### 7.2 Data Structures

```python
@dataclass
class AsymmetricPosition:
    conid: int
    symbol: str
    expiry: str          # "YYYYMMDD"
    strike: float
    right: str           # "C" or "P"

    quantity: int
    avg_entry_price: float  # option price
    capital_allocated: float
    max_loss_for_position: float

    entry_time: datetime
    underlying_entry_price: float
    entry_vix: float
    entry_beta: float

    stop_distance_pct: float
    stop_level_underlying: float   # initial stop
    stop_level_current: float      # trailing stop
    underlying_high: float

    # market data
    current_bid: float | None = None
    current_ask: float | None = None
    underlying_price: float | None = None
    delta: float | None = None
    implied_vol: float | None = None

    # P&L
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    exit_triggered: bool = False
    exit_time: datetime | None = None
    exit_price: float | None = None

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
```

Portfolio container:

```python
@dataclass
class PortfolioState:
    portfolio_size: float
    max_positions: int
    max_loss_per_position: float
    max_daily_loss: float
    k_aggression: float

    positions: dict[int, AsymmetricPosition] = field(default_factory=dict)
    closed_positions: list[AsymmetricPosition] = field(default_factory=list)

    def current_daily_loss(self) -> float:
        return sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl and p.realized_pnl < 0)

    def can_open_new(self) -> bool:
        return (
            len(self.positions) < self.max_positions
            and self.current_daily_loss() > -self.max_daily_loss
        )
```


***

## 8. Core Manager Class (for Claude to Implement)

```python
class AsymmetricStopManager:
    def __init__(self, portfolio_size: float, max_positions: int, k_aggression: float = 1.0):
        self.portfolio = PortfolioState(
            portfolio_size=portfolio_size,
            max_positions=max_positions,
            max_loss_per_position=0.005 * portfolio_size,  # configurable
            max_daily_loss=0.025 * portfolio_size,
            k_aggression=k_aggression,
        )
        self.vix_level = 20.0
        self.beta_cache: dict[str, float] = {}
        self.stats = AsymmetryStats()
```


### 8.1 Helper: Beta and VIX

- Beta from IBKR fundamentals or an external table.
- VIX from continuous market data subscription.

```python
def get_beta(self, symbol: str) -> float:
    return self.beta_cache.get(symbol, 1.0)
```


### 8.2 Opening a Position

Called when user opens an option trade (either detected via `position()` callback or explicitly):

```python
def open_position(
    self,
    conid: int,
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    option_price: float,
    underlying_price: float,
    delta: float,
    dte: int,
):
    if not self.portfolio.can_open_new():
        # log and return
        return

    beta = self.get_beta(symbol)
    vix = self.vix_level

    # stop computation
    d_base = self.portfolio.k_aggression * beta * (vix / 100 / math.sqrt(252))
    d_stop = clamp(adjust_for_dte(d_base, dte), STOP_MIN, STOP_MAX)
    S_stop = underlying_price * (1 - d_stop)

    # position size
    loss_cap = self.portfolio.max_loss_per_position
    loss_per_contract = option_price * 100 * delta * d_stop
    if loss_per_contract <= 0:
        return
    qty = max(1, int(loss_cap / loss_per_contract))

    pos = AsymmetricPosition(
        conid=conid,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        quantity=qty,
        avg_entry_price=option_price,
        capital_allocated=option_price * qty * 100,
        max_loss_for_position=loss_cap,
        entry_time=datetime.now(),
        underlying_entry_price=underlying_price,
        entry_vix=vix,
        entry_beta=beta,
        stop_distance_pct=d_stop,
        stop_level_underlying=S_stop,
        stop_level_current=S_stop,
        underlying_high=underlying_price,
    )
    self.portfolio.positions[conid] = pos
```


### 8.3 Tick Handlers Integration

In `EWrapper.tickPrice` / `tickOptionComputation`, update:

```python
def on_underlying_price(self, conid: int, price: float):
    pos = self.portfolio.positions.get(conid)
    if not pos:
        return
    pos.underlying_price = price
    triggered = update_trailing_stop(pos, price)
    if triggered:
        self._request_exit(pos)

def on_option_bid_ask(self, conid: int, bid: float | None, ask: float | None):
    pos = self.portfolio.positions.get(conid)
    if not pos:
        return
    if bid is not None:
        pos.current_bid = bid
    if ask is not None:
        pos.current_ask = ask
    # update unrealized pnl and excursions
    if pos.current_bid is not None:
        pos.unrealized_pnl = (pos.current_bid - pos.avg_entry_price) * pos.quantity * 100
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.unrealized_pnl)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, pos.unrealized_pnl)
```


### 8.4 Exit Request and Smart Limit

```python
def _request_exit(self, pos: AsymmetricPosition):
    # Compute theoretical price at stop level (can use BS or delta approx)
    if pos.delta is None or pos.underlying_price is None:
        theo = pos.current_bid or pos.avg_entry_price
    else:
        # simple: move with delta from current underlying to stop level
        dS = pos.stop_level_current - pos.underlying_price
        theo = max((pos.current_bid or pos.avg_entry_price) + pos.delta * dS, 0.01)

    bid = pos.current_bid or theo * 0.9
    ask = pos.current_ask or theo * 1.1

    # smart limit – somewhere between bid and theo, capped by theo
    spread = max(0.01, ask - bid)
    mid = bid + spread / 2
    limit = min(theo, max(bid, mid))  # conservative

    # build IB order (SELL to close, LMT @ limit)
    # place via EClient.placeOrder(...)
    # store orderId on pos for tracking
```

On `orderStatus` filled:

```python
def on_exit_filled(self, conid: int, fill_price: float):
    pos = self.portfolio.positions.pop(conid, None)
    if not pos:
        return
    pos.exit_price = fill_price
    pos.realized_pnl = (fill_price - pos.avg_entry_price) * pos.quantity * 100
    pos.exit_time = datetime.now()
    self.portfolio.closed_positions.append(pos)
    self.stats.add(pos.realized_pnl)
```


***

## 9. Testing Plan (for Antigravity)

1. **Unit tests**:
    - Verify stop distance math (β, VIX, DTE).
    - Verify position size respects max loss.
    - Verify stops only move up, not down.
2. **Paper trading tests**:
    - Open 2–3 positions in liquid names (SPY, QQQ).
    - Check:
        - Loss on stopped positions ≈ configured cap.
        - Winners’ stops ratchet up correctly.
        - No profit targets; winners keep running.
3. **Statistical validation**:
    - After N trades (e.g. 50–100 paper trades), compute:
        - Win rate, avg winner, avg loser, profit factor, expected value.
    - Confirm **avg winner > 2× avg loser** and profit factor > 1.5.

***

## 10. Instructions to Claude / Antigravity

- Implement this plan in **Python** using `ibapi` against IBKR TWS / Gateway.
- Separate modules:
    - `asymmetry_risk.py` – PortfolioState, AsymmetricPosition, AsymmetricStopManager.
    - `ib_connector.py` – EClient/EWrapper subclass, mapping IB callbacks to manager methods.
    - `analytics.py` – AsymmetryStats, dashboard printing.
- Focus on:
    - Accurate math for stop distances and sizing.
    - Clean integration with real‑time IBKR data.
    - Logging for every state change (position open, stop move, exit).
- Do **not** add discretionary strategies; this system is **purely risk/execution + asymmetry** on positions the user chooses.
<span style="display:none">[^1][^2][^3][^4][^5][^6][^7]</span>

<div align="center">⁂</div>

[^1]: image.jpg

[^2]: image.jpg

[^3]: image.jpg

[^4]: image.jpg

[^5]: image.jpg

[^6]: image.jpg

[^7]: image.jpg

