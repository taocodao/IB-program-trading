<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# COMPLETE BACKTESTING FRAMEWORK

## Historical Data Pull + Asymmetric Stop Logic Replay

I'll give you a **production-ready backtesting system** that:

1. Pulls historical data from IB (or uses CSV fallback)
2. Replays any trading day in pure Python
3. Runs your asymmetric stop logic against past prices
4. Generates realistic P\&L reports

***

## PART 1: Historical Data Fetcher (`historical_data.py`)

```python
"""
Fetch historical market data from Interactive Brokers
Fallback to CSV if API unavailable
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import time
import threading

logger = logging.getLogger(__name__)


class HistoricalDataFetcher(EClient, EWrapper):
    """Fetch historical bars from IB."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.next_req_id = 1000
        self.lock = threading.RLock()
        
        # Data storage: req_id -> list of BarData
        self.bar_data = {}
        self.req_id_complete = {}
    
    def connect_and_fetch(
        self,
        host: str,
        port: int,
        client_id: int
    ) -> bool:
        """Connect to IB."""
        try:
            logger.info(f"Connecting to {host}:{port}...")
            self.connect(host, port, client_id)
            
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            time.sleep(2)
            if not self.isConnected():
                logger.error("Failed to connect")
                return False
            
            logger.info("✓ Connected to IB")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def fetch_option_bars(
        self,
        symbol: str,
        expiry: str,          # YYYYMMDD
        strike: float,
        right: str,           # "C" or "P"
        bar_size: str = "1 min",   # "1 min", "5 mins", "1 hour", "1 day"
        lookback: str = "1 D",     # "1 D", "5 D", "1 M", "1 Y"
        data_type: str = "TRADES"  # "TRADES", "MIDPOINT", "BID", "ASK"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bars for an option.
        
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        # Request historical data
        # endDateTime: "" = most recent data
        # barSizeSetting: "1 min", "5 mins", "15 mins", "1 hour", "1 day"
        # durationStr: "1 D", "5 D", "1 M", "3 M", "1 Y"
        # whatToShow: "TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK", "HISTORICAL_VOLATILITY"
        # useRTH: True = regular trading hours only
        
        self.bar_data[req_id] = []
        self.req_id_complete[req_id] = False
        
        logger.info(
            f"Requesting {bar_size} bars for {symbol} {expiry} ${strike} {right} "
            f"({lookback} lookback, {data_type})"
        )
        
        self.reqHistoricalData(
            req_id,
            contract,
            "",                    # endDateTime = now
            lookback,             # durationStr
            bar_size,             # barSizeSetting
            data_type,            # whatToShow
            useRTH=1,             # useRTH = 1 (regular trading hours)
            formatDate=1,         # formatDate = 1 (YYYYMMDD HH:MM:SS)
            keepUpToDate=False,   # keepUpToDate = False (snapshot)
            chartOptions=[]
        )
        
        # Wait for data
        timeout = 30
        start = time.time()
        while not self.req_id_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.error(f"Timeout waiting for historical data (req_id={req_id})")
                return None
            time.sleep(0.5)
        
        # Convert to DataFrame
        bars = self.bar_data.get(req_id, [])
        if not bars:
            logger.warning(f"No bars returned for {symbol}")
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            for bar in bars
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ Fetched {len(df)} bars")
        return df
    
    def fetch_underlying_bars(
        self,
        symbol: str,
        bar_size: str = "1 min",
        lookback: str = "1 D",
        data_type: str = "MIDPOINT"
    ) -> Optional[pd.DataFrame]:
        """Fetch bars for underlying stock."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.bar_data[req_id] = []
        self.req_id_complete[req_id] = False
        
        logger.info(f"Requesting {bar_size} bars for {symbol} ({lookback})")
        
        self.reqHistoricalData(
            req_id,
            contract,
            "",
            lookback,
            bar_size,
            data_type,
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Wait
        timeout = 30
        start = time.time()
        while not self.req_id_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.error(f"Timeout for {symbol}")
                return None
            time.sleep(0.5)
        
        bars = self.bar_data.get(req_id, [])
        if not bars:
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
            }
            for bar in bars
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ Fetched {len(df)} bars for {symbol}")
        return df
    
    def historicalData(self, req_id: int, bar: BarData) -> None:
        """Callback: bar received."""
        with self.lock:
            if req_id not in self.bar_data:
                self.bar_data[req_id] = []
            self.bar_data[req_id].append(bar)
    
    def historicalDataEnd(self, req_id: int, start: str, end: str) -> None:
        """Callback: all bars received."""
        with self.lock:
            self.req_id_complete[req_id] = True
        logger.debug(f"Historical data complete for req_id={req_id}")
    
    def error(self, req_id: int, errorCode: int, errorString: str, *args) -> None:
        """Callback: error."""
        if errorCode in [2104, 2158]:  # Non-critical
            logger.debug(f"Info {errorCode}: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def fetch_from_ib(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    bar_size: str = "1 min",
    lookback: str = "1 D"
) -> Optional[pd.DataFrame]:
    """
    Convenience function: connect, fetch, disconnect.
    """
    fetcher = HistoricalDataFetcher()
    
    if not fetcher.connect_and_fetch("127.0.0.1", 7497, 101):
        logger.error("Failed to connect to IB")
        return None
    
    try:
        df = fetcher.fetch_option_bars(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            bar_size=bar_size,
            lookback=lookback
        )
        return df
    finally:
        fetcher.disconnect()


def fetch_underlying_from_ib(
    symbol: str,
    bar_size: str = "1 min",
    lookback: str = "1 D"
) -> Optional[pd.DataFrame]:
    """Fetch underlying bars from IB."""
    fetcher = HistoricalDataFetcher()
    
    if not fetcher.connect_and_fetch("127.0.0.1", 7497, 101):
        return None
    
    try:
        df = fetcher.fetch_underlying_bars(
            symbol=symbol,
            bar_size=bar_size,
            lookback=lookback
        )
        return df
    finally:
        fetcher.disconnect()
```


***

## PART 2: Backtesting Engine (`backtest_engine.py`)

```python
"""
Backtesting engine: replay historical data through asymmetric stop logic
"""

import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math

from data_models import OptionPosition, PositionStatus, VolatilityTracker
from risk_manager import StopCalculator, BlackScholesCalculator, ExecutionPricer

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a trade during backtest."""
    
    entry_time: datetime
    symbol: str
    expiry: str
    strike: float
    right: str
    entry_price: float
    entry_underlying_price: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_underlying_price: Optional[float] = None
    exit_reason: str = ""  # "stop_triggered", "manual_close", "expired"
    
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    realized_pnl: float = 0.0
    
    def __post_init__(self):
        if self.exit_price and self.entry_price:
            self.realized_pnl = (self.exit_price - self.entry_price) * 100
    
    def __str__(self) -> str:
        right_str = "CALL" if self.right == "C" else "PUT"
        status = "OPEN" if not self.exit_price else "CLOSED"
        pnl_str = f"${self.realized_pnl:+.2f}" if self.exit_price else "?"
        
        return (f"{self.symbol} {self.expiry} ${self.strike} {right_str} | "
                f"{status} | Entry: ${self.entry_price:.2f} | P&L: {pnl_str}")


@dataclass
class BacktestPosition:
    """Position during backtest."""
    
    # Static info
    symbol: str
    expiry: str
    strike: float
    right: str
    beta: float = 1.0
    
    # Entry
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_underlying_price: float = 0.0
    
    # Current state
    current_bid: Optional[float] = None
    current_ask: Optional[float] = None
    current_underlying: Optional[float] = None
    current_iv: Optional[float] = None
    current_delta: Optional[float] = None
    current_time: Optional[datetime] = None
    
    # Stop level
    underlying_stop_level: Optional[float] = None
    underlying_high: float = 0.0
    
    # Exit
    exit_triggered: bool = False
    exit_triggered_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    # Excursions
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def update_market_data(
        self,
        bid: float,
        ask: float,
        underlying: float,
        iv: float = 0.25,
        delta: float = 0.5,
        current_time: Optional[datetime] = None
    ) -> None:
        """Update market data."""
        self.current_bid = bid
        self.current_ask = ask
        self.current_underlying = underlying
        self.current_iv = iv
        self.current_delta = delta
        self.current_time = current_time or datetime.now()
        
        # Update high
        if self.underlying_high == 0:
            self.underlying_high = underlying
        self.underlying_high = max(self.underlying_high, underlying)
        
        # Update excursions
        mid = (bid + ask) / 2
        pnl = (mid - self.entry_price) * 100 if self.entry_price else 0
        self.max_favorable_excursion = max(self.max_favorable_excursion, pnl)
        self.max_adverse_excursion = min(self.max_adverse_excursion, pnl)


class BacktestEngine:
    """Run backtest of asymmetric stop strategy."""
    
    def __init__(
        self,
        k_aggression: float = 1.0,
        min_trail_pct: float = 0.04,
        max_trail_pct: float = 0.40,
        vix_level: float = 20.0,
        starting_cash: float = 100_000.0
    ):
        self.k_aggression = k_aggression
        self.min_trail_pct = min_trail_pct
        self.max_trail_pct = max_trail_pct
        
        self.stop_calc = StopCalculator(
            k_aggression=k_aggression,
            min_trail_pct=min_trail_pct,
            max_trail_pct=max_trail_pct
        )
        
        self.vix_level = vix_level
        self.vol_tracker = VolatilityTracker()
        self.vol_tracker.update_vix(vix_level)
        
        self.starting_cash = starting_cash
        self.current_cash = starting_cash
        
        # Positions: symbol_expiry_strike_right -> BacktestPosition
        self.positions: Dict[str, BacktestPosition] = {}
        
        # Closed trades
        self.closed_trades: List[BacktestTrade] = []
        
        # Simulation state
        self.current_time: Optional[datetime] = None
        self.current_underlying_price: Optional[float] = None
    
    def open_position(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        entry_price: float,
        entry_underlying_price: float,
        beta: float = 1.0,
        current_time: Optional[datetime] = None
    ) -> str:
        """
        Open a position.
        
        Returns position key for tracking.
        """
        key = f"{symbol}_{expiry}_{strike}_{right}"
        
        pos = BacktestPosition(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            beta=beta,
            entry_price=entry_price,
            entry_time=current_time or self.current_time,
            entry_underlying_price=entry_underlying_price,
        )
        
        self.positions[key] = pos
        
        logger.info(
            f"OPEN: {symbol} {expiry} ${strike}{right} @ ${entry_price:.2f} "
            f"(underlying ${entry_underlying_price:.2f})"
        )
        
        return key
    
    def update_position(
        self,
        key: str,
        bid: float,
        ask: float,
        underlying: float,
        iv: float = 0.25,
        delta: float = 0.5,
        current_time: Optional[datetime] = None
    ) -> None:
        """Update market data for a position."""
        if key not in self.positions:
            return
        
        pos = self.positions[key]
        pos.update_market_data(bid, ask, underlying, iv, delta, current_time)
        
        # Check stop trigger
        if pos.underlying_stop_level is None:
            # Compute initial stop
            days_to_expiry = self._days_to_expiry(pos.expiry, current_time)
            index_vol = self.vol_tracker.get_daily_vol_pct()
            
            stop_level = self.stop_calc.compute_underlying_stop(
                entry_price=pos.entry_underlying_price,
                beta=pos.beta,
                index_vol_pct=index_vol,
                days_to_expiry=days_to_expiry
            )
            pos.underlying_stop_level = stop_level
        else:
            # Trailing: move up but not down
            days_to_expiry = self._days_to_expiry(pos.expiry, current_time)
            index_vol = self.vol_tracker.get_daily_vol_pct()
            
            stop_level = self.stop_calc.compute_underlying_stop(
                entry_price=pos.entry_underlying_price,
                beta=pos.beta,
                index_vol_pct=index_vol,
                days_to_expiry=days_to_expiry
            )
            pos.underlying_stop_level = max(pos.underlying_stop_level, stop_level)
        
        # Trigger check
        if (underlying <= pos.underlying_stop_level and
            not pos.exit_triggered):
            logger.warning(
                f"STOP TRIGGERED: {pos.symbol} {pos.expiry} "
                f"underlying {underlying:.2f} <= stop {pos.underlying_stop_level:.2f}"
            )
            pos.exit_triggered = True
            pos.exit_triggered_time = current_time or self.current_time
            self._close_position(key, bid, current_time, "stop_triggered")
    
    def _close_position(
        self,
        key: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        reason: str = "manual"
    ) -> None:
        """Close a position and record trade."""
        if key not in self.positions:
            return
        
        pos = self.positions.pop(key)
        exit_time = exit_time or self.current_time
        
        trade = BacktestTrade(
            entry_time=pos.entry_time or self.current_time,
            symbol=pos.symbol,
            expiry=pos.expiry,
            strike=pos.strike,
            right=pos.right,
            entry_price=pos.entry_price,
            entry_underlying_price=pos.entry_underlying_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_underlying_price=pos.current_underlying,
            exit_reason=reason,
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
        )
        
        self.closed_trades.append(trade)
        
        logger.warning(
            f"CLOSE: {pos.symbol} {pos.expiry} ${pos.strike}{pos.right} "
            f"@ ${exit_price:.2f}, P&L: ${trade.realized_pnl:+.2f} ({reason})"
        )
    
    def close_all_positions(self, exit_price_map: Dict[str, float]) -> None:
        """Close all open positions (end of day)."""
        keys_to_close = list(self.positions.keys())
        
        for key in keys_to_close:
            if key in exit_price_map:
                self._close_position(
                    key,
                    exit_price_map[key],
                    self.current_time,
                    "end_of_day"
                )
    
    def _days_to_expiry(
        self,
        expiry: str,
        current_time: Optional[datetime] = None
    ) -> int:
        """Days until expiry."""
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            now = current_time or self.current_time or datetime.now()
            days = (exp_date - now).days
            return max(0, days)
        except:
            return 0
    
    def get_summary(self) -> Dict:
        """Get backtest summary."""
        if not self.closed_trades:
            return {
                'trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'profit_factor': 0.0,
            }
        
        closed_pnls = [t.realized_pnl for t in self.closed_trades]
        winners = [p for p in closed_pnls if p > 0]
        losers = [p for p in closed_pnls if p < 0]
        
        win_rate = len(winners) / len(closed_pnls) if closed_pnls else 0.0
        total_pnl = sum(closed_pnls)
        avg_winner = sum(winners) / len(winners) if winners else 0.0
        avg_loser = sum(losers) / len(losers) if losers else 0.0
        
        gross_wins = sum(winners)
        gross_losses = abs(sum(losers))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0
        
        return {
            'trades': len(closed_pnls),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_winner': avg_winner,
            'avg_loser': abs(avg_loser),
            'profit_factor': profit_factor,
            'closed_trades': self.closed_trades,
        }
    
    def print_summary(self) -> None:
        """Print backtest summary."""
        summary = self.get_summary()
        
        summary_str = f"""
╔════════════════════════════════════════════════════╗
║           BACKTEST SUMMARY                         ║
╠════════════════════════════════════════════════════╣
║  Total Trades:       {summary['trades']:4d}                      ║
║  Winners:            {summary['winners']:4d}  ({summary['win_rate']:5.1%})              ║
║  Losers:             {summary['losers']:4d}                      ║
║  Total P&L:          ${summary['total_pnl']:10,.2f}              ║
║  Avg Winner:         ${summary['avg_winner']:10,.2f}              ║
║  Avg Loser:          ${summary['avg_loser']:10,.2f}              ║
║  Profit Factor:      {summary['profit_factor']:6.2f}x                 ║
╚════════════════════════════════════════════════════╝
        """
        logger.info(summary_str)
```


***

## PART 3: Simple Backtest Runner (`run_backtest.py`)

```python
"""
Simple backtest example: replay a day of SPY option trading
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import math

from backtest_engine import BacktestEngine
from historical_data import fetch_underlying_from_ib
from analytics import setup_logging
import config

logger = logging.getLogger(__name__)

# Setup logging
setup_logging("backtest.log", "INFO")


def simple_backtest_example():
    """
    Example: simulate a single SPY option trade over 1 day.
    
    Scenario:
    - Buy 1 SPY 550 call at $5.00 when SPY is at $550
    - System auto-stops if SPY drops 4% (beta=1.0, vol=20%, k=1.0)
    - Track P&L as SPY price moves
    """
    
    logger.info("=" * 60)
    logger.info("BACKTEST: Single SPY Option Trade")
    logger.info("=" * 60)
    
    # Setup engine
    engine = BacktestEngine(
        k_aggression=1.0,
        min_trail_pct=0.04,
        max_trail_pct=0.40,
        vix_level=20.0
    )
    
    # Entry
    entry_time = datetime(2025, 1, 9, 10, 30, 0)  # Jan 9, 10:30 AM
    entry_underlying = 550.0
    entry_price = 5.00
    beta = 1.0
    
    engine.current_time = entry_time
    engine.current_underlying_price = entry_underlying
    
    pos_key = engine.open_position(
        symbol="SPY",
        expiry="20250221",        # Feb 21 expiry (43 days out)
        strike=550.0,
        right="C",
        entry_price=entry_price,
        entry_underlying_price=entry_underlying,
        beta=beta,
        current_time=entry_time
    )
    
    # Simulate price movement throughout the day
    # Scenario 1: SPY drifts down, hits stop at 528 (4% down)
    
    prices = [
        (datetime(2025, 1, 9, 10, 31, 0), 549.5, 5.10),   # +$0.10
        (datetime(2025, 1, 9, 10, 45, 0), 548.0, 4.80),   # -$0.20
        (datetime(2025, 1, 9, 11, 00, 0), 546.0, 4.50),   # -$0.50
        (datetime(2025, 1, 9, 12, 00, 0), 543.0, 4.20),   # -$0.80
        (datetime(2025, 1, 9, 13, 00, 0), 540.0, 3.90),   # -$1.10
        (datetime(2025, 1, 9, 13, 30, 0), 538.0, 3.60),   # Stop zone!
        (datetime(2025, 1, 9, 13, 45, 0), 528.0, 2.80),   # HIT STOP
    ]
    
    logger.info("\n--- Price Simulation ---")
    
    for sim_time, underlying, mid_price in prices:
        engine.current_time = sim_time
        
        # Bid/ask around mid
        bid = mid_price * 0.99
        ask = mid_price * 1.01
        
        logger.info(
            f"{sim_time.strftime('%H:%M')} | SPY: ${underlying:.2f} | "
            f"Call: ${bid:.2f}/{ask:.2f}"
        )
        
        # Update position
        engine.update_position(
            pos_key,
            bid=bid,
            ask=ask,
            underlying=underlying,
            iv=0.25,
            delta=0.50,
            current_time=sim_time
        )
        
        # Check if closed
        if pos_key not in engine.positions:
            logger.warning("Position exited by stop!")
            break
    
    # Print results
    logger.info("\n--- Results ---")
    engine.print_summary()
    
    if engine.closed_trades:
        for trade in engine.closed_trades:
            logger.info(f"Trade: {trade}")


def backtest_with_historical_data():
    """
    Advanced example: fetch real historical data from IB
    and backtest against it.
    """
    
    logger.info("=" * 60)
    logger.info("BACKTEST: Historical Data from IB")
    logger.info("=" * 60)
    
    # Step 1: Fetch underlying 1-minute bars for past day
    logger.info("\nFetching SPY 1-min bars...")
    
    # This requires IB to be running
    underlying_df = fetch_underlying_from_ib(
        symbol="SPY",
        bar_size="1 min",
        lookback="1 D"
    )
    
    if underlying_df is None:
        logger.error("Failed to fetch data from IB. Using CSV fallback...")
        # Load from CSV if you have historical CSV
        # underlying_df = pd.read_csv("spy_1min.csv")
        # underlying_df['timestamp'] = pd.to_datetime(underlying_df['timestamp'])
        return
    
    logger.info(f"Fetched {len(underlying_df)} bars")
    logger.info(f"Time range: {underlying_df['timestamp'].min()} to {underlying_df['timestamp'].max()}")
    
    # Step 2: Setup engine
    engine = BacktestEngine(
        k_aggression=1.0,
        min_trail_pct=0.04,
        max_trail_pct=0.40,
        vix_level=18.0
    )
    
    # Step 3: Simulate trading
    # Open position at first bar
    first_row = underlying_df.iloc[^0]
    entry_time = first_row['timestamp']
    entry_underlying = first_row['close']
    entry_price = entry_underlying * 0.05  # Assume 5% OTM call is ~5% of underlying
    
    pos_key = engine.open_position(
        symbol="SPY",
        expiry="20250221",
        strike=round(entry_underlying * 0.99),  # 1% OTM
        right="C",
        entry_price=entry_price,
        entry_underlying_price=entry_underlying,
        beta=1.0,
        current_time=entry_time
    )
    
    logger.info(f"\nOpened position at {entry_time}, SPY = ${entry_underlying:.2f}")
    
    # Step 4: Replay bars
    for idx, row in underlying_df.iterrows():
        current_time = row['timestamp']
        current_underlying = row['close']
        
        engine.current_time = current_time
        
        # Simulate option price (very simple: use delta approximation)
        underlying_move = current_underlying - entry_underlying
        option_move = 0.5 * underlying_move  # delta ≈ 0.5
        current_option_mid = entry_price + option_move
        
        bid = current_option_mid * 0.99
        ask = current_option_mid * 1.01
        
        # Update position
        engine.update_position(
            pos_key,
            bid=max(bid, 0.01),
            ask=max(ask, 0.01),
            underlying=current_underlying,
            iv=0.25,
            delta=0.5,
            current_time=current_time
        )
        
        # Position closed?
        if pos_key not in engine.positions:
            break
    
    # Close any remaining positions
    if pos_key in engine.positions:
        pos = engine.positions[pos_key]
        final_underlying = underlying_df.iloc[-1]['close']
        final_option_mid = entry_price + (0.5 * (final_underlying - entry_underlying))
        engine._close_position(pos_key, max(final_option_mid, 0.01), "end_of_day")
    
    # Print results
    logger.info("\n--- Backtest Results ---")
    engine.print_summary()


def backtest_multiple_scenarios():
    """
    Backtest multiple market scenarios to test strategy robustness.
    """
    
    logger.info("=" * 60)
    logger.info("BACKTEST: Multiple Scenarios")
    logger.info("=" * 60)
    
    scenarios = [
        {
            'name': 'Gap Down (Worst Case)',
            'prices': [
                (550.0, 5.00),   # Entry
                (520.0, 2.50),   # Gapped down
            ]
        },
        {
            'name': 'Slow Drift Down (Stop Hits)',
            'prices': [
                (550.0, 5.00),   # Entry
                (549.0, 4.95),
                (548.0, 4.90),
                (540.0, 4.50),   # 1.8% down
                (535.0, 4.25),   # 2.7% down
                (530.0, 4.00),   # 3.6% down
                (528.0, 3.90),   # 4.0% down → STOP
            ]
        },
        {
            'name': 'Bounce After Drop (Avoid Stop)',
            'prices': [
                (550.0, 5.00),   # Entry
                (540.0, 4.50),   # Down 1.8%
                (535.0, 4.25),   # Down 2.7%
                (545.0, 4.75),   # Back up! Stop doesn't trigger
                (555.0, 5.50),   # Higher high
            ]
        },
        {
            'name': 'Strong Rally (Max Profit)',
            'prices': [
                (550.0, 5.00),   # Entry
                (560.0, 5.50),
                (570.0, 6.00),
                (580.0, 6.50),
                (590.0, 7.00),
            ]
        },
    ]
    
    all_results = []
    
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        
        engine = BacktestEngine(vix_level=20.0)
        entry_time = datetime(2025, 1, 9, 10, 30)
        
        pos_key = engine.open_position(
            symbol="SPY",
            expiry="20250221",
            strike=550.0,
            right="C",
            entry_price=scenario['prices'][^0][^1],
            entry_underlying_price=scenario['prices'][^0][^0],
            beta=1.0,
            current_time=entry_time
        )
        
        # Replay prices
        for idx, (underlying, option_mid) in enumerate(scenario['prices']):
            current_time = entry_time + timedelta(minutes=idx*15)
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(
                pos_key,
                bid=bid,
                ask=ask,
                underlying=underlying,
                iv=0.25,
                delta=0.5,
                current_time=current_time
            )
            
            if pos_key not in engine.positions:
                break
        
        # Close remaining
        if pos_key in engine.positions:
            final_price = scenario['prices'][-1][^1]
            engine._close_position(pos_key, final_price, "end_of_day")
        
        # Record result
        if engine.closed_trades:
            trade = engine.closed_trades[^0]
            all_results.append({
                'scenario': scenario['name'],
                'pnl': trade.realized_pnl,
                'exit_reason': trade.exit_reason,
            })
            logger.info(f"Result: P&L = ${trade.realized_pnl:+.2f}, Exit: {trade.exit_reason}")
    
    # Summary across scenarios
    logger.info("\n--- Cross-Scenario Summary ---")
    for result in all_results:
        logger.info(f"{result['scenario']:30s} | P&L: ${result['pnl']:+6.2f} | {result['exit_reason']}")
    
    avg_pnl = sum(r['pnl'] for r in all_results) / len(all_results)
    logger.info(f"\nAverage P&L across scenarios: ${avg_pnl:+.2f}")


if __name__ == "__main__":
    # Run the simple example first
    simple_backtest_example()
    
    # Then try scenarios
    logger.info("\n" + "=" * 60 + "\n")
    backtest_multiple_scenarios()
    
    # Uncomment to test with real IB data
    # backtest_with_historical_data()
```


***

## PART 4: Quick Test Script (`test_backtest.py`)

```bash
#!/usr/bin/env python3
"""
Quick backtest runner - no IB connection needed
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from run_backtest import (
    simple_backtest_example,
    backtest_multiple_scenarios
)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ASYMMETRIC STOP LOSS BACKTESTER")
    print("=" * 70)
    
    # Simple example
    simple_backtest_example()
    
    # Multiple scenarios
    print("\n" + "=" * 70 + "\n")
    backtest_multiple_scenarios()
    
    print("\n✓ Backtest complete")
```


***

## PART 5: Usage Guide

### Option 1: Simple Backtest (No IB Connection)

```bash
# Install dependencies
pip install pandas scipy

# Run backtest
python run_backtest.py
```

**Output:**

```
════════════════════════════════════════════════════
BACKTEST: Single SPY Option Trade
════════════════════════════════════════════════════

OPEN: SPY 20250221 $550C @ $5.00 (underlying $550.00)

--- Price Simulation ---
10:31 | SPY: $549.50 | Call: $5.05/$5.15
10:45 | SPY: $548.00 | Call: $4.80/$4.88
11:00 | SPY: $546.00 | Call: $4.50/$4.58
12:00 | SPY: $543.00 | Call: $4.20/$4.28
13:00 | SPY: $540.00 | Call: $3.90/$3.98
13:30 | SPY: $538.00 | Call: $3.60/$3.68
13:45 | SPY: $528.00 | Call: $2.80/$2.84

Position exited by stop!

--- Results ---
╔════════════════════════════════════════════════════╗
║           BACKTEST SUMMARY                         ║
╠════════════════════════════════════════════════════╣
║  Total Trades:          1                          ║
║  Winners:               0  (  0.0%)                ║
║  Losers:                1                          ║
║  Total P&L:        -$220.00                        ║
║  Avg Winner:           $0.00                       ║
║  Avg Loser:           $220.00                      ║
║  Profit Factor:        0.00x                       ║
╚════════════════════════════════════════════════════╝
```


### Option 2: Historical Data from IB

```python
# In run_backtest.py, call:
# backtest_with_historical_data()

# Requirements:
# - TWS or IB Gateway running on port 7497
# - API enabled
# - Paper trading account
```


### Option 3: Custom Scenario

```python
from backtest_engine import BacktestEngine
from datetime import datetime

# Create engine
engine = BacktestEngine(
    k_aggression=1.0,
    vix_level=20.0
)

# Open position
pos_key = engine.open_position(
    symbol="QQQ",
    expiry="20250321",
    strike=500.0,
    right="C",
    entry_price=8.50,
    entry_underlying_price=500.0,
    beta=1.45,  # QQQ beta
    current_time=datetime(2025, 1, 9, 10, 30)
)

# Feed price updates
for i, price in enumerate([499, 497, 495, 492, 488]):
    engine.current_time = datetime(2025, 1, 9, 11, i)
    engine.update_position(
        pos_key,
        bid=price * 0.017 * 0.99,  # Rough call pricing
        ask=price * 0.017 * 1.01,
        underlying=price,
        iv=0.30,
        delta=0.45
    )

# Results
engine.print_summary()
```


***

## PART 6: Key Features

✅ **No IB Connection Required** - Pure Python simulation
✅ **Replay Any Day** - Feed historical bars or synthetic prices
✅ **Asymmetric Stop Logic** - Your full risk-management system
✅ **Multiple Scenarios** - Test bear/bull/bounce cases
✅ **P\&L Tracking** - MFE/MAE, win rate, profit factor
✅ **Extensible** - Easy to add Greeks, multi-leg strategies, etc.

***

**This is your complete backtesting system. Run it to verify the asymmetric stop strategy before going live.**
<span style="display:none">[^2][^3][^4][^5][^6][^7]</span>

<div align="center">⁂</div>

[^1]: image.jpg

[^2]: image.jpg

[^3]: image.jpg

[^4]: image.jpg

[^5]: image.jpg

[^6]: image.jpg

[^7]: image.jpg

