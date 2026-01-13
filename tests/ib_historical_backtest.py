"""
Real IB Historical Data Backtest
=================================

Fetches REAL historical data from IB Gateway and runs the full
trading system simulation:
1. Fetch 1-min bars for watchlist symbols
2. Screen for signals using real price data
3. Simulate entries and exits
4. Calculate P&L

REQUIRES: IB Gateway/TWS running on port 7497
"""

import sys
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import pandas as pd
import numpy as np

from screener.formulas import (
    expected_move, abnormality_score, enhanced_score,
    classify_signal, get_direction
)
from screener.indicators import get_all_indicators
from models import get_beta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
from screener.data_store import load_watchlist_with_betas
from stop_calculator import StopCalculator

# ============= Configuration =============

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 400

# Portfolio settings
PORTFOLIO_SIZE = 100_000       # $100K portfolio
MAX_POSITION_SIZE = 10_000     # Max $10K per position
MAX_POSITIONS = 10             # Max concurrent positions
MAX_CONTRACTS = 1              # Limit to 1 contract per position

# Watchlist
WATCHLIST_FILE = "watchlist.csv"  # Load full watchlist

# Thresholds
VIX_LEVEL = 22.0
ABN_THRESHOLD = 2.0
MIN_SCORE = 85             # Tightened from 75 to 85 for quality signals

# Stop System
STOP_CALC = StopCalculator(
    k_aggression=0.8,      # More conservative (tight stops)
    min_trail_pct=0.03,    # 3% floor
    max_trail_pct=0.15     # 15% cap
)
DAYS_TO_EXPIRY = 21        # Simulate 3-week options

# Simulation Assumptions (Pricing Model)
OPTION_PRICE_PCT = 0.02
OPTION_DELTA = 0.5
OPTION_THETA = 0.10       # Theta decay per day (approx $10/day per contract)
OPTION_SLIPPAGE = 0.02    # Slippage per share per trade (approx $2.00/contract)


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_underlying: float
    quantity: int = 1              # Number of contracts
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    pnl_per_contract: float = 0.0  # P&L per contract
    total_pnl: float = 0.0         # Total P&L (quantity * pnl)
    exit_reason: str = ""
    stop_level: float = 0.0        # Track stop level


class IBHistoricalBacktest(EClient, EWrapper):
    """Fetch real historical data from IB and run backtest."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 5000
        
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[BacktestTrade] = []
    
    def connect_and_start(self) -> bool:
        logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT}...")
        self.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
        
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect to IB")
            return False
        
        self.connected = True
        logger.info("✓ Connected to IB Gateway")
        return True
    
    def fetch_historical_bars(self, symbol: str, duration: str = "3 D") -> Optional[pd.DataFrame]:
        """Fetch historical 1-min bars for a symbol."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.bar_data[req_id] = []
        self.bar_complete[req_id] = False
        self.req_to_symbol[req_id] = symbol
        
        logger.info(f"Requesting {duration} of 1-min bars for {symbol}...")
        
        self.reqHistoricalData(
            req_id, contract, "", duration, "1 min",
            "TRADES", 1, 1, False, []
        )
        
        # Wait for data
        timeout = 30
        start = time.time()
        while not self.bar_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.warning(f"Timeout waiting for {symbol}")
                return None
            time.sleep(0.5)
        
        bars = self.bar_data.get(req_id, [])
        if not bars:
            logger.warning(f"No bars for {symbol}")
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            for bar in bars
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ {symbol}: {len(df)} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df
    
    def run_backtest(self):
        """Run backtest on fetched data."""
        print("\n" + "=" * 70)
        print("REAL HISTORICAL DATA BACKTEST")
        print("=" * 70)
        
        # Top 20 Liquid Symbols for Faster Testing
        symbols_to_test = [
            "AAPL", "NVDA", "TSLA", "AMD", "MSFT", 
            "AMZN", "GOOG", "META", "SPY", "QQQ",
            "NFLX", "INTC", "MU", "AVGO", "QCOM",
            "JPM", "BAC", "WMT", "DIS", "BA"
        ]
        
        # Manually assign betas (approximate)
        symbols_map = {
            "AAPL": 1.2, "NVDA": 1.8, "TSLA": 2.0, "AMD": 1.6, "MSFT": 1.1,
            "AMZN": 1.2, "GOOG": 1.1, "META": 1.3, "SPY": 1.0, "QQQ": 1.2,
            "NFLX": 1.4, "INTC": 1.1, "MU": 1.4, "AVGO": 1.3, "QCOM": 1.3,
            "JPM": 0.9, "BAC": 1.0, "WMT": 0.6, "DIS": 1.1, "BA": 1.4
        }
        
        print(f"Testing Top 20 Liquid Symbols (Fast Mode)")
        
        # Fetch data for all symbols
        for i, symbol in enumerate(symbols_to_test):
            print(f"Fetching {i+1}/{len(symbols_to_test)}: {symbol}...", end="\r")
            df = self.fetch_historical_bars(symbol, "3 D")
            if df is not None and len(df) > 50:
                self.symbol_data[symbol] = df
                time.sleep(0.1)  # Minimal rate limiting needed for 20
        
        if not self.symbol_data:
            logger.error("No data fetched!")
            return
        
        print(f"\nData fetched for {len(self.symbol_data)} symbols")
        print("-" * 70)
        
        # Get union of all timestamps
        all_times = set()
        for df in self.symbol_data.values():
            all_times.update(df['timestamp'].tolist())
        all_times = sorted(all_times)
        
        print(f"Simulating {len(all_times)} time steps...")
        
        # Tracking
        positions: Dict[str, BacktestTrade] = {}
        signals_found = 0
        
        # Simulate through time
        for i, current_time in enumerate(all_times):
            for symbol, df in self.symbol_data.items():
                # Get current and previous close
                mask = df['timestamp'] <= current_time
                if mask.sum() < 50:  # Need enough data for indicators
                    continue
                
                df_current = df[mask]
                current_row = df_current.iloc[-1]
                current_price = current_row['close']
                
                # Previous day close (approximate)
                prev_close = df_current.iloc[0]['close']
                if len(df_current) > 78:  # More than 1 day
                    prev_close = df_current.iloc[-78]['close']
                
                # Use beta from watchlist
                beta = symbols_map.get(symbol, 1.0)
                
                # ===== SCREENING =====
                actual_pct = (current_price - prev_close) / prev_close * 100
                exp_pct, _ = expected_move(beta, VIX_LEVEL, current_price)
                
                if exp_pct == 0:
                    continue
                
                abn = abnormality_score(actual_pct, exp_pct)
                
                # Get indicators
                indicators = get_all_indicators(df_current)
                direction = get_direction(actual_pct)
                
                score = enhanced_score(
                    actual_pct, exp_pct,
                    indicators['volume_ratio'],
                    indicators['macd_state'],
                    indicators['rsi'],
                    indicators['bb_pos'],
                    direction
                )
                
                # Check for signal
                if abn >= ABN_THRESHOLD and score >= MIN_SCORE:
                    # Check for existing position AND max positions limit
                    if symbol not in positions and len(positions) < MAX_POSITIONS:
                        signals_found += 1
                        
                        # Calculate position size (limited to 1 contract)
                        option_price = current_price * OPTION_PRICE_PCT
                        cost_per_contract = option_price * 100
                        quantity = MAX_CONTRACTS  # Force 1 contract
                        total_cost = quantity * cost_per_contract
                        
                        # Initial Stop Level (Volatility-Aware)
                        idx_vol = VIX_LEVEL / 100 / math.sqrt(252) # Daily Vol approx
                        init_stop = STOP_CALC.compute_underlying_stop(
                            entry_price=current_price,
                            beta=beta,
                            index_vol_pct=idx_vol,
                            days_to_expiry=DAYS_TO_EXPIRY,
                            direction="long"
                        )
                        
                        trade = BacktestTrade(
                            symbol=symbol,
                            entry_time=current_time,
                            entry_price=option_price,
                            entry_underlying=current_price,
                            quantity=quantity,
                            stop_level=init_stop
                        )
                        positions[symbol] = trade
                        
                        print(f"[{current_time}] BUY {quantity} {symbol} @ ${option_price:.2f} | "
                              f"Cost: ${total_cost:,.0f} | Score: {score:.0f} | StopUnder: ${init_stop:.2f}")
                
                # ===== POSITION MANAGEMENT =====
                if symbol in positions:
                    trade = positions[symbol]
                    
                    # Update Trailing Stop
                    # Only move stop UP if price moves up
                    current_idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                    new_stop = STOP_CALC.compute_underlying_stop(
                        entry_price=current_price, # Trailing from current price
                        beta=beta,
                        index_vol_pct=current_idx_vol,
                        days_to_expiry=DAYS_TO_EXPIRY,
                        direction="long"
                    )
                    
                    if new_stop > trade.stop_level:
                        trade.stop_level = new_stop
                    
                    # Check stop trigger
                    if current_price <= trade.stop_level:
                        # Close position - Delta ~0.5, plus Theta & Slippage
                        underlying_move = current_price - trade.entry_underlying
                        delta = OPTION_DELTA
                        
                        # Time decay
                        days_held = (current_time - trade.entry_time).total_seconds() / 86400
                        theta_decay = days_held * OPTION_THETA
                        
                        # Raw Price change
                        raw_price_change = underlying_move * delta
                        
                        # Net Exit Price (Entry + Change - Theta - Slippage)
                        # We apply slippage twice (Entry + Exit) effectively reducing the capture
                        exit_price = max(trade.entry_price + raw_price_change - theta_decay - (OPTION_SLIPPAGE * 2), 0.01)
                        
                        trade.exit_time = current_time
                        trade.exit_price = exit_price
                        trade.exit_underlying = current_price
                        trade.pnl_per_contract = (exit_price - trade.entry_price) * 100
                        trade.total_pnl = trade.pnl_per_contract * trade.quantity
                        trade.exit_reason = "stop"
                        
                        self.trades.append(trade)
                        del positions[symbol]
                        
                        print(f"[{current_time}] SELL {trade.quantity} {symbol} @ ${exit_price:.2f} | "
                              f"P&L: ${trade.total_pnl:+,.2f} (STOP)")
        
        # Close remaining positions
        print("\n--- End of Data ---")
        for symbol, trade in positions.items():
            df = self.symbol_data[symbol]
            final_price = df['close'].iloc[-1]
            underlying_move = final_price - trade.entry_underlying
            delta = OPTION_DELTA
            
            # Time decay
            days_held = (df['timestamp'].iloc[-1] - trade.entry_time).total_seconds() / 86400
            theta_decay = days_held * OPTION_THETA
            
            # Raw Price change
            raw_price_change = underlying_move * delta
            
            # Net Exit Price (Mark-to-Market)
            # Apply 1x slippage for entry, 1x for theoretical exit if calculating value
            exit_price = max(trade.entry_price + raw_price_change - theta_decay - (OPTION_SLIPPAGE * 2), 0.01)
            
            trade.exit_time = df['timestamp'].iloc[-1]
            trade.exit_price = exit_price
            trade.exit_underlying = final_price
            trade.pnl_per_contract = (exit_price - trade.entry_price) * 100
            trade.total_pnl = trade.pnl_per_contract * trade.quantity
            trade.exit_reason = "open"  # Marked as open (unrealized)
            
            self.trades.append(trade)
            print(f"[MARK] Open Position {trade.quantity} {symbol} @ ${exit_price:.2f} | Unrealized P&L: ${trade.total_pnl:+,.2f}")
        
        self.print_summary(signals_found)
    
    def print_summary(self, signals_found: int):
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY - $100,000 PORTFOLIO")
        print("=" * 70)
        print(f"  Portfolio Size:     ${PORTFOLIO_SIZE:,}")
        print(f"  Max Position Size:  ${MAX_POSITION_SIZE:,}")
        print(f"  Signals Found:      {signals_found}")
        print(f"  Trades Executed:    {len(self.trades)}")
        
        if self.trades:
            pnls = [t.total_pnl for t in self.trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            
            total_contracts = sum(t.quantity for t in self.trades)
            total_invested = sum(t.entry_price * 100 * t.quantity for t in self.trades)
            
            total_contracts = sum(t.quantity for t in self.trades)
            total_invested = sum(t.entry_price * 100 * t.quantity for t in self.trades)
            
            # Separate Realized vs Unrealized
            realized_trades = [t for t in self.trades if t.exit_reason == "stop"]
            unrealized_trades = [t for t in self.trades if t.exit_reason == "open"]
            
            if realized_trades:
                print(f"\n  CLOSED TRADES (Realized):")
                for t in realized_trades:
                    print(f"    {t.quantity:2d}x {t.symbol:5s}: ${t.entry_price:.2f} → ${t.exit_price:.2f} | "
                          f"P&L: ${t.total_pnl:+8,.2f} ({t.exit_reason})")

            if unrealized_trades:
                print(f"\n  OPEN POSITIONS (Unrealized/Mark-to-Market):")
                for t in unrealized_trades:
                    print(f"    {t.quantity:2d}x {t.symbol:5s}: ${t.entry_price:.2f} → ${t.exit_price:.2f} | "
                          f"P&L: ${t.total_pnl:+8,.2f} (holding)")
            
            realized_pnl = sum(t.total_pnl for t in realized_trades)
            unrealized_pnl = sum(t.total_pnl for t in unrealized_trades)
            total_pnl = realized_pnl + unrealized_pnl
            
            print(f"\n  Total Contracts:    {total_contracts}")
            print(f"  Total Invested:     ${total_invested:,.2f}")
            print(f"  Realized P&L:       ${realized_pnl:+,.2f}")
            print(f"  Unrealized P&L:     ${unrealized_pnl:+,.2f}")
            print(f"  TOTAL P&L:          ${total_pnl:+,.2f}")
            print(f"  Return on Capital:  {total_pnl/total_invested*100:+.2f}%")
            print(f"  Win Rate:           {len(winners)}/{len(pnls)} ({len(winners)/len(pnls)*100:.0f}%)")
            if winners:
                print(f"  Avg Winner:         ${sum(winners)/len(winners):+,.2f}")
            if losers:
                print(f"  Avg Loser:          ${abs(sum(losers)/len(losers)):,.2f}")
                pf = abs(sum(winners)) / abs(sum(losers)) if sum(losers) else float('inf')
                print(f"  Profit Factor:      {pf:.2f}")
        
        print("=" * 70 + "\n")
    
    # ========== EWrapper Callbacks ==========
    
    def historicalData(self, reqId, bar):
        if reqId not in self.bar_data:
            self.bar_data[reqId] = []
        self.bar_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId, start, end):
        self.bar_complete[reqId] = True
    
    def nextValidId(self, orderId):
        logger.info(f"Next order ID: {orderId}")
    
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode in [2104, 2106, 2158]:
            pass  # Info messages
        elif errorCode in [162, 10167, 10168]:
            logger.debug(f"Data warning: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def main():
    print("\n" + "=" * 70)
    print("IB HISTORICAL DATA BACKTEST")
    print("=" * 70)
    print(f"Watchlist: {WATCHLIST_FILE}")
    print(f"Duration: 3 days of 1-min bars")
    print("=" * 70 + "\n")
    
    backtest = IBHistoricalBacktest()
    
    if not backtest.connect_and_start():
        return
    
    try:
        backtest.run_backtest()
    finally:
        backtest.disconnect()
        logger.info("Disconnected from IB")


if __name__ == "__main__":
    main()
