"""
AI Signal Backtest with IB Historical Data
===========================================

Comprehensive backtest using:
1. IB historical data for all watchlist symbols
2. AI Signal Generator for BUY CALL / BUY PUT signals
3. Trailing stop strategy with volatility-aware stops
4. Full P&L tracking and analysis

REQUIRES: IB Gateway/TWS running on port 7497
"""

import sys
import time
import math
import csv
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import pandas as pd
import numpy as np

# Import AI signal modules
from ai_signal_generator import AISignalGenerator, SignalType, SignalResult
from stop_calculator import StopCalculator
from models import get_beta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

IB_HOST = "127.0.0.1"
IB_PORT = 7497              # Paper trading port
IB_CLIENT_ID = 450

# Portfolio settings
PORTFOLIO_SIZE = 100_000    # $100K portfolio
MAX_POSITION_SIZE = 10_000  # Max $10K per position
MAX_POSITIONS = 10          # Max concurrent positions
MAX_CONTRACTS = 2           # Contracts per position

# Watchlist
WATCHLIST_FILE = "watchlist.csv"

# AI Signal settings (matching dashboard defaults)
AI_MIN_SCORE = 60           # Minimum consensus score
AI_AUTO_EXECUTE = 85        # Auto-execute threshold

# VIX for volatility calculations
VIX_LEVEL = 22.0

# Stop System
STOP_CALC = StopCalculator(
    k_aggression=1.0,       # Default aggression
    min_trail_pct=0.04,     # 4% floor
    max_trail_pct=0.15      # 15% cap
)
DAYS_TO_EXPIRY = 21         # 3-week options

# Option Pricing (simulation)
OPTION_PRICE_CALL_PCT = 0.025   # ~2.5% of underlying for ATM call
OPTION_PRICE_PUT_PCT = 0.020    # ~2% of underlying for ATM put
OPTION_DELTA_CALL = 0.50
OPTION_DELTA_PUT = -0.50
OPTION_THETA = 0.08         # Theta decay per day ($8 per contract)
OPTION_SLIPPAGE = 0.02      # Slippage per contract side


@dataclass
class BacktestTrade:
    """Track a single backtest trade."""
    symbol: str
    signal_type: str        # BUY_CALL or BUY_PUT
    ai_score: float
    ai_reasons: List[str]
    
    entry_time: datetime
    entry_price: float      # Option price
    entry_underlying: float
    quantity: int = 1
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    
    pnl_per_contract: float = 0.0
    total_pnl: float = 0.0
    exit_reason: str = ""
    
    # Stop tracking
    stop_level: float = 0.0
    high_water_mark: float = 0.0    # For calls
    low_water_mark: float = float('inf')  # For puts


class AIBacktester(EClient, EWrapper):
    """Backtest AI signals with IB historical data."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 6000
        
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[BacktestTrade] = []
        
        # AI Signal Generator
        self.ai_gen = AISignalGenerator()
        
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
    
    def load_watchlist(self, filepath: str) -> List[Tuple[str, float]]:
        """Load symbols and estimate betas from watchlist."""
        symbols = []
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"Watchlist not found: {filepath}")
            return symbols
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('Symbol', '').strip()
                if symbol and symbol != '':
                    beta = get_beta(symbol)
                    symbols.append((symbol, beta))
        
        logger.info(f"Loaded {len(symbols)} symbols from watchlist")
        return symbols
    
    def fetch_historical_bars(self, symbol: str, duration: str = "5 D") -> Optional[pd.DataFrame]:
        """Fetch 5-min bars for a symbol (better for AI signals)."""
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
        
        # Use 5-min bars for better AI signal detection
        self.reqHistoricalData(
            req_id, contract, "", duration, "5 mins",
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
            return None
        
        df = pd.DataFrame([{
            'timestamp': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ {symbol}: {len(df)} bars")
        return df
    
    def run_backtest(self, symbols: List[Tuple[str, float]], max_symbols: int = 30):
        """Run full backtest on symbols."""
        print("\n" + "=" * 70)
        print("AI SIGNAL BACKTEST - IB HISTORICAL DATA")
        print("=" * 70)
        print(f"AI Score Threshold: {AI_MIN_SCORE}")
        print(f"Auto-Execute Threshold: {AI_AUTO_EXECUTE}")
        print(f"Portfolio Size: ${PORTFOLIO_SIZE:,}")
        print("=" * 70 + "\n")
        
        # Limit symbols for testing
        test_symbols = symbols[:max_symbols]
        print(f"Testing {len(test_symbols)} symbols...\n")
        
        # Fetch data for all symbols
        for i, (symbol, beta) in enumerate(test_symbols):
            print(f"Fetching {i+1}/{len(test_symbols)}: {symbol}...", end="\r")
            df = self.fetch_historical_bars(symbol, "5 D")
            if df is not None and len(df) > 60:  # Need enough for AI
                self.symbol_data[symbol] = df
                time.sleep(0.2)  # Rate limit
        
        if not self.symbol_data:
            logger.error("No data fetched!")
            return
        
        print(f"\nData fetched for {len(self.symbol_data)} symbols")
        print("-" * 70)
        
        # Create beta lookup
        beta_map = {s: b for s, b in test_symbols}
        
        # Get union of timestamps
        all_times = set()
        for df in self.symbol_data.values():
            all_times.update(df['timestamp'].tolist())
        all_times = sorted(all_times)
        
        print(f"Simulating {len(all_times)} time steps...\n")
        
        # Tracking
        positions: Dict[str, BacktestTrade] = {}
        signals_found = 0
        
        # Simulate through time
        for i, current_time in enumerate(all_times):
            for symbol, df in self.symbol_data.items():
                # Get historical data up to current time
                mask = df['timestamp'] <= current_time
                if mask.sum() < 60:  # Need 60 bars minimum for AI
                    continue
                
                df_current = df[mask].copy()
                current_price = df_current['close'].iloc[-1]
                beta = beta_map.get(symbol, 1.0)
                
                # ===== AI SIGNAL CHECK =====
                if symbol not in positions and len(positions) < MAX_POSITIONS:
                    try:
                        signal = self.ai_gen.generate_signal_from_data(
                            df_current, symbol, "5m"
                        )
                        
                        if signal.signal_type != SignalType.NO_SIGNAL and signal.consensus_score >= AI_MIN_SCORE:
                            signals_found += 1
                            
                            # Calculate option price
                            if signal.signal_type == SignalType.BUY_CALL:
                                option_price = current_price * OPTION_PRICE_CALL_PCT
                                delta = OPTION_DELTA_CALL
                            else:  # BUY_PUT
                                option_price = current_price * OPTION_PRICE_PUT_PCT
                                delta = OPTION_DELTA_PUT
                            
                            # Calculate position size
                            cost_per_contract = option_price * 100
                            affordable = min(MAX_CONTRACTS, int(MAX_POSITION_SIZE / cost_per_contract))
                            quantity = max(1, affordable)
                            
                            # Initial stop level
                            idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                            
                            if signal.signal_type == SignalType.BUY_CALL:
                                stop_level = STOP_CALC.compute_underlying_stop(
                                    current_price, beta, idx_vol, DAYS_TO_EXPIRY, "long"
                                )
                            else:
                                # For puts, stop is above entry
                                stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, DAYS_TO_EXPIRY)
                                stop_level = current_price * (1 + stop_dist)
                            
                            trade = BacktestTrade(
                                symbol=symbol,
                                signal_type=signal.signal_type.value,
                                ai_score=signal.consensus_score,
                                ai_reasons=signal.reasons,
                                entry_time=current_time,
                                entry_price=option_price,
                                entry_underlying=current_price,
                                quantity=quantity,
                                stop_level=stop_level,
                                high_water_mark=current_price,
                                low_water_mark=current_price
                            )
                            positions[symbol] = trade
                            
                            print(f"[{current_time}] {signal.signal_type.value} {quantity}x {symbol} @ ${option_price:.2f} | "
                                  f"Score: {signal.consensus_score:.0f} | Stop: ${stop_level:.2f}")
                            
                    except Exception as e:
                        pass  # Skip on error
                
                # ===== POSITION MANAGEMENT =====
                if symbol in positions:
                    trade = positions[symbol]
                    
                    # Update trailing stop
                    idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                    
                    if trade.signal_type == "BUY_CALL":
                        # Trail stop up on calls
                        if current_price > trade.high_water_mark:
                            trade.high_water_mark = current_price
                            new_stop = STOP_CALC.compute_trail_from_high(
                                current_price, beta, idx_vol, DAYS_TO_EXPIRY
                            )
                            trade.stop_level = max(trade.stop_level, new_stop)
                        
                        # Check stop (price dropped below stop)
                        stopped_out = current_price <= trade.stop_level
                        
                    else:  # BUY_PUT
                        # Trail stop down on puts
                        if current_price < trade.low_water_mark:
                            trade.low_water_mark = current_price
                            stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, DAYS_TO_EXPIRY)
                            new_stop = current_price * (1 + stop_dist)
                            trade.stop_level = min(trade.stop_level, new_stop)
                        
                        # Check stop (price rose above stop)
                        stopped_out = current_price >= trade.stop_level
                    
                    if stopped_out:
                        # Close position
                        underlying_move = current_price - trade.entry_underlying
                        delta = OPTION_DELTA_CALL if trade.signal_type == "BUY_CALL" else OPTION_DELTA_PUT
                        
                        # Time decay
                        days_held = (current_time - trade.entry_time).total_seconds() / 86400
                        theta_decay = days_held * OPTION_THETA
                        
                        # Calculate exit price
                        raw_change = underlying_move * delta
                        exit_price = max(trade.entry_price + raw_change - theta_decay - OPTION_SLIPPAGE * 2, 0.01)
                        
                        trade.exit_time = current_time
                        trade.exit_price = exit_price
                        trade.exit_underlying = current_price
                        trade.pnl_per_contract = (exit_price - trade.entry_price) * 100
                        trade.total_pnl = trade.pnl_per_contract * trade.quantity
                        trade.exit_reason = "stop"
                        
                        self.trades.append(trade)
                        del positions[symbol]
                        
                        pnl_str = f"${trade.total_pnl:+,.2f}"
                        print(f"[{current_time}] STOP {trade.quantity}x {symbol} @ ${exit_price:.2f} | P&L: {pnl_str}")
        
        # Close remaining positions at end
        print("\n--- End of Data ---")
        for symbol, trade in positions.items():
            df = self.symbol_data[symbol]
            final_price = df['close'].iloc[-1]
            underlying_move = final_price - trade.entry_underlying
            delta = OPTION_DELTA_CALL if trade.signal_type == "BUY_CALL" else OPTION_DELTA_PUT
            
            days_held = (df['timestamp'].iloc[-1] - trade.entry_time).total_seconds() / 86400
            theta_decay = days_held * OPTION_THETA
            
            raw_change = underlying_move * delta
            exit_price = max(trade.entry_price + raw_change - theta_decay - OPTION_SLIPPAGE * 2, 0.01)
            
            trade.exit_time = df['timestamp'].iloc[-1]
            trade.exit_price = exit_price
            trade.exit_underlying = final_price
            trade.pnl_per_contract = (exit_price - trade.entry_price) * 100
            trade.total_pnl = trade.pnl_per_contract * trade.quantity
            trade.exit_reason = "open"
            
            self.trades.append(trade)
            print(f"[MARK] {trade.quantity}x {symbol} | Unrealized P&L: ${trade.total_pnl:+,.2f}")
        
        self.print_summary(signals_found)
    
    def print_summary(self, signals_found: int):
        """Print comprehensive backtest summary."""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)
        print(f"  Portfolio Size:     ${PORTFOLIO_SIZE:,}")
        print(f"  AI Score Threshold: {AI_MIN_SCORE}")
        print(f"  Signals Found:      {signals_found}")
        print(f"  Trades Executed:    {len(self.trades)}")
        
        if not self.trades:
            print("  No trades executed")
            return
        
        # Separate by type
        call_trades = [t for t in self.trades if t.signal_type == "BUY_CALL"]
        put_trades = [t for t in self.trades if t.signal_type == "BUY_PUT"]
        
        print(f"\n  BUY_CALL Trades:    {len(call_trades)}")
        print(f"  BUY_PUT Trades:     {len(put_trades)}")
        
        # P&L Analysis
        pnls = [t.total_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        # Separate realized vs unrealized
        realized = [t for t in self.trades if t.exit_reason == "stop"]
        unrealized = [t for t in self.trades if t.exit_reason == "open"]
        
        if realized:
            print(f"\n  CLOSED TRADES (Realized):")
            for t in realized:
                print(f"    {t.signal_type:10s} {t.quantity}x {t.symbol:5s}: "
                      f"${t.entry_price:.2f} → ${t.exit_price:.2f} | "
                      f"P&L: ${t.total_pnl:+8,.2f} | Score: {t.ai_score:.0f}")
        
        if unrealized:
            print(f"\n  OPEN POSITIONS (Mark-to-Market):")
            for t in unrealized:
                print(f"    {t.signal_type:10s} {t.quantity}x {t.symbol:5s}: "
                      f"${t.entry_price:.2f} → ${t.exit_price:.2f} | "
                      f"P&L: ${t.total_pnl:+8,.2f} | Score: {t.ai_score:.0f}")
        
        total_invested = sum(t.entry_price * 100 * t.quantity for t in self.trades)
        realized_pnl = sum(t.total_pnl for t in realized)
        unrealized_pnl = sum(t.total_pnl for t in unrealized)
        total_pnl = realized_pnl + unrealized_pnl
        
        print(f"\n  {'─' * 50}")
        print(f"  Total Contracts:    {sum(t.quantity for t in self.trades)}")
        print(f"  Total Invested:     ${total_invested:,.2f}")
        print(f"  Realized P&L:       ${realized_pnl:+,.2f}")
        print(f"  Unrealized P&L:     ${unrealized_pnl:+,.2f}")
        print(f"  TOTAL P&L:          ${total_pnl:+,.2f}")
        
        if total_invested > 0:
            print(f"  Return on Capital:  {total_pnl/total_invested*100:+.2f}%")
        
        if pnls:
            print(f"  Win Rate:           {len(winners)}/{len(pnls)} ({len(winners)/len(pnls)*100:.0f}%)")
            if winners:
                print(f"  Avg Winner:         ${sum(winners)/len(winners):+,.2f}")
            if losers:
                print(f"  Avg Loser:          ${abs(sum(losers)/len(losers)):,.2f}")
                if sum(losers) != 0:
                    pf = abs(sum(winners)) / abs(sum(losers))
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
            pass
        elif errorCode in [162, 10167, 10168]:
            pass  # Data warnings
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def main():
    print("\n" + "=" * 70)
    print("AI SIGNAL BACKTEST WITH IB HISTORICAL DATA")
    print("=" * 70)
    print(f"Watchlist: {WATCHLIST_FILE}")
    print(f"Duration: 5 days of 5-min bars")
    print(f"AI Min Score: {AI_MIN_SCORE}")
    print("=" * 70 + "\n")
    
    backtester = AIBacktester()
    
    # Load watchlist
    symbols = backtester.load_watchlist(WATCHLIST_FILE)
    if not symbols:
        logger.error("No symbols loaded!")
        return
    
    if not backtester.connect_and_start():
        return
    
    try:
        # Run with top 30 symbols for reasonable test time
        backtester.run_backtest(symbols, max_symbols=30)
    finally:
        backtester.disconnect()
        logger.info("Disconnected from IB")


if __name__ == "__main__":
    main()
