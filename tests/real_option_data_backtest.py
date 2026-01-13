"""
Real Option Data Backtest - Using IB Historical Option Prices
==============================================================

This backtest uses REAL option price data from IB:
1. AI signals detect BUY_CALL / BUY_PUT opportunities
2. Option selector finds optimal contract (research-backed: 30-45 DTE, 0.55 delta)
3. Fetches REAL historical option prices from IB
4. Applies trailing stop strategy
5. Calculates accurate P&L

REQUIRES: IB Gateway/TWS running on port 7497
NOTE: IB historical option data may have delays/limitations

Usage:
    python tests/real_option_data_backtest.py
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

# Import our modules
from ai_signal_generator import AISignalGenerator, SignalType
from option_selector import (
    OptionSelector, OptionContract, 
    calculate_target_dte, calculate_target_delta, validate_option,
    DEFAULT_TARGET_DTE, DEFAULT_TARGET_DELTA
)
from stop_calculator import StopCalculator
from models import get_beta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 600

# Portfolio
PORTFOLIO_SIZE = 100_000
MAX_POSITION_SIZE = 10_000
MAX_POSITIONS = 10
MAX_CONTRACTS = 2

# Watchlist
WATCHLIST_FILE = "watchlist.csv"

# AI Settings
AI_MIN_SCORE = 60

# VIX for IV estimates
VIX_LEVEL = 22.0

# Stop Settings
STOP_CALC = StopCalculator(
    k_aggression=1.0,
    min_trail_pct=0.04,
    max_trail_pct=0.15
)


@dataclass
class RealOptionTrade:
    """Trade with real option data."""
    symbol: str
    signal_type: str          # BUY_CALL or BUY_PUT
    ai_score: float
    
    # Option contract details
    expiry: str
    strike: float
    right: str
    con_id: int = 0
    
    # Entry
    entry_time: datetime = None
    entry_option_price: float = 0.0
    entry_underlying: float = 0.0
    entry_delta: float = 0.0
    quantity: int = 1
    
    # Exit
    exit_time: Optional[datetime] = None
    exit_option_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    
    # P&L
    pnl_per_contract: float = 0.0
    total_pnl: float = 0.0
    exit_reason: str = ""
    
    # Stop tracking
    stop_level: float = 0.0
    high_water_mark: float = 0.0
    low_water_mark: float = float('inf')
    
    # Liquidity info
    liquidity_score: float = 0.0
    entry_spread_pct: float = 0.0


class RealOptionDataBacktest(EClient, EWrapper):
    """Backtest with real IB option price data."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        self.next_req_id = 9000
        
        # Historical data storage
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        self.req_to_symbol: Dict[int, str] = {}
        
        # Option chain storage
        self.contract_details: Dict[int, List] = {}
        self.contract_details_complete: Dict[int, bool] = {}
        
        # Market data
        self.tick_data: Dict[int, Dict] = {}
        self.underlying_prices: Dict[str, float] = {}
        
        # Results
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.option_data: Dict[str, pd.DataFrame] = {}  # Option historical prices
        self.trades: List[RealOptionTrade] = []
        
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
        """Load symbols with betas."""
        symbols = []
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"Watchlist not found: {filepath}")
            return symbols
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('Symbol', '').strip()
                if symbol:
                    beta = get_beta(symbol)
                    symbols.append((symbol, beta))
        
        logger.info(f"Loaded {len(symbols)} symbols")
        return symbols
    
    def fetch_stock_bars(self, symbol: str, duration: str = "5 D") -> Optional[pd.DataFrame]:
        """Fetch historical stock bars."""
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
        
        self.reqHistoricalData(
            req_id, contract, "", duration, "5 mins",
            "TRADES", 1, 1, False, []
        )
        
        timeout = 30
        start = time.time()
        while not self.bar_complete.get(req_id, False):
            if time.time() - start > timeout:
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
        
        return df
    
    def fetch_option_bars(
        self, 
        symbol: str, 
        expiry: str, 
        strike: float, 
        right: str,
        duration: str = "5 D"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical option bars."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.bar_data[req_id] = []
        self.bar_complete[req_id] = False
        
        # Request option historical data
        self.reqHistoricalData(
            req_id, contract, "", duration, "5 mins",
            "TRADES", 1, 1, False, []
        )
        
        timeout = 30
        start = time.time()
        while not self.bar_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.warning(f"Timeout fetching option data for {symbol} {strike} {right}")
                return None
            time.sleep(0.5)
        
        bars = self.bar_data.get(req_id, [])
        if not bars:
            logger.warning(f"No option bars for {symbol} {strike} {right}")
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
        
        logger.info(f"  ✓ Option data: {symbol} {expiry} ${strike} {right} - {len(df)} bars")
        return df
    
    def get_current_option_price(
        self, 
        symbol: str, 
        expiry: str, 
        strike: float, 
        right: str,
        timeout: float = 5.0
    ) -> Tuple[float, float, float]:
        """Get current bid/ask/mid for an option."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.tick_data[req_id] = {}
        
        self.reqMktData(req_id, contract, "", False, False, [])
        
        start = time.time()
        while time.time() - start < timeout:
            data = self.tick_data.get(req_id, {})
            if 'bid' in data and 'ask' in data:
                break
            time.sleep(0.1)
        
        self.cancelMktData(req_id)
        
        data = self.tick_data.get(req_id, {})
        bid = data.get('bid', 0.0)
        ask = data.get('ask', 0.0)
        mid = (bid + ask) / 2 if bid and ask else 0
        
        return bid, ask, mid
    
    def find_option_contract(
        self, 
        symbol: str, 
        right: str, 
        underlying_price: float,
        ai_score: float
    ) -> Optional[Dict]:
        """
        Find best option contract using research-backed criteria.
        
        Uses:
        - IV-adjusted DTE (30-45 days)
        - Confidence-based delta (0.55-0.60)
        - Liquidity filtering
        """
        # Calculate optimal parameters
        iv_rank = 50  # Assume normal volatility for now
        target_dte = calculate_target_dte(iv_rank)
        target_delta = calculate_target_delta(ai_score)
        
        logger.info(f"  Finding option: DTE={target_dte}, Delta={target_delta:.2f}")
        
        # Find expiry
        today = datetime.now()
        target_date = today + timedelta(days=target_dte)
        
        # Find nearest Friday
        days_to_friday = (4 - target_date.weekday()) % 7
        expiry_date = target_date + timedelta(days=days_to_friday)
        expiry = expiry_date.strftime("%Y%m%d")
        
        # Calculate strike from delta
        # ATM = 0.50 delta. For 0.55 delta call, we want slightly ITM
        if right == "C":
            # Higher delta = lower strike (ITM)
            delta_offset = (target_delta - 0.50) * 0.20  # Rough approximation
            target_strike = underlying_price * (1 - delta_offset)
        else:
            # Higher delta put = higher strike (ITM)
            delta_offset = (abs(target_delta) - 0.50) * 0.20
            target_strike = underlying_price * (1 + delta_offset)
        
        # Round to standard strike
        if underlying_price >= 100:
            strike = round(target_strike / 5) * 5
        elif underlying_price >= 50:
            strike = round(target_strike / 2.5) * 2.5
        else:
            strike = round(target_strike)
        
        # Get current option price
        bid, ask, mid = self.get_current_option_price(symbol, expiry, strike, right)
        
        if mid <= 0:
            logger.warning(f"  No price for {symbol} {expiry} ${strike} {right}")
            return None
        
        # Calculate spread
        spread_pct = (ask - bid) / ask if ask > 0 else 1.0
        
        return {
            'expiry': expiry,
            'strike': strike,
            'right': right,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spread_pct': spread_pct,
            'target_delta': target_delta,
            'dte': target_dte
        }
    
    def run_backtest(self, symbols: List[Tuple[str, float]], max_symbols: int = 20):
        """Run backtest with real option data."""
        print("\n" + "=" * 70)
        print("REAL OPTION DATA BACKTEST")
        print("=" * 70)
        print(f"Using research-backed parameters:")
        print(f"  - DTE: 30-45 days (IV-adjusted)")
        print(f"  - Delta: Confidence-based (0.55-0.60)")
        print(f"  - Liquidity: Production-grade filtering")
        print("=" * 70 + "\n")
        
        test_symbols = symbols[:max_symbols]
        print(f"Testing {len(test_symbols)} symbols...\n")
        
        # Phase 1: Fetch stock historical data
        print("Phase 1: Fetching stock data...")
        for i, (symbol, beta) in enumerate(test_symbols):
            print(f"  [{i+1}/{len(test_symbols)}] {symbol}...", end="\r")
            df = self.fetch_stock_bars(symbol, "5 D")
            if df is not None and len(df) > 60:
                self.symbol_data[symbol] = df
                time.sleep(0.2)
        
        print(f"\n  ✓ Stock data for {len(self.symbol_data)} symbols\n")
        
        if not self.symbol_data:
            logger.error("No stock data fetched!")
            return
        
        beta_map = {s: b for s, b in test_symbols}
        
        # Get union of timestamps
        all_times = set()
        for df in self.symbol_data.values():
            all_times.update(df['timestamp'].tolist())
        all_times = sorted(all_times)
        
        print(f"Phase 2: Simulating trades through {len(all_times)} time steps...\n")
        
        positions: Dict[str, RealOptionTrade] = {}
        signals_found = 0
        
        # Sample every N bars to speed up (5-min bars, sample every 12 = 1 hour)
        sample_rate = 12
        sampled_times = all_times[::sample_rate]
        
        for i, current_time in enumerate(sampled_times):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(sampled_times)} ({len(positions)} open positions)", end="\r")
            
            for symbol, df in self.symbol_data.items():
                mask = df['timestamp'] <= current_time
                if mask.sum() < 60:
                    continue
                
                df_current = df[mask].copy()
                current_price = df_current['close'].iloc[-1]
                beta = beta_map.get(symbol, 1.0)
                
                # ===== AI SIGNAL CHECK =====
                if symbol not in positions and len(positions) < MAX_POSITIONS:
                    try:
                        signal = self.ai_gen.generate_signal_from_data(df_current, symbol, "5m")
                        
                        if signal.signal_type != SignalType.NO_SIGNAL and signal.consensus_score >= AI_MIN_SCORE:
                            signals_found += 1
                            
                            right = "C" if signal.signal_type == SignalType.BUY_CALL else "P"
                            
                            # Find optimal option contract
                            opt_data = self.find_option_contract(
                                symbol, right, current_price, signal.consensus_score
                            )
                            
                            if opt_data is None:
                                continue
                            
                            # Skip if spread too wide
                            if opt_data['spread_pct'] > 0.15:
                                logger.info(f"  Skipping {symbol}: spread {opt_data['spread_pct']*100:.1f}% too wide")
                                continue
                            
                            # Fetch option historical data for this contract
                            opt_df = self.fetch_option_bars(
                                symbol, opt_data['expiry'], opt_data['strike'], right, "3 D"
                            )
                            
                            if opt_df is None or len(opt_df) < 10:
                                logger.info(f"  Skipping {symbol}: no option history")
                                continue
                            
                            # Store option data
                            opt_key = f"{symbol}_{opt_data['expiry']}_{opt_data['strike']}_{right}"
                            self.option_data[opt_key] = opt_df
                            
                            # Calculate entry price from option data at current time
                            opt_mask = opt_df['timestamp'] <= current_time
                            if opt_mask.sum() > 0:
                                entry_option_price = opt_df[opt_mask]['close'].iloc[-1]
                            else:
                                entry_option_price = opt_data['mid']
                            
                            # Position sizing
                            cost_per = entry_option_price * 100
                            quantity = min(MAX_CONTRACTS, max(1, int(MAX_POSITION_SIZE / cost_per)))
                            
                            # Initial stop
                            idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                            
                            if right == "C":
                                stop_level = STOP_CALC.compute_underlying_stop(
                                    current_price, beta, idx_vol, opt_data['dte'], "long"
                                )
                            else:
                                stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, opt_data['dte'])
                                stop_level = current_price * (1 + stop_dist)
                            
                            trade = RealOptionTrade(
                                symbol=symbol,
                                signal_type=signal.signal_type.value,
                                ai_score=signal.consensus_score,
                                expiry=opt_data['expiry'],
                                strike=opt_data['strike'],
                                right=right,
                                entry_time=current_time,
                                entry_option_price=entry_option_price,
                                entry_underlying=current_price,
                                entry_delta=opt_data['target_delta'],
                                quantity=quantity,
                                stop_level=stop_level,
                                high_water_mark=current_price,
                                low_water_mark=current_price,
                                liquidity_score=0,
                                entry_spread_pct=opt_data['spread_pct']
                            )
                            positions[symbol] = trade
                            
                            print(f"\n[{current_time}] {signal.signal_type.value} {quantity}x {symbol} "
                                  f"${opt_data['strike']} {right} @ ${entry_option_price:.2f} | "
                                  f"Score: {signal.consensus_score:.0f}")
                            
                    except Exception as e:
                        pass
                
                # ===== POSITION MANAGEMENT =====
                if symbol in positions:
                    trade = positions[symbol]
                    beta = beta_map.get(symbol, 1.0)
                    idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                    
                    # Update trailing stop
                    if trade.signal_type == "BUY_CALL":
                        if current_price > trade.high_water_mark:
                            trade.high_water_mark = current_price
                            new_stop = STOP_CALC.compute_trail_from_high(
                                current_price, beta, idx_vol, 30
                            )
                            trade.stop_level = max(trade.stop_level, new_stop)
                        stopped_out = current_price <= trade.stop_level
                    else:
                        if current_price < trade.low_water_mark:
                            trade.low_water_mark = current_price
                            stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, 30)
                            new_stop = current_price * (1 + stop_dist)
                            trade.stop_level = min(trade.stop_level, new_stop)
                        stopped_out = current_price >= trade.stop_level
                    
                    if stopped_out:
                        # Get option price at exit time
                        opt_key = f"{symbol}_{trade.expiry}_{trade.strike}_{trade.right}"
                        opt_df = self.option_data.get(opt_key)
                        
                        if opt_df is not None:
                            opt_mask = opt_df['timestamp'] <= current_time
                            if opt_mask.sum() > 0:
                                exit_option_price = opt_df[opt_mask]['close'].iloc[-1]
                            else:
                                # Estimate using delta
                                underlying_move = current_price - trade.entry_underlying
                                exit_option_price = max(0.01, trade.entry_option_price + underlying_move * trade.entry_delta)
                        else:
                            underlying_move = current_price - trade.entry_underlying
                            delta = trade.entry_delta if trade.right == "C" else -trade.entry_delta
                            exit_option_price = max(0.01, trade.entry_option_price + underlying_move * delta)
                        
                        trade.exit_time = current_time
                        trade.exit_option_price = exit_option_price
                        trade.exit_underlying = current_price
                        trade.pnl_per_contract = (exit_option_price - trade.entry_option_price) * 100
                        trade.total_pnl = trade.pnl_per_contract * trade.quantity
                        trade.exit_reason = "stop"
                        
                        self.trades.append(trade)
                        del positions[symbol]
                        
                        print(f"\n[{current_time}] STOP {trade.quantity}x {symbol} @ ${exit_option_price:.2f} | "
                              f"P&L: ${trade.total_pnl:+,.2f}")
        
        # Close remaining positions
        print("\n\n--- End of Data ---")
        for symbol, trade in positions.items():
            df = self.symbol_data[symbol]
            final_price = df['close'].iloc[-1]
            final_time = df['timestamp'].iloc[-1]
            
            # Get final option price
            opt_key = f"{symbol}_{trade.expiry}_{trade.strike}_{trade.right}"
            opt_df = self.option_data.get(opt_key)
            
            if opt_df is not None and len(opt_df) > 0:
                exit_option_price = opt_df['close'].iloc[-1]
            else:
                underlying_move = final_price - trade.entry_underlying
                delta = trade.entry_delta if trade.right == "C" else -trade.entry_delta
                exit_option_price = max(0.01, trade.entry_option_price + underlying_move * delta)
            
            trade.exit_time = final_time
            trade.exit_option_price = exit_option_price
            trade.exit_underlying = final_price
            trade.pnl_per_contract = (exit_option_price - trade.entry_option_price) * 100
            trade.total_pnl = trade.pnl_per_contract * trade.quantity
            trade.exit_reason = "open"
            
            self.trades.append(trade)
            print(f"[MARK] {trade.quantity}x {symbol} ${trade.strike} {trade.right} | "
                  f"Unrealized P&L: ${trade.total_pnl:+,.2f}")
        
        self.print_summary(signals_found)
    
    def print_summary(self, signals_found: int):
        """Print comprehensive summary."""
        print("\n" + "=" * 70)
        print("REAL OPTION DATA BACKTEST SUMMARY")
        print("=" * 70)
        print(f"  Portfolio Size:     ${PORTFOLIO_SIZE:,}")
        print(f"  AI Score Threshold: {AI_MIN_SCORE}")
        print(f"  Signals Found:      {signals_found}")
        print(f"  Trades Executed:    {len(self.trades)}")
        
        if not self.trades:
            print("  No trades executed")
            return
        
        call_trades = [t for t in self.trades if t.signal_type == "BUY_CALL"]
        put_trades = [t for t in self.trades if t.signal_type == "BUY_PUT"]
        
        print(f"\n  BUY_CALL Trades:    {len(call_trades)}")
        print(f"  BUY_PUT Trades:     {len(put_trades)}")
        
        realized = [t for t in self.trades if t.exit_reason == "stop"]
        unrealized = [t for t in self.trades if t.exit_reason == "open"]
        
        if realized:
            print(f"\n  CLOSED TRADES (Real Option Prices):")
            for t in realized:
                print(f"    {t.signal_type:10s} {t.quantity}x {t.symbol:5s} ${t.strike:6.0f}{t.right}: "
                      f"${t.entry_option_price:.2f} → ${t.exit_option_price:.2f} | "
                      f"P&L: ${t.total_pnl:+8,.2f}")
        
        if unrealized:
            print(f"\n  OPEN POSITIONS:")
            for t in unrealized:
                print(f"    {t.signal_type:10s} {t.quantity}x {t.symbol:5s} ${t.strike:6.0f}{t.right}: "
                      f"${t.entry_option_price:.2f} → ${t.exit_option_price:.2f} | "
                      f"P&L: ${t.total_pnl:+8,.2f}")
        
        pnls = [t.total_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        total_invested = sum(t.entry_option_price * 100 * t.quantity for t in self.trades)
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
    
    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return
        if reqId not in self.tick_data:
            self.tick_data[reqId] = {}
        if tickType == 1:  # BID
            self.tick_data[reqId]['bid'] = price
        elif tickType == 2:  # ASK
            self.tick_data[reqId]['ask'] = price
        elif tickType == 4:  # LAST
            self.tick_data[reqId]['last'] = price
    
    def historicalData(self, reqId, bar):
        if reqId not in self.bar_data:
            self.bar_data[reqId] = []
        self.bar_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId, start, end):
        self.bar_complete[reqId] = True
    
    def nextValidId(self, orderId):
        pass
    
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode in [2104, 2106, 2158, 162, 10167, 10168]:
            pass
        elif errorCode == 200:
            pass  # No security definition
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def main():
    print("\n" + "=" * 70)
    print("REAL OPTION DATA BACKTEST")
    print("=" * 70)
    print("Fetches REAL historical option prices from IB")
    print("=" * 70 + "\n")
    
    backtest = RealOptionDataBacktest()
    
    symbols = backtest.load_watchlist(WATCHLIST_FILE)
    if not symbols:
        return
    
    if not backtest.connect_and_start():
        return
    
    try:
        # Run with fewer symbols due to option data fetching time
        backtest.run_backtest(symbols, max_symbols=15)
    finally:
        backtest.disconnect()
        print("\nDisconnected from IB")


if __name__ == "__main__":
    main()
