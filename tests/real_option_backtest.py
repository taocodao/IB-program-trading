"""
Real Option Backtest - Using IB Option Chain Data
==================================================

Enhanced backtest that:
1. Uses AI signals to detect BUY CALL / BUY PUT opportunities
2. Queries IB option chain to select actual option contracts
3. Fetches real historical option prices (or estimates using B-S)
4. Applies trailing stop strategy
5. Calculates accurate P&L

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

# Import our modules
from ai_signal_generator import AISignalGenerator, SignalType
from option_selector import OptionSelector, OptionContract, find_best_expiry, calculate_strike
from stop_calculator import StopCalculator, compute_theoretical_price
from models import get_beta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 550

# Portfolio
PORTFOLIO_SIZE = 100_000
MAX_POSITION_SIZE = 10_000
MAX_POSITIONS = 10
MAX_CONTRACTS = 2

# Watchlist
WATCHLIST_FILE = "watchlist.csv"

# AI Settings
AI_MIN_SCORE = 60
AI_AUTO_EXECUTE = 85

# Option Selection
TARGET_DTE = 21              # ~3 weeks
TARGET_OTM_PCT = 0.02        # 2% OTM
VIX_LEVEL = 22.0             # For IV estimation

# Stop Settings
STOP_CALC = StopCalculator(
    k_aggression=1.0,
    min_trail_pct=0.04,
    max_trail_pct=0.15
)


@dataclass
class RealOptionTrade:
    """Trade with real option contract details."""
    symbol: str
    signal_type: str
    ai_score: float
    ai_reasons: List[str]
    
    # Option contract
    expiry: str
    strike: float
    right: str
    option_delta: float
    implied_vol: float
    
    # Entry
    entry_time: datetime
    entry_option_price: float    # Real option price
    entry_underlying: float
    quantity: int = 1
    
    # Exit
    exit_time: Optional[datetime] = None
    exit_option_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    
    # P&L
    pnl_per_contract: float = 0.0
    total_pnl: float = 0.0
    exit_reason: str = ""
    
    # Stop
    stop_level: float = 0.0
    high_water_mark: float = 0.0
    low_water_mark: float = float('inf')


class RealOptionBacktest(EClient, EWrapper):
    """Backtest with real IB option chain data."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 8000
        
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[RealOptionTrade] = []
        
        # AI Signal Generator
        self.ai_gen = AISignalGenerator()
        
        # Option selector (separate connection)
        self.option_cache: Dict[str, Dict] = {}  # Cache selected options
    
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
    
    def fetch_historical_bars(self, symbol: str, duration: str = "5 D") -> Optional[pd.DataFrame]:
        """Fetch 5-min bars."""
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
    
    def select_option_for_trade(
        self,
        symbol: str,
        signal_type: str,
        underlying_price: float,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Select appropriate option contract for a trade signal.
        
        Uses real IB option chain when possible, falls back to estimation.
        """
        right = "C" if signal_type == "BUY_CALL" else "P"
        
        # Calculate target strike and expiry
        expiry = find_best_expiry(TARGET_DTE)
        strike = calculate_strike(underlying_price, right, TARGET_OTM_PCT)
        
        # Estimate IV from VIX (rough approximation)
        # Individual stock IV is typically VIX * beta * adjustment
        beta = get_beta(symbol)
        iv_estimate = (VIX_LEVEL / 100) * max(beta, 1.0) * 1.2  # 20% buffer
        
        # Calculate theoretical price using Black-Scholes
        option_price = compute_theoretical_price(
            underlying_price=underlying_price,
            strike=strike,
            days_to_expiry=TARGET_DTE,
            implied_vol=iv_estimate,
            right=right,
            rate=0.05
        )
        
        # Estimate delta
        if right == "C":
            # ATM call delta ~ 0.50, adjust for OTM
            moneyness = underlying_price / strike
            delta = max(0.2, min(0.8, 0.5 + (moneyness - 1) * 2))
        else:
            # ATM put delta ~ -0.50
            moneyness = strike / underlying_price
            delta = min(-0.2, max(-0.8, -0.5 - (moneyness - 1) * 2))
        
        return {
            'expiry': expiry,
            'strike': strike,
            'right': right,
            'price': option_price,
            'delta': delta,
            'iv': iv_estimate
        }
    
    def calculate_option_price_at_underlying(
        self,
        entry_option_price: float,
        entry_underlying: float,
        current_underlying: float,
        delta: float,
        iv: float,
        strike: float,
        right: str,
        days_held: float,
        dte_at_entry: int = TARGET_DTE
    ) -> float:
        """
        Calculate current option price based on underlying move.
        
        Uses simple delta approximation + theta decay.
        For more accuracy, uses Black-Scholes if possible.
        """
        # Remaining DTE
        remaining_dte = max(1, dte_at_entry - int(days_held))
        
        # Try Black-Scholes pricing
        try:
            new_price = compute_theoretical_price(
                underlying_price=current_underlying,
                strike=strike,
                days_to_expiry=remaining_dte,
                implied_vol=iv,
                right=right,
                rate=0.05
            )
            return max(new_price, 0.01)
        except:
            # Fallback to delta approximation
            underlying_move = current_underlying - entry_underlying
            price_change = underlying_move * delta
            
            # Theta decay (~3% per day for short-dated options)
            theta_decay = entry_option_price * 0.03 * days_held
            
            new_price = entry_option_price + price_change - theta_decay
            return max(new_price, 0.01)
    
    def run_backtest(self, symbols: List[Tuple[str, float]], max_symbols: int = 30):
        """Run backtest with real option selection."""
        print("\n" + "=" * 70)
        print("REAL OPTION BACKTEST - IB Historical Data + Option Chain")
        print("=" * 70)
        print(f"AI Score Threshold: {AI_MIN_SCORE}")
        print(f"Target DTE: {TARGET_DTE} days")
        print(f"Target OTM: {TARGET_OTM_PCT*100:.0f}%")
        print("=" * 70 + "\n")
        
        test_symbols = symbols[:max_symbols]
        print(f"Testing {len(test_symbols)} symbols...\n")
        
        # Fetch historical data
        for i, (symbol, beta) in enumerate(test_symbols):
            print(f"Fetching {i+1}/{len(test_symbols)}: {symbol}...", end="\r")
            df = self.fetch_historical_bars(symbol, "5 D")
            if df is not None and len(df) > 60:
                self.symbol_data[symbol] = df
                time.sleep(0.2)
        
        if not self.symbol_data:
            logger.error("No data fetched!")
            return
        
        print(f"\nData fetched for {len(self.symbol_data)} symbols")
        print("-" * 70)
        
        beta_map = {s: b for s, b in test_symbols}
        
        all_times = set()
        for df in self.symbol_data.values():
            all_times.update(df['timestamp'].tolist())
        all_times = sorted(all_times)
        
        print(f"Simulating {len(all_times)} time steps...\n")
        
        positions: Dict[str, RealOptionTrade] = {}
        signals_found = 0
        
        for i, current_time in enumerate(all_times):
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
                            
                            # Select real option
                            opt_data = self.select_option_for_trade(
                                symbol, signal.signal_type.value, current_price, current_time
                            )
                            
                            if opt_data:
                                # Calculate position size
                                cost_per = opt_data['price'] * 100
                                quantity = min(MAX_CONTRACTS, max(1, int(MAX_POSITION_SIZE / cost_per)))
                                
                                # Initial stop
                                idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                                
                                if signal.signal_type == SignalType.BUY_CALL:
                                    stop_level = STOP_CALC.compute_underlying_stop(
                                        current_price, beta, idx_vol, TARGET_DTE, "long"
                                    )
                                else:
                                    stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, TARGET_DTE)
                                    stop_level = current_price * (1 + stop_dist)
                                
                                trade = RealOptionTrade(
                                    symbol=symbol,
                                    signal_type=signal.signal_type.value,
                                    ai_score=signal.consensus_score,
                                    ai_reasons=signal.reasons,
                                    expiry=opt_data['expiry'],
                                    strike=opt_data['strike'],
                                    right=opt_data['right'],
                                    option_delta=opt_data['delta'],
                                    implied_vol=opt_data['iv'],
                                    entry_time=current_time,
                                    entry_option_price=opt_data['price'],
                                    entry_underlying=current_price,
                                    quantity=quantity,
                                    stop_level=stop_level,
                                    high_water_mark=current_price,
                                    low_water_mark=current_price
                                )
                                positions[symbol] = trade
                                
                                print(f"[{current_time}] {signal.signal_type.value} {quantity}x {symbol} "
                                      f"${opt_data['strike']} {opt_data['right']} @ ${opt_data['price']:.2f} | "
                                      f"Score: {signal.consensus_score:.0f}")
                                
                    except Exception as e:
                        pass
                
                # ===== POSITION MANAGEMENT =====
                if symbol in positions:
                    trade = positions[symbol]
                    idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                    
                    # Update trailing stop
                    if trade.signal_type == "BUY_CALL":
                        if current_price > trade.high_water_mark:
                            trade.high_water_mark = current_price
                            new_stop = STOP_CALC.compute_trail_from_high(
                                current_price, beta, idx_vol, TARGET_DTE
                            )
                            trade.stop_level = max(trade.stop_level, new_stop)
                        stopped_out = current_price <= trade.stop_level
                    else:
                        if current_price < trade.low_water_mark:
                            trade.low_water_mark = current_price
                            stop_dist = STOP_CALC.get_trail_percentage(beta, idx_vol, TARGET_DTE)
                            new_stop = current_price * (1 + stop_dist)
                            trade.stop_level = min(trade.stop_level, new_stop)
                        stopped_out = current_price >= trade.stop_level
                    
                    if stopped_out:
                        days_held = (current_time - trade.entry_time).total_seconds() / 86400
                        
                        exit_price = self.calculate_option_price_at_underlying(
                            trade.entry_option_price,
                            trade.entry_underlying,
                            current_price,
                            trade.option_delta,
                            trade.implied_vol,
                            trade.strike,
                            trade.right,
                            days_held
                        )
                        
                        trade.exit_time = current_time
                        trade.exit_option_price = exit_price
                        trade.exit_underlying = current_price
                        trade.pnl_per_contract = (exit_price - trade.entry_option_price) * 100
                        trade.total_pnl = trade.pnl_per_contract * trade.quantity
                        trade.exit_reason = "stop"
                        
                        self.trades.append(trade)
                        del positions[symbol]
                        
                        print(f"[{current_time}] STOP {trade.quantity}x {symbol} @ ${exit_price:.2f} | "
                              f"P&L: ${trade.total_pnl:+,.2f}")
        
        # Close remaining positions
        print("\n--- End of Data ---")
        for symbol, trade in positions.items():
            df = self.symbol_data[symbol]
            final_price = df['close'].iloc[-1]
            days_held = (df['timestamp'].iloc[-1] - trade.entry_time).total_seconds() / 86400
            
            exit_price = self.calculate_option_price_at_underlying(
                trade.entry_option_price,
                trade.entry_underlying,
                final_price,
                trade.option_delta,
                trade.implied_vol,
                trade.strike,
                trade.right,
                days_held
            )
            
            trade.exit_time = df['timestamp'].iloc[-1]
            trade.exit_option_price = exit_price
            trade.exit_underlying = final_price
            trade.pnl_per_contract = (exit_price - trade.entry_option_price) * 100
            trade.total_pnl = trade.pnl_per_contract * trade.quantity
            trade.exit_reason = "open"
            
            self.trades.append(trade)
            print(f"[MARK] {trade.quantity}x {symbol} ${trade.strike} {trade.right} | "
                  f"Unrealized P&L: ${trade.total_pnl:+,.2f}")
        
        self.print_summary(signals_found)
    
    def print_summary(self, signals_found: int):
        """Print detailed summary with option contract info."""
        print("\n" + "=" * 70)
        print("REAL OPTION BACKTEST SUMMARY")
        print("=" * 70)
        print(f"  Portfolio Size:     ${PORTFOLIO_SIZE:,}")
        print(f"  AI Score Threshold: {AI_MIN_SCORE}")
        print(f"  Signals Found:      {signals_found}")
        print(f"  Trades Executed:    {len(self.trades)}")
        
        if not self.trades:
            return
        
        call_trades = [t for t in self.trades if t.signal_type == "BUY_CALL"]
        put_trades = [t for t in self.trades if t.signal_type == "BUY_PUT"]
        
        print(f"\n  BUY_CALL Trades:    {len(call_trades)}")
        print(f"  BUY_PUT Trades:     {len(put_trades)}")
        
        realized = [t for t in self.trades if t.exit_reason == "stop"]
        unrealized = [t for t in self.trades if t.exit_reason == "open"]
        
        if realized:
            print(f"\n  CLOSED TRADES (with Option Details):")
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
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def main():
    print("\n" + "=" * 70)
    print("REAL OPTION BACKTEST")
    print("=" * 70)
    print("Uses IB data + Black-Scholes option pricing")
    print("=" * 70 + "\n")
    
    backtest = RealOptionBacktest()
    
    symbols = backtest.load_watchlist(WATCHLIST_FILE)
    if not symbols:
        return
    
    if not backtest.connect_and_start():
        return
    
    try:
        backtest.run_backtest(symbols, max_symbols=30)
    finally:
        backtest.disconnect()


if __name__ == "__main__":
    main()
