"""
Backtest Validation Suite
=========================

Runs multiple backtests with different:
1. Symbol sets (different sectors, different volatility)
2. Time periods (different durations)
3. Score thresholds (60 vs 70 vs 80)

Purpose: Validate if 44% return is realistic or an anomaly.

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
import pandas as pd
import numpy as np

from ai_signal_generator import AISignalGenerator, SignalType
from stop_calculator import StopCalculator
from models import get_beta

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Test Configurations =============

# Different symbol groups to test
SYMBOL_GROUPS = {
    "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B"],
    "tech_volatile": ["AMD", "COIN", "MARA", "RIOT", "PLTR", "SOFI", "AFRM", "UPST"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V"],
    "healthcare": ["UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR"],
    "industrials": ["CAT", "DE", "BA", "GE", "HON", "UPS", "FDX", "RTX"],
    "mixed_20": ["AAPL", "MSFT", "AMD", "COIN", "JPM", "BA", "TSLA", "NVDA", 
                  "META", "GOOGL", "AMZN", "NFLX", "DIS", "WMT", "HD", "LOW",
                  "COST", "TGT", "NKE", "SBUX"],
}

# Different durations to test
DURATIONS = ["3 D", "5 D", "10 D"]

# Different score thresholds
SCORE_THRESHOLDS = [60, 70, 80]


@dataclass
class BacktestResult:
    """Single backtest result."""
    group_name: str
    duration: str
    score_threshold: int
    
    total_signals: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    total_invested: float = 0.0
    return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    call_trades: int = 0
    put_trades: int = 0
    winners: int = 0
    losers: int = 0


class ValidationBacktester(EClient, EWrapper):
    """Run multiple backtests for validation."""
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        self.next_req_id = 10000
        
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        
        self.ai_gen = AISignalGenerator()
        self.stop_calc = StopCalculator(k_aggression=1.0, min_trail_pct=0.04, max_trail_pct=0.15)
        
        self.results: List[BacktestResult] = []
    
    def connect_and_start(self) -> bool:
        self.connect("127.0.0.1", 7497, 700)
        
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            print("Failed to connect to IB")
            return False
        
        self.connected = True
        print("✓ Connected to IB Gateway\n")
        return True
    
    def fetch_bars(self, symbol: str, duration: str) -> Optional[pd.DataFrame]:
        """Fetch historical bars."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.bar_data[req_id] = []
        self.bar_complete[req_id] = False
        
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
    
    def run_single_backtest(
        self, 
        symbols: List[str], 
        duration: str, 
        score_threshold: int,
        group_name: str
    ) -> BacktestResult:
        """Run a single backtest configuration."""
        result = BacktestResult(
            group_name=group_name,
            duration=duration,
            score_threshold=score_threshold
        )
        
        # Fetch data for all symbols
        symbol_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.fetch_bars(symbol, duration)
            if df is not None and len(df) > 60:
                symbol_data[symbol] = df
            time.sleep(0.15)
        
        if not symbol_data:
            return result
        
        # Get all timestamps
        all_times = set()
        for df in symbol_data.values():
            all_times.update(df['timestamp'].tolist())
        all_times = sorted(all_times)
        
        # Track positions and trades
        positions: Dict[str, dict] = {}
        trades = []
        
        VIX_LEVEL = 22.0
        MAX_POSITIONS = 10
        
        # Option pricing constants
        OPT_PCT_CALL = 0.025
        OPT_PCT_PUT = 0.020
        DELTA_CALL = 0.50
        DELTA_PUT = -0.50
        THETA_DAY = 0.08
        SLIPPAGE = 0.02
        DTE = 21
        
        for current_time in all_times:
            for symbol, df in symbol_data.items():
                mask = df['timestamp'] <= current_time
                if mask.sum() < 60:
                    continue
                
                df_current = df[mask].copy()
                current_price = df_current['close'].iloc[-1]
                beta = get_beta(symbol)
                
                # Signal check
                if symbol not in positions and len(positions) < MAX_POSITIONS:
                    try:
                        signal = self.ai_gen.generate_signal_from_data(df_current, symbol, "5m")
                        
                        if signal.signal_type != SignalType.NO_SIGNAL and signal.consensus_score >= score_threshold:
                            result.total_signals += 1
                            
                            if signal.signal_type == SignalType.BUY_CALL:
                                opt_price = current_price * OPT_PCT_CALL
                                delta = DELTA_CALL
                                result.call_trades += 1
                            else:
                                opt_price = current_price * OPT_PCT_PUT
                                delta = DELTA_PUT
                                result.put_trades += 1
                            
                            idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                            
                            if signal.signal_type == SignalType.BUY_CALL:
                                stop_level = self.stop_calc.compute_underlying_stop(
                                    current_price, beta, idx_vol, DTE, "long"
                                )
                            else:
                                stop_dist = self.stop_calc.get_trail_percentage(beta, idx_vol, DTE)
                                stop_level = current_price * (1 + stop_dist)
                            
                            positions[symbol] = {
                                'signal_type': signal.signal_type.value,
                                'entry_time': current_time,
                                'entry_price': opt_price,
                                'entry_underlying': current_price,
                                'delta': delta,
                                'stop_level': stop_level,
                                'high_water': current_price,
                                'low_water': current_price
                            }
                            
                    except:
                        pass
                
                # Position management
                if symbol in positions:
                    pos = positions[symbol]
                    idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                    
                    if pos['signal_type'] == "BUY_CALL":
                        if current_price > pos['high_water']:
                            pos['high_water'] = current_price
                            new_stop = self.stop_calc.compute_trail_from_high(
                                current_price, beta, idx_vol, DTE
                            )
                            pos['stop_level'] = max(pos['stop_level'], new_stop)
                        stopped_out = current_price <= pos['stop_level']
                    else:
                        if current_price < pos['low_water']:
                            pos['low_water'] = current_price
                            stop_dist = self.stop_calc.get_trail_percentage(beta, idx_vol, DTE)
                            new_stop = current_price * (1 + stop_dist)
                            pos['stop_level'] = min(pos['stop_level'], new_stop)
                        stopped_out = current_price >= pos['stop_level']
                    
                    if stopped_out:
                        underlying_move = current_price - pos['entry_underlying']
                        days_held = (current_time - pos['entry_time']).total_seconds() / 86400
                        theta_decay = days_held * THETA_DAY
                        
                        raw_change = underlying_move * pos['delta']
                        exit_price = max(pos['entry_price'] + raw_change - theta_decay - SLIPPAGE * 2, 0.01)
                        
                        pnl = (exit_price - pos['entry_price']) * 100 * 2  # 2 contracts
                        trades.append(pnl)
                        
                        del positions[symbol]
        
        # Close remaining positions
        for symbol, pos in positions.items():
            df = symbol_data[symbol]
            final_price = df['close'].iloc[-1]
            underlying_move = final_price - pos['entry_underlying']
            days_held = (df['timestamp'].iloc[-1] - pos['entry_time']).total_seconds() / 86400
            theta_decay = days_held * THETA_DAY
            
            raw_change = underlying_move * pos['delta']
            exit_price = max(pos['entry_price'] + raw_change - theta_decay - SLIPPAGE * 2, 0.01)
            
            pnl = (exit_price - pos['entry_price']) * 100 * 2
            trades.append(pnl)
        
        # Calculate results
        result.total_trades = len(trades)
        
        if trades:
            result.total_pnl = sum(trades)
            winners = [p for p in trades if p > 0]
            losers = [p for p in trades if p < 0]
            
            result.winners = len(winners)
            result.losers = len(losers)
            result.win_rate = len(winners) / len(trades) * 100
            
            # Estimate total invested (rough)
            result.total_invested = result.total_trades * 500  # ~$5 per contract * 100 * 2
            
            if result.total_invested > 0:
                result.return_pct = result.total_pnl / result.total_invested * 100
            
            if losers and sum(losers) != 0:
                result.profit_factor = abs(sum(winners)) / abs(sum(losers))
        
        return result
    
    def run_validation_suite(self):
        """Run full validation suite."""
        print("=" * 80)
        print("BACKTEST VALIDATION SUITE")
        print("=" * 80)
        print("Testing multiple scenarios to validate performance...")
        print("=" * 80 + "\n")
        
        total_tests = len(SYMBOL_GROUPS) * len(DURATIONS) * len(SCORE_THRESHOLDS)
        test_num = 0
        
        for group_name, symbols in SYMBOL_GROUPS.items():
            for duration in DURATIONS:
                for threshold in SCORE_THRESHOLDS:
                    test_num += 1
                    print(f"[{test_num}/{total_tests}] {group_name} | {duration} | Score≥{threshold}...", end=" ")
                    
                    result = self.run_single_backtest(
                        symbols=symbols,
                        duration=duration,
                        score_threshold=threshold,
                        group_name=group_name
                    )
                    
                    self.results.append(result)
                    
                    if result.total_trades > 0:
                        print(f"Trades: {result.total_trades}, Return: {result.return_pct:+.1f}%, "
                              f"Win: {result.win_rate:.0f}%, PF: {result.profit_factor:.2f}")
                    else:
                        print("No trades")
        
        self.print_validation_summary()
    
    def print_validation_summary(self):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        # Filter results with trades
        valid_results = [r for r in self.results if r.total_trades > 0]
        
        if not valid_results:
            print("No valid results!")
            return
        
        # Overall statistics
        returns = [r.return_pct for r in valid_results]
        win_rates = [r.win_rate for r in valid_results]
        profit_factors = [r.profit_factor for r in valid_results if r.profit_factor > 0]
        
        print(f"\nTotal Tests with Trades: {len(valid_results)}")
        print(f"\n{'─' * 50}")
        print("RETURN % STATISTICS:")
        print(f"  Mean Return:   {np.mean(returns):+.1f}%")
        print(f"  Median Return: {np.median(returns):+.1f}%")
        print(f"  Std Dev:       {np.std(returns):.1f}%")
        print(f"  Min Return:    {min(returns):+.1f}%")
        print(f"  Max Return:    {max(returns):+.1f}%")
        
        print(f"\n{'─' * 50}")
        print("WIN RATE STATISTICS:")
        print(f"  Mean Win Rate:   {np.mean(win_rates):.1f}%")
        print(f"  Median Win Rate: {np.median(win_rates):.1f}%")
        print(f"  Min Win Rate:    {min(win_rates):.0f}%")
        print(f"  Max Win Rate:    {max(win_rates):.0f}%")
        
        if profit_factors:
            print(f"\n{'─' * 50}")
            print("PROFIT FACTOR STATISTICS:")
            print(f"  Mean PF:   {np.mean(profit_factors):.2f}")
            print(f"  Median PF: {np.median(profit_factors):.2f}")
            print(f"  Min PF:    {min(profit_factors):.2f}")
            print(f"  Max PF:    {max(profit_factors):.2f}")
        
        # Breakdown by group
        print(f"\n{'─' * 50}")
        print("BREAKDOWN BY SYMBOL GROUP:")
        for group_name in SYMBOL_GROUPS.keys():
            group_results = [r for r in valid_results if r.group_name == group_name]
            if group_results:
                avg_return = np.mean([r.return_pct for r in group_results])
                avg_win = np.mean([r.win_rate for r in group_results])
                print(f"  {group_name:15s}: Avg Return {avg_return:+6.1f}%, Avg Win Rate {avg_win:.0f}%")
        
        # Breakdown by duration
        print(f"\n{'─' * 50}")
        print("BREAKDOWN BY DURATION:")
        for duration in DURATIONS:
            dur_results = [r for r in valid_results if r.duration == duration]
            if dur_results:
                avg_return = np.mean([r.return_pct for r in dur_results])
                avg_win = np.mean([r.win_rate for r in dur_results])
                print(f"  {duration:8s}: Avg Return {avg_return:+6.1f}%, Avg Win Rate {avg_win:.0f}%")
        
        # Breakdown by threshold
        print(f"\n{'─' * 50}")
        print("BREAKDOWN BY SCORE THRESHOLD:")
        for threshold in SCORE_THRESHOLDS:
            thr_results = [r for r in valid_results if r.score_threshold == threshold]
            if thr_results:
                avg_return = np.mean([r.return_pct for r in thr_results])
                avg_win = np.mean([r.win_rate for r in thr_results])
                trades = sum(r.total_trades for r in thr_results)
                print(f"  Score≥{threshold}: Avg Return {avg_return:+6.1f}%, Avg Win Rate {avg_win:.0f}%, "
                      f"Total Trades: {trades}")
        
        # Best and worst scenarios
        print(f"\n{'─' * 50}")
        print("BEST SCENARIOS (Top 5):")
        sorted_results = sorted(valid_results, key=lambda x: x.return_pct, reverse=True)[:5]
        for r in sorted_results:
            print(f"  {r.group_name:15s} | {r.duration} | ≥{r.score_threshold}: "
                  f"Return {r.return_pct:+.1f}%, {r.total_trades} trades")
        
        print(f"\nWORST SCENARIOS (Bottom 5):")
        sorted_results = sorted(valid_results, key=lambda x: x.return_pct)[:5]
        for r in sorted_results:
            print(f"  {r.group_name:15s} | {r.duration} | ≥{r.score_threshold}: "
                  f"Return {r.return_pct:+.1f}%, {r.total_trades} trades")
        
        # Confidence assessment
        print(f"\n{'─' * 50}")
        print("CONFIDENCE ASSESSMENT:")
        positive_returns = len([r for r in returns if r > 0])
        negative_returns = len([r for r in returns if r < 0])
        
        print(f"  Positive Return Scenarios: {positive_returns}/{len(returns)} ({positive_returns/len(returns)*100:.0f}%)")
        print(f"  Negative Return Scenarios: {negative_returns}/{len(returns)} ({negative_returns/len(returns)*100:.0f}%)")
        
        if np.mean(returns) > 20:
            print("\n  ⚠️  WARNING: Average returns seem HIGH. Consider:")
            print("      - Survivorship bias (only testing liquid stocks)")
            print("      - Short test period (may not capture bear markets)")
            print("      - Simulated option pricing (real spreads may be wider)")
            print("      - Paper trading recommended before live")
        
        print("\n" + "=" * 80)
    
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
        if errorCode not in [2104, 2106, 2158, 162, 10167, 10168]:
            pass


def main():
    validator = ValidationBacktester()
    
    if not validator.connect_and_start():
        return
    
    try:
        validator.run_validation_suite()
    finally:
        validator.disconnect()
        print("\nDisconnected from IB")


if __name__ == "__main__":
    main()
