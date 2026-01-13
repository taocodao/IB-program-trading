"""
Parameter Sweep for Strategy Optimization
==========================================

Tests multiple parameter combinations to find optimal settings.
Varies: k_aggression, max_trail_pct, duration

Outputs a comparison table of all runs.
"""

import sys
import time
import math
import logging
from datetime import datetime, timedelta
import threading
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from screener.formulas import expected_move, abnormality_score, enhanced_score
from screener.indicators import get_all_indicators
from stop_calculator import StopCalculator

logging.basicConfig(level=logging.WARNING) # Reduce noise
logger = logging.getLogger(__name__)

# ============= Configuration =============
IB_HOST = "127.0.0.1"
IB_PORT = 7497

VIX_LEVEL = 22.0
ABN_THRESHOLD = 2.0
MIN_SCORE = 85
TARGET_EXPIRY = "20260220"
TEST_SYMBOLS = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "GOOG", "META", "SPY", "QQQ"]

SYMBOLS_BETA = {
    "AAPL": 1.2, "NVDA": 1.8, "TSLA": 2.0, "AMD": 1.6, "MSFT": 1.1,
    "AMZN": 1.2, "GOOG": 1.1, "META": 1.3, "SPY": 1.0, "QQQ": 1.2
}

# Parameter Grid: Test ENTRY Strictness
PARAM_GRID = [
    {"k": 0.8, "min_score": 85, "days": "5 D"},
    {"k": 0.8, "min_score": 90, "days": "5 D"},
    {"k": 0.8, "min_score": 95, "days": "5 D"},
    {"k": 1.0, "min_score": 85, "days": "5 D"},
    {"k": 1.0, "min_score": 90, "days": "5 D"},
    {"k": 1.0, "min_score": 95, "days": "5 D"},
]

@dataclass
class TradeResult:
    symbol: str
    entry_price: float
    exit_price: float
    pnl: float
    is_open: bool

class ParameterSweep(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        self.next_req_id = 8000
        self.req_complete = {}
        self.req_data = {}
        self.found_contracts = []
        self.stock_cache = {}  # Cache stock data across runs
        self.option_cache = {}

    def connect_and_start(self):
        self.connect(IB_HOST, IB_PORT, 600)
        threading.Thread(target=self.run, daemon=True).start()
        time.sleep(2)

    def error(self, reqId, errorCode, errorString, *args):
        if errorCode not in [2104, 2106, 2158, 2176]:
            pass # Suppress most errors

    def historicalData(self, reqId, bar):
        if reqId not in self.req_data: self.req_data[reqId] = []
        self.req_data[reqId].append({
            'timestamp': bar.date, 'close': bar.close, 'open': bar.open,
            'high': bar.high, 'low': bar.low, 'volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        self.req_complete[reqId] = True

    def contractDetails(self, reqId, cd):
        self.found_contracts.append(cd.contract)

    def contractDetailsEnd(self, reqId):
        self.req_complete[reqId] = True

    def get_data(self, contract, duration, what="TRADES"):
        req_id = self.next_req_id
        self.next_req_id += 1
        self.req_data[req_id] = []
        self.req_complete[req_id] = False
        self.reqHistoricalData(req_id, contract, "", duration, "1 min", what, 1, 1, False, [])
        start = time.time()
        while not self.req_complete.get(req_id) and time.time() - start < 30:
            time.sleep(0.1)
        if not self.req_data.get(req_id): return None
        df = pd.DataFrame(self.req_data[req_id])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').sort_index()

    def find_option(self, symbol, price):
        req_id = self.next_req_id
        self.next_req_id += 1
        self.found_contracts = []
        self.req_complete[req_id] = False
        c = Contract()
        c.symbol, c.secType, c.exchange, c.currency = symbol, "OPT", "SMART", "USD"
        c.lastTradeDateOrContractMonth, c.right, c.multiplier = TARGET_EXPIRY, "C", "100"
        self.reqContractDetails(req_id, c)
        start = time.time()
        while not self.req_complete.get(req_id) and time.time() - start < 10:
            time.sleep(0.1)
        if not self.found_contracts: return None
        return min(self.found_contracts, key=lambda x: abs(x.strike - price))

    def run_single_test(self, k_agg: float, min_score_thresh: int, duration: str) -> Dict:
        """Run a single backtest with given parameters. Returns stats."""
        stop_calc = StopCalculator(k_aggression=k_agg, min_trail_pct=0.03, max_trail_pct=0.15)
        
        stock_data = {}
        option_data = {}
        contract_map = {}
        
        # Load Data (use cache if available for same duration)
        cache_key = duration
        if cache_key not in self.stock_cache:
            print(f"Loading data for {duration}...")
            self.stock_cache[cache_key] = {}
            self.option_cache[cache_key] = {}
            
            for symbol in TEST_SYMBOLS:
                c = Contract()
                c.symbol, c.secType, c.exchange, c.currency = symbol, "STK", "SMART", "USD"
                df_stk = self.get_data(c, duration)
                if df_stk is None or len(df_stk) < 100: continue
                self.stock_cache[cache_key][symbol] = df_stk
                
                opt = self.find_option(symbol, df_stk.iloc[0]['close'])
                if not opt: continue
                df_opt = self.get_data(opt, duration, "MIDPOINT")
                if df_opt is None: continue
                self.option_cache[cache_key][symbol] = df_opt
                contract_map[symbol] = opt
                time.sleep(0.5)
        
        stock_data = self.stock_cache[cache_key]
        option_data = self.option_cache[cache_key]
        
        # Simulate
        positions = {}
        trades = []
        all_times = sorted(set().union(*[df.index.tolist() for df in stock_data.values()]))
        
        for t in all_times:
            for sym in list(stock_data.keys()):
                if sym not in option_data: continue
                df_stk, df_opt = stock_data[sym], option_data[sym]
                if t not in df_stk.index: continue
                
                try:
                    stk_idx = df_stk.index.get_indexer([t], method='pad')[0]
                    opt_idx = df_opt.index.get_indexer([t], method='pad')[0]
                    if stk_idx == -1 or opt_idx == -1: continue
                    
                    price = df_stk.iloc[stk_idx]['close']
                    opt_price = df_opt.iloc[opt_idx]['close']
                    beta = SYMBOLS_BETA.get(sym, 1.0)
                    
                    # Entry
                    if sym not in positions and len(positions) < 10:
                        if stk_idx < 50: continue
                        subset = df_stk.iloc[:stk_idx+1].tail(100)
                        prev = subset.iloc[0]['close']
                        pct = (price - prev) / prev * 100
                        exp, _ = expected_move(beta, VIX_LEVEL, price)
                        abn = abnormality_score(pct, exp)
                        inds = get_all_indicators(subset)
                        score = enhanced_score(pct, exp, inds['volume_ratio'], inds['macd_state'], inds['rsi'], inds['bb_pos'], "long")
                        
                        if abn >= ABN_THRESHOLD and score >= min_score_thresh:
                            idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                            stop = stop_calc.compute_underlying_stop(price, beta, idx_vol, 21, "long")
                            positions[sym] = {"entry_opt": opt_price, "stop": stop}
                    
                    # Exit
                    elif sym in positions:
                        pos = positions[sym]
                        idx_vol = VIX_LEVEL / 100 / math.sqrt(252)
                        new_stop = stop_calc.compute_underlying_stop(price, beta, idx_vol, 21, "long")
                        if new_stop > pos["stop"]: pos["stop"] = new_stop
                        
                        if price <= pos["stop"]:
                            pnl = (opt_price - pos["entry_opt"]) * 100
                            trades.append(TradeResult(sym, pos["entry_opt"], opt_price, pnl, False))
                            del positions[sym]
                except:
                    continue
        
        # Mark open positions
        for sym, pos in positions.items():
            last = option_data[sym].iloc[-1]['close']
            pnl = (last - pos["entry_opt"]) * 100
            trades.append(TradeResult(sym, pos["entry_opt"], last, pnl, True))
        
        # Stats
        total_pnl = sum(t.pnl for t in trades)
        closed = [t for t in trades if not t.is_open]
        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]
        win_rate = len(winners) / len(closed) * 100 if closed else 0
        
        return {
            "k": k_agg,
            "min_score": min_score_thresh,
            "days": duration,
            "total_pnl": total_pnl,
            "trades": len(trades),
            "closed": len(closed),
            "win_rate": win_rate,
            "avg_win": np.mean([t.pnl for t in winners]) if winners else 0,
            "avg_loss": np.mean([t.pnl for t in losers]) if losers else 0
        }

    def run_sweep(self):
        results = []
        for i, params in enumerate(PARAM_GRID):
            print(f"\n[{i+1}/{len(PARAM_GRID)}] Testing k={params['k']}, min_score={params['min_score']}, days={params['days']}")
            result = self.run_single_test(params['k'], params['min_score'], params['days'])
            results.append(result)
            print(f"   → P&L: ${result['total_pnl']:.0f} | Win Rate: {result['win_rate']:.0f}%")
        
        # Summary Table
        print("\n" + "="*80)
        print("PARAMETER SWEEP RESULTS")
        print("="*80)
        print(f"{'k':<6} | {'MinScore':<10} | {'Days':<6} | {'P&L':<12} | {'Trades':<8} | {'WinRate':<8}")
        print("-"*80)
        for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
            print(f"{r['k']:<6} | {r['min_score']:<10} | {r['days']:<6} | ${r['total_pnl']:<11.0f} | {r['trades']:<8} | {r['win_rate']:<7.0f}%")
        print("="*80)
        
        best = max(results, key=lambda x: x['total_pnl'])
        print(f"\n🏆 BEST: k={best['k']}, min_score={best['min_score']}, days={best['days']} → P&L: ${best['total_pnl']:.0f}")

def main():
    sweep = ParameterSweep()
    sweep.connect_and_start()
    sweep.run_sweep()
    print("\nDone!")

if __name__ == "__main__":
    main()
