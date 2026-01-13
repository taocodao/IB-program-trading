"""
Backtesting Engine for Asymmetric Stop-Loss Strategy
=====================================================

Replays historical price data through the stop-loss logic
to verify strategy performance before live trading.

Key Features:
- No IB connection required (pure simulation)
- Uses same StopCalculator as live system
- Tracks MFE/MAE, win rate, profit factor
- Multiple scenario testing
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models import VolatilityTracker, get_beta
from stop_calculator import StopCalculator

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a completed trade during backtest."""
    
    symbol: str
    expiry: str
    strike: float
    right: str  # "C" or "P"
    beta: float
    
    entry_time: datetime
    entry_price: float
    entry_underlying: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    exit_reason: str = ""  # "stop_triggered", "end_of_day", "manual"
    
    realized_pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Excursions
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def close(self, exit_price: float, exit_underlying: float, 
              exit_time: datetime, reason: str):
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_underlying = exit_underlying
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # P&L per contract (multiply by 100)
        self.realized_pnl = (exit_price - self.entry_price) * 100
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100


@dataclass
class BacktestPosition:
    """Position being tracked during backtest."""
    
    # Static info
    symbol: str
    expiry: str
    strike: float
    right: str
    beta: float
    
    # Entry
    entry_price: float
    entry_underlying: float
    entry_time: datetime
    
    # Current state
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_underlying: float = 0.0
    current_time: Optional[datetime] = None
    
    # Stop tracking
    underlying_stop_level: Optional[float] = None
    underlying_high: float = 0.0
    stop_initialized: bool = False
    
    # Exit state
    exit_triggered: bool = False
    
    # Excursions
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    
    def update_excursions(self, mid_price: float):
        """Track best and worst unrealized P&L."""
        unrealized = (mid_price - self.entry_price) * 100
        self.max_favorable = max(self.max_favorable, unrealized)
        self.max_adverse = min(self.max_adverse, unrealized)


class BacktestEngine:
    """
    Backtesting engine for asymmetric stop-loss strategy.
    
    Usage:
        engine = BacktestEngine(vix_level=20.0)
        key = engine.open_position(symbol="SPY", ...)
        
        for price in historical_prices:
            engine.update_position(key, bid, ask, underlying, time)
        
        engine.print_summary()
    """
    
    def __init__(
        self,
        k_aggression: float = 1.0,
        min_trail_pct: float = 0.04,
        max_trail_pct: float = 0.40,
        vix_level: float = 20.0
    ):
        self.stop_calc = StopCalculator(
            k_aggression=k_aggression,
            min_trail_pct=min_trail_pct,
            max_trail_pct=max_trail_pct
        )
        
        self.vol_tracker = VolatilityTracker()
        self.vol_tracker.update_vix(vix_level)
        
        self.positions: Dict[str, BacktestPosition] = {}
        self.closed_trades: List[BacktestTrade] = []
        
        self.current_time: Optional[datetime] = None
    
    def open_position(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        entry_price: float,
        entry_underlying: float,
        beta: Optional[float] = None,
        entry_time: Optional[datetime] = None
    ) -> str:
        """
        Open a new position for backtesting.
        
        Returns position key for tracking.
        """
        key = f"{symbol}_{expiry}_{strike}_{right}"
        
        if beta is None:
            beta = get_beta(symbol)
        
        pos = BacktestPosition(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            beta=beta,
            entry_price=entry_price,
            entry_underlying=entry_underlying,
            entry_time=entry_time or self.current_time or datetime.now(),
            underlying_high=entry_underlying,
        )
        
        self.positions[key] = pos
        
        logger.info(
            f"OPEN: {symbol} {expiry} ${strike}{right} @ ${entry_price:.2f} "
            f"(underlying ${entry_underlying:.2f}, β={beta:.2f})"
        )
        
        return key
    
    def update_position(
        self,
        key: str,
        bid: float,
        ask: float,
        underlying: float,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Update position with new market data.
        
        Returns True if stop was triggered.
        """
        if key not in self.positions:
            return False
        
        pos = self.positions[key]
        pos.current_bid = bid
        pos.current_ask = ask
        pos.current_underlying = underlying
        pos.current_time = current_time or self.current_time
        
        # Update excursions
        mid = (bid + ask) / 2
        pos.update_excursions(mid)
        
        # Calculate DTE
        dte = self._days_to_expiry(pos.expiry, current_time)
        index_vol = self.vol_tracker.get_daily_vol_pct()
        
        # Initialize or update stop
        if not pos.stop_initialized:
            stop = self.stop_calc.compute_underlying_stop(
                entry_price=pos.entry_underlying,
                beta=pos.beta,
                index_vol_pct=index_vol,
                days_to_expiry=dte
            )
            pos.underlying_stop_level = stop
            pos.stop_initialized = True
            
            trail_pct = self.stop_calc.get_trail_percentage(pos.beta, index_vol, dte)
            logger.info(
                f"  Stop set: ${stop:.2f} ({trail_pct*100:.1f}% trail)"
            )
        
        # Update trailing high and stop
        if underlying > pos.underlying_high:
            pos.underlying_high = underlying
            
            new_stop = self.stop_calc.compute_trail_from_high(
                underlying_high=pos.underlying_high,
                beta=pos.beta,
                index_vol_pct=index_vol,
                days_to_expiry=dte
            )
            
            if new_stop > pos.underlying_stop_level:
                old_stop = pos.underlying_stop_level
                pos.underlying_stop_level = new_stop
                logger.debug(
                    f"  Stop trailed: ${old_stop:.2f} → ${new_stop:.2f}"
                )
        
        # Check trigger
        if underlying <= pos.underlying_stop_level and not pos.exit_triggered:
            pos.exit_triggered = True
            
            logger.warning(
                f"*** STOP TRIGGERED *** {pos.symbol}: "
                f"underlying ${underlying:.2f} <= stop ${pos.underlying_stop_level:.2f}"
            )
            
            # Close at bid
            self._close_position(key, bid, underlying, current_time, "stop_triggered")
            return True
        
        return False
    
    def _close_position(
        self,
        key: str,
        exit_price: float,
        exit_underlying: float,
        exit_time: Optional[datetime],
        reason: str
    ):
        """Close a position and record the trade."""
        if key not in self.positions:
            return
        
        pos = self.positions.pop(key)
        
        trade = BacktestTrade(
            symbol=pos.symbol,
            expiry=pos.expiry,
            strike=pos.strike,
            right=pos.right,
            beta=pos.beta,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            entry_underlying=pos.entry_underlying,
            max_favorable_excursion=pos.max_favorable,
            max_adverse_excursion=pos.max_adverse,
        )
        
        trade.close(exit_price, exit_underlying, 
                   exit_time or self.current_time or datetime.now(), reason)
        
        self.closed_trades.append(trade)
        
        logger.info(
            f"CLOSE: {pos.symbol} @ ${exit_price:.2f}, "
            f"P&L: ${trade.realized_pnl:+.2f} ({reason})"
        )
    
    def close_all(self, exit_price_map: Optional[Dict[str, float]] = None):
        """Close all open positions (end of day)."""
        for key in list(self.positions.keys()):
            pos = self.positions[key]
            exit_price = pos.current_bid
            if exit_price_map and key in exit_price_map:
                exit_price = exit_price_map[key]
            
            self._close_position(
                key, exit_price, pos.current_underlying,
                self.current_time, "end_of_day"
            )
    
    def _days_to_expiry(self, expiry: str, current_time: Optional[datetime] = None) -> int:
        """Calculate days until expiry."""
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            now = current_time or self.current_time or datetime.now()
            return max(0, (exp_date - now).days)
        except:
            return 30  # Default
    
    def get_summary(self) -> Dict:
        """Get backtest summary statistics."""
        if not self.closed_trades:
            return {
                'trades': 0, 'winners': 0, 'losers': 0,
                'win_rate': 0.0, 'total_pnl': 0.0,
                'avg_winner': 0.0, 'avg_loser': 0.0,
                'profit_factor': 0.0, 'win_loss_ratio': 0.0,
            }
        
        pnls = [t.realized_pnl for t in self.closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        win_rate = len(winners) / len(pnls) if pnls else 0
        total_pnl = sum(pnls)
        avg_winner = sum(winners) / len(winners) if winners else 0
        avg_loser = abs(sum(losers) / len(losers)) if losers else 0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        win_loss_ratio = avg_winner / avg_loser if avg_loser > 0 else float('inf')
        
        return {
            'trades': len(pnls),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
        }
    
    def print_summary(self):
        """Print backtest summary."""
        s = self.get_summary()
        
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Total Trades:    {s['trades']}")
        print(f"  Winners:         {s['winners']} ({s['win_rate']*100:.1f}%)")
        print(f"  Losers:          {s['losers']}")
        print(f"  Total P&L:       ${s['total_pnl']:+,.2f}")
        print(f"  Avg Winner:      ${s['avg_winner']:+,.2f}")
        print(f"  Avg Loser:       ${s['avg_loser']:,.2f}")
        print(f"  Win/Loss Ratio:  {s['win_loss_ratio']:.2f}x")
        print(f"  Profit Factor:   {s['profit_factor']:.2f}")
        print("=" * 60)
        
        # Target metrics check
        if s['trades'] > 0:
            print("\nTarget Metrics Check:")
            wl_ok = "✓" if s['win_loss_ratio'] >= 2.0 else "✗"
            pf_ok = "✓" if s['profit_factor'] >= 1.5 else "✗"
            print(f"  {wl_ok} Win/Loss Ratio >= 2.0: {s['win_loss_ratio']:.2f}")
            print(f"  {pf_ok} Profit Factor >= 1.5: {s['profit_factor']:.2f}")


def run_simple_backtest():
    """
    Simple backtest example: single SPY option trade.
    """
    print("\n" + "=" * 60)
    print("BACKTEST: Single SPY Option Trade")
    print("=" * 60)
    
    engine = BacktestEngine(
        k_aggression=1.0,
        min_trail_pct=0.04,
        vix_level=20.0
    )
    
    # Entry
    entry_time = datetime(2026, 1, 9, 10, 30)
    engine.current_time = entry_time
    
    key = engine.open_position(
        symbol="SPY",
        expiry="20260220",
        strike=585.0,
        right="C",
        entry_price=12.00,
        entry_underlying=585.0,
        beta=1.0,
        entry_time=entry_time
    )
    
    # Simulate price movement: SPY drifts down, hits stop
    prices = [
        (datetime(2026, 1, 9, 10, 45), 583.0, 11.50),
        (datetime(2026, 1, 9, 11, 00), 580.0, 10.80),
        (datetime(2026, 1, 9, 11, 30), 575.0, 9.50),
        (datetime(2026, 1, 9, 12, 00), 570.0, 8.20),
        (datetime(2026, 1, 9, 12, 30), 565.0, 7.00),
        (datetime(2026, 1, 9, 13, 00), 560.0, 5.80),  # Stop zone (~4% down)
        (datetime(2026, 1, 9, 13, 30), 555.0, 4.50),  # HIT STOP
    ]
    
    print("\n--- Price Simulation ---")
    
    for sim_time, underlying, mid_price in prices:
        engine.current_time = sim_time
        
        bid = mid_price * 0.99
        ask = mid_price * 1.01
        
        print(f"{sim_time.strftime('%H:%M')} | SPY: ${underlying:.2f} | Option: ${bid:.2f}/{ask:.2f}")
        
        triggered = engine.update_position(key, bid, ask, underlying, sim_time)
        
        if triggered:
            break
    
    engine.print_summary()
    
    return engine


def run_multiple_scenarios():
    """
    Test multiple market scenarios.
    """
    print("\n" + "=" * 60)
    print("BACKTEST: Multiple Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Gap Down (Worst Case)',
            'prices': [
                (585.0, 12.00),
                (555.0, 5.00),  # Gapped down 5%
            ]
        },
        {
            'name': 'Slow Drift Down (Stop Hit)',
            'prices': [
                (585.0, 12.00),
                (580.0, 11.00),
                (575.0, 10.00),
                (570.0, 9.00),
                (565.0, 8.00),
                (560.0, 7.00),  # ~4.3% down
                (555.0, 6.00),  # HIT STOP
            ]
        },
        {
            'name': 'V-Bounce (Survive Dip)',
            'prices': [
                (585.0, 12.00),
                (575.0, 10.00),  # -1.7%
                (570.0, 9.00),   # -2.6%
                (580.0, 11.00),  # Bounce back
                (590.0, 13.50),  # New high
                (595.0, 15.00),  # Higher
            ]
        },
        {
            'name': 'Strong Rally (Max Profit)',
            'prices': [
                (585.0, 12.00),
                (590.0, 13.00),
                (600.0, 15.00),
                (610.0, 18.00),
                (620.0, 22.00),
            ]
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        engine = BacktestEngine(k_aggression=1.0, vix_level=20.0)
        entry_time = datetime(2026, 1, 9, 10, 30)
        
        key = engine.open_position(
            symbol="SPY",
            expiry="20260220",
            strike=585.0,
            right="C",
            entry_price=scenario['prices'][0][1],
            entry_underlying=scenario['prices'][0][0],
            beta=1.0,
            entry_time=entry_time
        )
        
        for idx, (underlying, option_mid) in enumerate(scenario['prices']):
            sim_time = entry_time + timedelta(minutes=idx*30)
            engine.current_time = sim_time
            
            bid = option_mid * 0.99
            ask = option_mid * 1.01
            
            engine.update_position(key, bid, ask, underlying, sim_time)
            
            if key not in engine.positions:
                break
        
        # Close remaining
        if key in engine.positions:
            pos = engine.positions[key]
            engine._close_position(
                key, pos.current_bid, pos.current_underlying,
                engine.current_time, "end_of_day"
            )
        
        if engine.closed_trades:
            trade = engine.closed_trades[0]
            results.append({
                'scenario': scenario['name'],
                'pnl': trade.realized_pnl,
                'reason': trade.exit_reason,
            })
            print(f"  Result: P&L ${trade.realized_pnl:+.2f} ({trade.exit_reason})")
    
    # Summary
    print("\n" + "-" * 60)
    print("SCENARIO SUMMARY")
    print("-" * 60)
    
    for r in results:
        print(f"  {r['scenario']:30s} | P&L: ${r['pnl']:+8.2f} | {r['reason']}")
    
    avg_pnl = sum(r['pnl'] for r in results) / len(results) if results else 0
    print(f"\n  Average P&L: ${avg_pnl:+.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run backtests
    run_simple_backtest()
    run_multiple_scenarios()
    
    print("\n✓ Backtest complete")
