"""
Asymmetry Statistics Tracking
=============================

Tracks P&L distribution to verify asymmetric outcomes:
- Win rate (target: 40-60%)
- Avg winner vs avg loser (target: winner >= 2x loser)
- Profit factor (target: >= 1.5)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class ClosedTrade:
    """Record of a closed option trade."""
    symbol: str
    expiry: str
    strike: float
    right: str  # "C" or "P"
    
    entry_time: datetime
    exit_time: datetime
    
    entry_price: float  # Option premium at entry
    exit_price: float   # Option premium at exit
    quantity: int
    
    realized_pnl: float  # Dollar P&L
    pnl_pct: float       # Percentage P&L
    
    # Context at entry
    underlying_entry: float
    underlying_exit: float
    entry_beta: float
    entry_vix: float
    
    # Excursions
    max_favorable_excursion: float = 0.0  # Highest unrealized P&L
    max_adverse_excursion: float = 0.0    # Lowest unrealized P&L
    
    # Exit reason
    exit_reason: str = "stop"  # "stop", "expiry", "manual"


@dataclass
class AsymmetryStats:
    """
    Tracks P&L statistics to verify asymmetric outcomes.
    
    Target metrics:
    - Win rate: 40-60%
    - Avg winner >= 2x avg loser
    - Profit factor >= 1.5
    """
    
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    
    def add_trade(self, trade: ClosedTrade):
        """Add a closed trade to statistics."""
        self.closed_trades.append(trade)
    
    def add_pnl(self, pnl: float, symbol: str = ""):
        """Quick add just P&L amount (for simple tracking)."""
        trade = ClosedTrade(
            symbol=symbol, expiry="", strike=0, right="C",
            entry_time=datetime.now(), exit_time=datetime.now(),
            entry_price=0, exit_price=0, quantity=1,
            realized_pnl=pnl, pnl_pct=0,
            underlying_entry=0, underlying_exit=0,
            entry_beta=1.0, entry_vix=20.0
        )
        self.closed_trades.append(trade)
    
    @property
    def pnls(self) -> List[float]:
        """Get list of realized P&Ls."""
        return [t.realized_pnl for t in self.closed_trades]
    
    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)
    
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self.pnls:
            return 0.0
        wins = sum(1 for p in self.pnls if p > 0)
        return wins / len(self.pnls)
    
    def loss_rate(self) -> float:
        """Percentage of losing trades."""
        return 1.0 - self.win_rate()
    
    def avg_winner(self) -> float:
        """Average winning trade P&L."""
        wins = [p for p in self.pnls if p > 0]
        return sum(wins) / len(wins) if wins else 0.0
    
    def avg_loser(self) -> float:
        """Average losing trade P&L (absolute value)."""
        losses = [p for p in self.pnls if p < 0]
        return abs(sum(losses) / len(losses)) if losses else 0.0
    
    def profit_factor(self) -> float:
        """Gross profit / Gross loss."""
        gross_profit = sum(p for p in self.pnls if p > 0)
        gross_loss = abs(sum(p for p in self.pnls if p < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def expected_value(self) -> float:
        """Expected P&L per trade."""
        wr = self.win_rate()
        aw = self.avg_winner()
        al = self.avg_loser()
        return wr * aw - (1 - wr) * al
    
    def total_pnl(self) -> float:
        """Total realized P&L."""
        return sum(self.pnls)
    
    def win_loss_ratio(self) -> float:
        """Avg winner / Avg loser."""
        al = self.avg_loser()
        return self.avg_winner() / al if al > 0 else float('inf')
    
    def is_asymmetric(self) -> bool:
        """Check if results are sufficiently asymmetric."""
        return (
            self.win_loss_ratio() >= 2.0 and
            self.profit_factor() >= 1.5
        )
    
    def summary(self) -> str:
        """Generate summary report."""
        if not self.closed_trades:
            return "No trades recorded yet."
        
        lines = [
            "=" * 50,
            "ASYMMETRY STATISTICS",
            "=" * 50,
            f"Total Trades: {self.total_trades}",
            f"Win Rate: {self.win_rate()*100:.1f}%",
            f"",
            f"Avg Winner: ${self.avg_winner():.2f}",
            f"Avg Loser: ${self.avg_loser():.2f}",
            f"Win/Loss Ratio: {self.win_loss_ratio():.2f}x",
            f"",
            f"Profit Factor: {self.profit_factor():.2f}",
            f"Expected Value: ${self.expected_value():.2f}",
            f"Total P&L: ${self.total_pnl():.2f}",
            f"",
            f"Asymmetric: {'YES ✓' if self.is_asymmetric() else 'NO'}",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def daily_summary(self, date: Optional[datetime] = None) -> str:
        """Generate daily summary."""
        if date is None:
            date = datetime.now()
        
        today = date.date()
        today_trades = [
            t for t in self.closed_trades 
            if t.exit_time.date() == today
        ]
        
        if not today_trades:
            return f"No trades closed on {today}"
        
        pnls = [t.realized_pnl for t in today_trades]
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        
        return (
            f"Daily Summary ({today}):\n"
            f"  Trades: {len(today_trades)}\n"
            f"  Wins: {wins}/{len(today_trades)}\n"
            f"  P&L: ${total:.2f}"
        )
    
    def save(self, filepath: str):
        """Save statistics to JSON file."""
        data = {
            "trades": [
                {
                    "symbol": t.symbol,
                    "expiry": t.expiry,
                    "strike": t.strike,
                    "right": t.right,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "realized_pnl": t.realized_pnl,
                    "pnl_pct": t.pnl_pct,
                    "underlying_entry": t.underlying_entry,
                    "underlying_exit": t.underlying_exit,
                    "entry_beta": t.entry_beta,
                    "entry_vix": t.entry_vix,
                    "max_favorable_excursion": t.max_favorable_excursion,
                    "max_adverse_excursion": t.max_adverse_excursion,
                    "exit_reason": t.exit_reason,
                }
                for t in self.closed_trades
            ]
        }
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: str) -> "AsymmetryStats":
        """Load statistics from JSON file."""
        stats = cls()
        path = Path(filepath)
        if not path.exists():
            return stats
        
        data = json.loads(path.read_text())
        for t in data.get("trades", []):
            trade = ClosedTrade(
                symbol=t["symbol"],
                expiry=t["expiry"],
                strike=t["strike"],
                right=t["right"],
                entry_time=datetime.fromisoformat(t["entry_time"]),
                exit_time=datetime.fromisoformat(t["exit_time"]),
                entry_price=t["entry_price"],
                exit_price=t["exit_price"],
                quantity=t["quantity"],
                realized_pnl=t["realized_pnl"],
                pnl_pct=t["pnl_pct"],
                underlying_entry=t["underlying_entry"],
                underlying_exit=t["underlying_exit"],
                entry_beta=t["entry_beta"],
                entry_vix=t["entry_vix"],
                max_favorable_excursion=t.get("max_favorable_excursion", 0),
                max_adverse_excursion=t.get("max_adverse_excursion", 0),
                exit_reason=t.get("exit_reason", "stop"),
            )
            stats.closed_trades.append(trade)
        
        return stats
