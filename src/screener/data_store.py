"""
Data Store for Screener
=======================

Handles:
- Watchlist loading with betas
- Previous close caching
- Alert storage (SQLite)
"""

import csv
import sqlite3
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WatchlistItem:
    """Single watchlist entry."""
    symbol: str
    beta: float = 1.0


def load_watchlist_with_betas(filepath: str) -> List[WatchlistItem]:
    """
    Load watchlist from CSV and add betas.
    
    Uses beta from models.py if available.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import get_beta
    
    items = []
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"Watchlist not found: {filepath}")
        return items
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get('Symbol', '').strip()
            if symbol and symbol != '':
                beta = get_beta(symbol)
                items.append(WatchlistItem(symbol=symbol, beta=beta))
    
    logger.info(f"Loaded {len(items)} symbols from watchlist")
    return items


class DataStore:
    """
    Simple data store for screener.
    
    Features:
    - SQLite for alerts
    - In-memory cache for prev close
    """
    
    def __init__(self, db_path: str = "screener_alerts.db"):
        self.db_path = db_path
        self.prev_close_cache: Dict[str, float] = {}
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create database tables if needed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                price REAL,
                actual_move REAL,
                expected_move REAL,
                abnormality REAL,
                rating REAL,
                signal_type TEXT,
                direction TEXT,
                volume_ratio REAL,
                macd_state TEXT,
                rsi REAL,
                bb_pos TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prev_close (
                symbol TEXT PRIMARY KEY,
                close_price REAL,
                date TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: dict):
        """Save an alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (
                symbol, timestamp, price, actual_move, expected_move,
                abnormality, rating, signal_type, direction,
                volume_ratio, macd_state, rsi, bb_pos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.get('symbol'),
            datetime.now().isoformat(),
            alert.get('price'),
            alert.get('actual_pct'),
            alert.get('expected_pct'),
            alert.get('abnormality'),
            alert.get('score'),
            alert.get('signal'),
            alert.get('direction'),
            alert.get('volume_ratio'),
            alert.get('macd_state'),
            alert.get('rsi'),
            alert.get('bb_pos')
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Alert saved: {alert.get('symbol')} - {alert.get('signal')}")
    
    def set_prev_close(self, symbol: str, close_price: float):
        """Store previous close for a symbol."""
        self.prev_close_cache[symbol] = close_price
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO prev_close (symbol, close_price, date)
            VALUES (?, ?, ?)
        """, (symbol, close_price, date.today().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_prev_close(self, symbol: str) -> Optional[float]:
        """Get previous close for a symbol."""
        # Check cache first
        if symbol in self.prev_close_cache:
            return self.prev_close_cache[symbol]
        
        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT close_price FROM prev_close WHERE symbol = ?",
            (symbol,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.prev_close_cache[symbol] = row[0]
            return row[0]
        
        return None
    
    def get_alerts_today(self) -> List[dict]:
        """Get all alerts from today."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        cursor.execute(
            "SELECT * FROM alerts WHERE timestamp LIKE ?",
            (f"{today}%",)
        )
        
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return alerts
