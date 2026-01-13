"""
Signal Distribution Service
============================

Centralized signal generation with Redis pub/sub for multi-tenant distribution.

This service:
1. Aggregates all symbols from all user watchlists
2. Generates AI signals ONCE per symbol
3. Publishes signals to Redis for per-user consumption
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional
import threading

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import redis
import pandas as pd

from models.multi_tenant import MultiTenantDB
from ai_signal_generator import AISignalGenerator, SignalType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SIGNAL_REFRESH_SECONDS = 60  # How often to regenerate signals
SIGNAL_EXPIRY_SECONDS = 300  # How long signals stay in cache


class SignalService:
    """Centralized signal generation and distribution."""
    
    def __init__(self):
        self.db = MultiTenantDB()
        self.ai_gen = AISignalGenerator()
        
        # Redis connection
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        
        # State
        self.active_symbols: Set[str] = set()
        self.running = False
        
        logger.info("Signal Service initialized")
    
    def refresh_active_symbols(self):
        """Get all unique symbols across all user watchlists."""
        self.active_symbols = self.db.get_all_active_symbols()
        logger.info(f"Active symbols: {len(self.active_symbols)}")
        return self.active_symbols
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[dict]:
        """Generate AI signal for a symbol."""
        try:
            signal = self.ai_gen.generate_signal_from_data(data, symbol, "5m")
            
            if signal.signal_type == SignalType.NO_SIGNAL:
                return None
            
            return {
                "symbol": symbol,
                "signal_type": signal.signal_type.value,
                "score": signal.consensus_score,
                "indicators": {
                    "rsi": signal.indicator_scores.get("rsi_score", 0),
                    "macd": signal.indicator_scores.get("macd_score", 0),
                    "momentum": signal.indicator_scores.get("momentum_score", 0),
                    "volume": signal.indicator_scores.get("volume_score", 0),
                },
                "timestamp": datetime.now().isoformat(),
                "expiry": (datetime.now().timestamp() + SIGNAL_EXPIRY_SECONDS)
            }
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def publish_signal(self, signal: dict):
        """Publish signal to Redis pub/sub and cache."""
        symbol = signal["symbol"]
        signal_json = json.dumps(signal)
        
        # Publish to channel (real-time subscribers)
        self.redis.publish(f"signals:{symbol}", signal_json)
        
        # Cache for later retrieval
        self.redis.setex(
            f"signal_cache:{symbol}",
            SIGNAL_EXPIRY_SECONDS,
            signal_json
        )
        
        # Also publish to global channel
        self.redis.publish("signals:all", signal_json)
        
        logger.info(f"Published signal: {symbol} {signal['signal_type']} (score: {signal['score']})")
    
    def get_cached_signal(self, symbol: str) -> Optional[dict]:
        """Get cached signal for a symbol."""
        cached = self.redis.get(f"signal_cache:{symbol}")
        if cached:
            return json.loads(cached)
        return None
    
    def get_all_cached_signals(self) -> Dict[str, dict]:
        """Get all cached signals."""
        signals = {}
        for key in self.redis.scan_iter("signal_cache:*"):
            symbol = key.split(":")[-1]
            cached = self.redis.get(key)
            if cached:
                signals[symbol] = json.loads(cached)
        return signals
    
    def get_users_for_signal(self, symbol: str) -> list:
        """Get all user IDs watching this symbol."""
        return self.db.get_users_watching_symbol(symbol)
    
    def run_once(self, market_data: Dict[str, pd.DataFrame]):
        """
        Run one cycle of signal generation.
        
        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
        """
        self.refresh_active_symbols()
        
        signals_generated = 0
        
        for symbol in self.active_symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if len(data) < 50:  # Need enough bars for indicators
                continue
            
            signal = self.generate_signal(symbol, data)
            
            if signal:
                self.publish_signal(signal)
                signals_generated += 1
        
        logger.info(f"Generated {signals_generated} signals from {len(self.active_symbols)} symbols")
        return signals_generated
    
    def health_check(self) -> dict:
        """Check service health."""
        try:
            self.redis.ping()
            redis_ok = True
        except:
            redis_ok = False
        
        return {
            "status": "healthy" if redis_ok else "degraded",
            "redis": redis_ok,
            "active_symbols": len(self.active_symbols),
            "cached_signals": len(self.get_all_cached_signals())
        }


# ============= Signal Subscriber (for trading bots) =============

class SignalSubscriber:
    """Subscribe to signals for specific symbols."""
    
    def __init__(self, symbols: list = None):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self.symbols = symbols or []
        self.callback = None
    
    def subscribe(self, symbols: list, callback):
        """
        Subscribe to signals for specific symbols.
        
        Args:
            symbols: List of symbols to watch
            callback: Function to call with (signal_dict)
        """
        self.symbols = symbols
        self.callback = callback
        
        channels = [f"signals:{s}" for s in symbols]
        self.pubsub.subscribe(*channels)
        
        logger.info(f"Subscribed to {len(channels)} symbol channels")
    
    def subscribe_all(self, callback):
        """Subscribe to ALL signals."""
        self.callback = callback
        self.pubsub.subscribe("signals:all")
        logger.info("Subscribed to all signals")
    
    def listen(self):
        """Start listening for signals (blocking)."""
        for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    signal = json.loads(message["data"])
                    if self.callback:
                        self.callback(signal)
                except Exception as e:
                    logger.error(f"Error processing signal: {e}")
    
    def get_cached_signal(self, symbol: str) -> Optional[dict]:
        """Get cached signal for a symbol."""
        cached = self.redis.get(f"signal_cache:{symbol}")
        if cached:
            return json.loads(cached)
        return None


# ============= CLI Test =============

if __name__ == "__main__":
    print("=" * 60)
    print("Signal Distribution Service")
    print("=" * 60)
    
    service = SignalService()
    
    # Health check
    health = service.health_check()
    print(f"\nHealth: {health}")
    
    # Refresh symbols
    symbols = service.refresh_active_symbols()
    print(f"\nActive symbols from all watchlists: {symbols}")
    
    # Show cached signals
    cached = service.get_all_cached_signals()
    print(f"\nCached signals: {len(cached)}")
    for symbol, signal in cached.items():
        print(f"  {symbol}: {signal['signal_type']} (score: {signal['score']})")
