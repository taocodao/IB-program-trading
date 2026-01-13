"""
Per-User Trading Bot
====================

Trading bot that:
1. Subscribes to signals for user's watchlist
2. Applies user's risk settings
3. Executes trades via user's IB Gateway
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import threading

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

from models.multi_tenant import MultiTenantDB, User
from signal_service import SignalSubscriber
from stop_calculator import StopCalculator
from option_selector import (
    calculate_target_dte, calculate_target_delta, 
    DEFAULT_TARGET_DTE, DEFAULT_TARGET_DELTA
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserTradingBot(EClient, EWrapper):
    """Trading bot for a single user with their own IB Gateway."""
    
    def __init__(self, user_id: str, gateway_port: int):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.user_id = user_id
        self.gateway_port = gateway_port
        
        # Database
        self.db = MultiTenantDB()
        
        # User data
        self.settings = self.db.get_user_settings(user_id)
        self.watchlist = self.db.get_user_watchlist(user_id)
        
        # Trading state
        self.positions: Dict[str, dict] = {}
        self.pending_signals: Dict[str, dict] = {}
        self.next_order_id = 0
        
        # Signal subscriber
        self.signal_sub = SignalSubscriber()
        
        # Stop calculator with user's aggression
        self.stop_calc = StopCalculator(
            k_aggression=self.settings.get('stop_aggression', 1.0)
        )
        
        logger.info(f"UserTradingBot initialized for user {user_id}")
        logger.info(f"  Watchlist: {len(self.watchlist)} symbols")
        logger.info(f"  Min score: {self.settings.get('min_ai_score', 60)}")
        logger.info(f"  Max positions: {self.settings.get('max_positions', 10)}")
    
    def connect_and_start(self) -> bool:
        """Connect to user's IB Gateway."""
        logger.info(f"Connecting to gateway on port {self.gateway_port}...")
        
        self.connect("127.0.0.1", self.gateway_port, clientId=100)
        
        # Start API thread
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect to gateway")
            return False
        
        logger.info("Connected to IB Gateway")
        return True
    
    def start_signal_listener(self):
        """Start listening for signals."""
        def on_signal(signal):
            self.handle_signal(signal)
        
        # Subscribe to user's watchlist symbols
        self.signal_sub.subscribe(self.watchlist, on_signal)
        
        # Start listening in background
        listener_thread = threading.Thread(
            target=self.signal_sub.listen, 
            daemon=True
        )
        listener_thread.start()
        
        logger.info(f"Listening for signals on {len(self.watchlist)} symbols")
    
    def handle_signal(self, signal: dict):
        """Process incoming signal based on user's settings."""
        symbol = signal["symbol"]
        score = signal["score"]
        signal_type = signal["signal_type"]
        
        logger.info(f"Signal: {symbol} {signal_type} (score: {score})")
        
        # Check if symbol is in user's watchlist
        if symbol not in self.watchlist:
            return
        
        # Check score threshold
        min_score = self.settings.get('min_ai_score', 60)
        if score < min_score:
            logger.info(f"  Skipped: score {score} < threshold {min_score}")
            return
        
        # Check position limits
        max_positions = self.settings.get('max_positions', 10)
        if len(self.positions) >= max_positions:
            logger.info(f"  Skipped: max positions ({max_positions}) reached")
            return
        
        # Check if already in position for this symbol
        if symbol in self.positions:
            logger.info(f"  Skipped: already in position")
            return
        
        # Check auto-execute threshold
        auto_execute_score = self.settings.get('auto_execute_score', 85)
        
        if score >= auto_execute_score:
            # Auto-execute
            logger.info(f"  AUTO-EXECUTING (score >= {auto_execute_score})")
            self.execute_trade(signal)
        else:
            # Queue for manual approval
            self.pending_signals[symbol] = signal
            logger.info(f"  Queued for manual approval")
    
    def execute_trade(self, signal: dict):
        """Execute a trade based on signal and user settings."""
        symbol = signal["symbol"]
        signal_type = signal["signal_type"]
        score = signal["score"]
        
        # Determine option parameters
        target_delta = calculate_target_delta(score)
        target_dte = calculate_target_dte(50)  # Assume normal IV
        
        # Override with user settings if specified
        if self.settings.get('target_delta'):
            target_delta = float(self.settings['target_delta'])
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            return
        
        # Calculate position size
        max_position = self.settings.get('max_position_size', 10000)
        max_contracts = self.settings.get('max_contracts', 2)
        
        # Estimate option price (~2.5% of underlying for ATM)
        estimated_premium = current_price * 0.025
        cost_per_contract = estimated_premium * 100
        
        quantity = min(
            max_contracts,
            max(1, int(max_position / cost_per_contract))
        )
        
        # Determine option type
        right = "C" if signal_type == "BUY_CALL" else "P"
        
        # Calculate strike
        if right == "C":
            # Slightly ITM for higher delta
            strike = current_price * (1 - (target_delta - 0.50) * 0.20)
        else:
            strike = current_price * (1 + (target_delta - 0.50) * 0.20)
        
        # Round strike
        if current_price >= 100:
            strike = round(strike / 5) * 5
        else:
            strike = round(strike)
        
        # Calculate expiry
        from datetime import timedelta
        expiry_date = datetime.now() + timedelta(days=target_dte)
        # Find next Friday
        days_to_friday = (4 - expiry_date.weekday()) % 7
        expiry_date = expiry_date + timedelta(days=days_to_friday)
        expiry = expiry_date.strftime("%Y%m%d")
        
        logger.info(f"Executing: {quantity}x {symbol} {expiry} ${strike}{right}")
        
        # Place order
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"
        
        order = Order()
        order.action = "BUY"
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = estimated_premium  # Use limit order
        order.tif = "DAY"
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        self.placeOrder(order_id, contract, order)
        
        # Track position
        self.positions[symbol] = {
            "signal": signal,
            "order_id": order_id,
            "quantity": quantity,
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "entry_time": datetime.now(),
            "status": "pending"
        }
        
        logger.info(f"Order placed: {order_id}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # TODO: Implement real-time price fetching
        # For now, return None
        return None
    
    def refresh_settings(self):
        """Reload user settings from database."""
        self.settings = self.db.get_user_settings(self.user_id)
        self.watchlist = self.db.get_user_watchlist(self.user_id)
        
        # Update stop calculator
        self.stop_calc = StopCalculator(
            k_aggression=self.settings.get('stop_aggression', 1.0)
        )
        
        logger.info("Settings refreshed")
    
    # ========== EWrapper Callbacks ==========
    
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        logger.info(f"Next order ID: {orderId}")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        logger.info(f"Order {orderId}: {status} (filled: {filled})")
        
        # Update position status
        for symbol, pos in self.positions.items():
            if pos.get("order_id") == orderId:
                pos["status"] = status.lower()
                if status == "Filled":
                    pos["fill_price"] = avgFillPrice
                    logger.info(f"  Filled {symbol} @ ${avgFillPrice}")
    
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode not in [2104, 2106, 2158]:
            logger.error(f"Error {errorCode}: {errorString}")


# ============= CLI =============

if __name__ == "__main__":
    print("UserTradingBot - Per-User Trading Engine")
    print("=" * 50)
    print("This bot is spawned for each user with their own:")
    print("  - IB Gateway connection")
    print("  - Risk settings")
    print("  - Watchlist")
    print("\nTo run for a specific user:")
    print("  python user_trading_bot.py <user_id> <gateway_port>")
