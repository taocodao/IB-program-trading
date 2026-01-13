"""
Automated Trailing Stop-Loss Manager for Options Trading
=========================================================

STRATEGY:
---------
When market opens, this system:
1. Loads all open option positions from your IB account
2. For each position, places a STOP LOSS order at 10% below current bid price
3. Monitors prices continuously:
   - If price moves UP: Updates stop to 10% below new higher bid
   - If price moves DOWN: Stop stays fixed (does not move down)
4. When stop is hit: Position is automatically sold

Example:
--------
Buy SPY 550 CALL @ $10.00 (bid = $10.50)
→ Initial stop placed at $10.50 × 0.90 = $9.45

Bid rises to $12.00:
→ Stop updated to $12.00 × 0.90 = $10.80

Bid falls to $11.00:
→ Stop stays at $10.80 (no change)

Bid hits $10.80:
→ Position automatically SOLD at $10.80

Author: Options Trading Bot
License: MIT
"""

import sys
import time
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
from threading import Thread, Lock
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import OrderId, TickAttrib

# Import configuration
from config import (
    IB_HOST, IB_PORT, IB_CLIENT_ID,
    TRAIL_PERCENT, MIN_UPDATE_INTERVAL,
    LOG_FILE, LOG_LEVEL,
    MAX_POSITIONS, ALLOWED_SYMBOLS, EXCLUDED_SYMBOLS,
    MIN_POSITION_QUANTITY, RECONNECT_ATTEMPTS, RECONNECT_DELAY
)

# ============= LOGGING SETUP =============

# Create logs directory if needed
log_path = Path(LOG_FILE)
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============= DATA CLASSES =============

class OptionPosition:
    """Represents an open option position with stop-loss tracking."""
    
    def __init__(self, conid: int, symbol: str, sectype: str, 
                 expiry: str, strike: float, right: str, 
                 quantity: int, avg_price: float):
        self.conid = conid
        self.symbol = symbol
        self.sectype = sectype
        self.expiry = expiry
        self.strike = strike
        self.right = right  # 'C' for call, 'P' for put
        self.quantity = quantity
        self.avg_price = avg_price
        
        # Price tracking
        self.current_bid: Optional[float] = None
        self.current_ask: Optional[float] = None
        self.highest_bid: Optional[float] = None  # For trailing logic
        
        # Order tracking
        self.stop_order_id: Optional[int] = None
        self.stop_price: Optional[float] = None
        
        # Timestamps
        self.created_at = datetime.now()
        self.last_update: Optional[datetime] = None
    
    def __str__(self):
        right_str = "CALL" if self.right == 'C' else "PUT"
        return f"{self.symbol} {self.expiry} {self.strike} {right_str} x{self.quantity}"


# ============= MAIN APPLICATION =============

class TrailingStopManager(EClient, EWrapper):
    """
    Manages trailing stop-loss orders for options positions.
    
    Key Features:
    - Connects to Interactive Brokers TWS/IB Gateway
    - Fetches all open option positions on startup
    - Places and monitors stop-loss orders at 10% below bid
    - Trails stops UP as prices rise, keeps fixed when prices fall
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Position and order tracking
        self.positions: Dict[int, OptionPosition] = {}
        self.order_id_counter = 1000
        self.stop_orders: Dict[OrderId, int] = {}  # orderId -> conid
        
        # Thread safety
        self.lock = Lock()
        
        # State
        self.monitoring = False
        self.connected = False
        
        # Configuration (from config.py)
        self.trail_percent = TRAIL_PERCENT
        self.min_update_interval = MIN_UPDATE_INTERVAL
        self.last_order_time: Dict[int, float] = {}  # conid -> timestamp
        
    # ============= IB API CALLBACKS =============
    
    def nextValidId(self, orderId: int):
        """Called when connection is established - get starting order ID."""
        logger.info(f"Connected to IB. Next valid order ID: {orderId}")
        self.order_id_counter = orderId
        self.connected = True
    
    def position(self, account: str, contract: Contract, quantity: Decimal, avgCost: float):
        """
        Called for each open position in the account.
        We filter for options only.
        """
        # Only process options (OPT)
        if contract.secType != "OPT":
            return
        
        # Skip zero/closed positions
        if quantity == 0:
            return
        
        # Skip if quantity below minimum
        if abs(int(quantity)) < MIN_POSITION_QUANTITY:
            return
        
        # Apply symbol filters
        if ALLOWED_SYMBOLS and contract.symbol not in ALLOWED_SYMBOLS:
            logger.debug(f"Skipping {contract.symbol} - not in ALLOWED_SYMBOLS")
            return
        
        if contract.symbol in EXCLUDED_SYMBOLS:
            logger.debug(f"Skipping {contract.symbol} - in EXCLUDED_SYMBOLS")
            return
        
        # Create position object
        pos = OptionPosition(
            conid=contract.conId,
            symbol=contract.symbol,
            sectype=contract.secType,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=contract.strike,
            right=contract.right,
            quantity=int(quantity),
            avg_price=float(avgCost)
        )
        
        with self.lock:
            self.positions[contract.conId] = pos
        
        logger.info(f"Position loaded: {pos} @ ${avgCost:.2f}")
    
    def positionEnd(self):
        """Called after all positions have been received."""
        count = len(self.positions)
        logger.info(f"Position loading complete. Found {count} option positions")
        
        if count > MAX_POSITIONS:
            logger.warning(f"Position count {count} exceeds MAX_POSITIONS {MAX_POSITIONS}!")
        
        # Subscribe to market data for each position
        for conid, position in self.positions.items():
            self._subscribe_market_data(conid, position)
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib):
        """
        Called when price updates arrive.
        tickType: 1=BID, 2=ASK, 4=LAST, 9=CLOSE
        """
        if not self.positions or price <= 0:
            return
        
        # Find the position by request ID (we use conid as reqId)
        position = None
        for p in self.positions.values():
            if p.conid == reqId:
                position = p
                break
        
        if not position:
            return
        
        with self.lock:
            if tickType == 1:  # BID price
                position.current_bid = price
                # Track highest bid for trailing logic
                if position.highest_bid is None or price > position.highest_bid:
                    position.highest_bid = price
                    logger.debug(f"{position.symbol}: New high bid ${price:.2f}")
            
            elif tickType == 2:  # ASK price
                position.current_ask = price
            
            position.last_update = datetime.now()
    
    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState):
        """Called when an order is placed or updated."""
        logger.info(f"Order {orderId}: {order.action} {order.totalQuantity} "
                   f"{contract.symbol} @ ${order.auxPrice:.2f} Status: {orderState.status}")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal, 
                   remaining: Decimal, avgFillPrice: float, permId: int, 
                   parentId: int, lastFillPrice: float, clientId: int, 
                   whyHeld: str, mktCapPrice: float):
        """Called when order status changes."""
        if status == "Filled":
            logger.warning(f"*** STOP LOSS EXECUTED *** Order {orderId} filled at ${lastFillPrice:.2f}")
            with self.lock:
                if orderId in self.stop_orders:
                    conid = self.stop_orders[orderId]
                    if conid in self.positions:
                        position = self.positions[conid]
                        logger.warning(f"Position CLOSED: {position}")
                        position.stop_order_id = None
        
        elif status == "Cancelled":
            logger.debug(f"Order {orderId} cancelled")
    
    def error(self, reqId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Called on errors and warnings."""
        # Market data subscription confirmations (not errors)
        if errorCode in [2104, 2106, 2158]:
            logger.info(f"Market Data: {errorString}")
        # Connectivity
        elif errorCode == 1100:
            logger.error("Connection lost to TWS!")
            self.connected = False
        elif errorCode == 1102:
            logger.info("Connection restored to TWS")
            self.connected = True
        else:
            logger.error(f"Error {errorCode}: {errorString}")
    
    def connectionClosed(self):
        """Called when connection is closed."""
        logger.error("Connection to TWS/IB Gateway closed!")
        self.monitoring = False
        self.connected = False
    
    # ============= CORE TRADING LOGIC =============
    
    def _subscribe_market_data(self, conid: int, position: OptionPosition):
        """Subscribe to real-time market data for a position."""
        contract = Contract()
        contract.conId = conid
        contract.symbol = position.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = position.expiry
        contract.strike = position.strike
        contract.right = position.right
        
        # Use conid as request ID for easy lookup
        self.reqMktData(conid, contract, "", False, False, [])
        logger.info(f"Subscribed to market data: {position}")
    
    def place_stop_loss_order(self, position: OptionPosition) -> bool:
        """
        Place initial stop-loss order at TRAIL_PERCENT below current bid.
        Returns True if successful.
        """
        if position.current_bid is None:
            logger.warning(f"No bid price yet for {position.symbol}")
            return False
        
        # Calculate stop price: 10% below bid
        stop_price = round(position.current_bid * (1 - self.trail_percent), 2)
        
        # Build contract
        contract = Contract()
        contract.conId = position.conid
        contract.symbol = position.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = position.expiry
        contract.strike = position.strike
        contract.right = position.right
        
        # Build STOP order
        order = Order()
        order.orderId = self.order_id_counter
        order.clientId = IB_CLIENT_ID
        order.action = "SELL"  # Sell to close long position
        order.orderType = "STP"  # Stop order
        order.totalQuantity = abs(position.quantity)
        order.auxPrice = stop_price  # Stop trigger price
        order.tif = "GTC"  # Good till canceled
        order.transmit = True
        
        # Track the order
        position.stop_order_id = order.orderId
        position.stop_price = stop_price
        
        with self.lock:
            self.stop_orders[order.orderId] = position.conid
        
        # Place the order
        self.placeOrder(order.orderId, contract, order)
        self.order_id_counter += 1
        
        logger.info(f"PLACED STOP-LOSS: {position.symbol} @ ${stop_price:.2f} "
                   f"(Bid: ${position.current_bid:.2f}, Trail: {self.trail_percent*100:.0f}%)")
        
        self.last_order_time[position.conid] = time.time()
        return True
    
    def update_stop_loss_order(self, position: OptionPosition) -> bool:
        """
        Update stop-loss if price has moved UP.
        - ONLY moves UP (trails up with price)
        - STAYS FIXED if price drops (does not move down)
        Returns True if order was updated.
        """
        if position.current_bid is None or position.stop_order_id is None:
            return False
        
        # Throttle: don't update too frequently
        last_update = self.last_order_time.get(position.conid, 0)
        if time.time() - last_update < self.min_update_interval:
            return False
        
        # Calculate new potential stop price
        new_stop_price = round(position.current_bid * (1 - self.trail_percent), 2)
        current_stop = position.stop_price
        
        # KEY LOGIC: Only update if new stop is HIGHER (price moved up)
        if new_stop_price <= current_stop:
            return False  # Price dropped or stayed same - keep existing stop
        
        price_increase = new_stop_price - current_stop
        logger.info(f"{position.symbol}: Bid rose! Updating stop: "
                   f"${current_stop:.2f} → ${new_stop_price:.2f} (+${price_increase:.2f})")
        
        # Cancel old order, place new one
        self.cancelOrder(position.stop_order_id, "")
        
        # Build new order
        contract = Contract()
        contract.conId = position.conid
        contract.symbol = position.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = position.expiry
        contract.strike = position.strike
        contract.right = position.right
        
        order = Order()
        order.orderId = self.order_id_counter
        order.clientId = IB_CLIENT_ID
        order.action = "SELL"
        order.orderType = "STP"
        order.totalQuantity = abs(position.quantity)
        order.auxPrice = new_stop_price
        order.tif = "GTC"
        order.transmit = True
        
        position.stop_order_id = order.orderId
        position.stop_price = new_stop_price
        
        with self.lock:
            self.stop_orders[order.orderId] = position.conid
        
        self.placeOrder(order.orderId, contract, order)
        self.order_id_counter += 1
        
        self.last_order_time[position.conid] = time.time()
        return True
    
    # ============= MONITORING LOOP =============
    
    def monitor_positions(self):
        """
        Main monitoring loop.
        Runs continuously during trading day to update stops as prices move.
        """
        logger.info("Starting position monitoring loop...")
        self.monitoring = True
        check_count = 0
        
        while self.monitoring and self.isConnected():
            try:
                check_count += 1
                
                with self.lock:
                    for conid, position in list(self.positions.items()):
                        # Skip if no bid price yet
                        if position.current_bid is None:
                            continue
                        
                        # Place initial stop if not yet placed
                        if position.stop_order_id is None:
                            self.place_stop_loss_order(position)
                        else:
                            # Update stop if price moved up
                            self.update_stop_loss_order(position)
                
                # Status log every 60 checks (~1 minute)
                if check_count % 60 == 0:
                    active_stops = len([p for p in self.positions.values() if p.stop_order_id])
                    logger.info(f"[Check #{check_count}] {len(self.positions)} positions, "
                               f"{active_stops} active stops")
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Monitoring loop ended.")
    
    # ============= CONTROL METHODS =============
    
    def start(self, host: str = None, port: int = None, client_id: int = None):
        """Connect to TWS/IB Gateway and start monitoring."""
        host = host or IB_HOST
        port = port or IB_PORT
        client_id = client_id or IB_CLIENT_ID
        
        logger.info(f"Connecting to {host}:{port} (client ID: {client_id})...")
        
        self.connect(host, port, client_id)
        
        # Start API message thread
        api_thread = Thread(target=self.run, daemon=True)
        api_thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        if self.isConnected():
            logger.info("Connected successfully!")
            
            # Request all positions
            logger.info("Requesting portfolio positions...")
            self.reqPositions()
            
            time.sleep(3)  # Wait for positions to load
            
            # Start monitoring
            monitor_thread = Thread(target=self.monitor_positions, daemon=False)
            monitor_thread.start()
            
            return True
        else:
            logger.error("Failed to connect to TWS/IB Gateway")
            return False
    
    def stop(self):
        """Gracefully stop monitoring and disconnect."""
        logger.info("Stopping monitoring...")
        self.monitoring = False
        time.sleep(1)
        
        # Cancel all pending stop orders
        logger.info("Cancelling pending stop orders...")
        for order_id in list(self.stop_orders.keys()):
            try:
                self.cancelOrder(order_id, "")
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
        
        time.sleep(1)
        self.disconnect()
        logger.info("Disconnected. Shutdown complete.")


# ============= MAIN ENTRY POINT =============

def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("OPTIONS TRAILING STOP-LOSS MANAGER")
    logger.info("=" * 70)
    logger.info(f"Configuration: Trail={TRAIL_PERCENT*100:.0f}%, "
               f"Update Interval={MIN_UPDATE_INTERVAL}s")
    logger.info("")
    
    manager = TrailingStopManager()
    
    try:
        if not manager.start():
            logger.error("Failed to start manager. Exiting.")
            return 1
        
        logger.info("")
        logger.info("Monitoring is now ACTIVE. Press Ctrl+C to stop.")
        logger.info("")
        
        # Keep main thread alive
        while manager.monitoring:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Keyboard interrupt detected. Shutting down...")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        manager.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
