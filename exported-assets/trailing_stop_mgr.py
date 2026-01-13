"""
Automated Trailing Stop-Loss Manager for Options Trading
========================================================

This system monitors open option positions and dynamically manages stop-loss orders.
- Initial stop-loss: 10% below current bid price
- Trailing behavior: Moves up with price, stays fixed on drops
- Execution: Automatically updates orders throughout the trading day

Author: Options Trading Bot
License: MIT
"""

import sys
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from threading import Thread, Lock
import logging

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import OrderId, TickAttrib
from ibapi.tag_value import TagValue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trailing_stop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        self.current_bid = None
        self.current_ask = None
        self.highest_bid = None  # Track highest bid for trailing logic
        self.stop_order_id = None
        self.stop_price = None
        self.created_at = datetime.now()
        self.last_update = None
    
    def __str__(self):
        right_str = "CALL" if self.right == 'C' else "PUT"
        return f"{self.symbol} {self.expiry} {self.strike} {right_str} x{self.quantity}"


class TrailingStopManager(EClient, EWrapper):
    """
    Manages trailing stop-loss orders for options positions.
    
    Key Features:
    - Connects to Interactive Brokers TWS/IB Gateway
    - Fetches all open option positions
    - Places and monitors stop-loss orders
    - Dynamically adjusts stops as prices move
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        self.positions: Dict[int, OptionPosition] = {}
        self.order_id_counter = 1000
        self.stop_orders: Dict[OrderId, int] = {}  # orderId -> conid
        self.highest_prices: Dict[int, float] = {}  # conid -> highest bid
        self.lock = Lock()
        self.market_open = False
        self.monitoring = False
        self.trail_percent = 0.10  # 10% below bid
        self.min_update_interval = 2  # seconds between order updates
        self.last_order_time = {}  # conid -> last update timestamp
        
    # ============= EWrapper Callbacks =============
    
    def nextValidId(self, orderId: int):
        """Callback when connection established - get starting order ID."""
        logger.info(f"Connected. Next valid order ID: {orderId}")
        self.order_id_counter = orderId
    
    def contractDetails(self, reqId: int, contractDetails):
        """Receive contract details (currently unused but logged)."""
        logger.debug(f"Contract details received for reqId {reqId}")
    
    def position(self, account: str, contract: Contract, quantity: Decimal, avgCost: float):
        """
        Callback: Receive portfolio position.
        Called for each open position in account.
        """
        # Only process options
        if contract.secType != "OPT":
            return
        
        if quantity == 0:
            return
        
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
        
        logger.info(f"Position loaded: {pos} @ ${avgCost}")
    
    def positionEnd(self):
        """Callback: All positions received."""
        logger.info(f"Position loading complete. Found {len(self.positions)} option positions")
        
        # Request market data for all positions
        for conid, position in self.positions.items():
            self.request_market_data(conid, position)
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib):
        """
        Callback: Receive tick prices (bid, ask, last, etc).
        tickType: 1=BID, 2=ASK, 4=LAST, 9=CLOSE, etc.
        """
        if not self.positions:
            return
        
        # Find position by market data request ID
        position = None
        for p in self.positions.values():
            if p.conid == reqId:
                position = p
                break
        
        if not position:
            return
        
        with self.lock:
            if tickType == 1:  # BID
                position.current_bid = price
                # Track highest bid for trailing logic
                if position.highest_bid is None or price > position.highest_bid:
                    position.highest_bid = price
                    logger.debug(f"{position.symbol}: New high bid ${price}")
            
            elif tickType == 2:  # ASK
                position.current_ask = price
            
            position.last_update = datetime.now()
    
    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState):
        """Callback: Order placed/updated."""
        logger.info(f"Order {orderId}: {order.action} {order.totalQuantity} "
                   f"{contract.symbol} @ {order.auxPrice} Status: {orderState.status}")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal, 
                   remaining: Decimal, avgFillPrice: float, permId: int, 
                   parentId: int, lastFillPrice: float, clientId: int, 
                   whyHeld: str, mktCapPrice: float):
        """Callback: Order status update."""
        if status == "Filled":
            logger.warning(f"*** STOP LOSS EXECUTED *** Order {orderId} filled at ${lastFillPrice}")
            with self.lock:
                if orderId in self.stop_orders:
                    conid = self.stop_orders[orderId]
                    if conid in self.positions:
                        position = self.positions[conid]
                        position.stop_order_id = None
                        logger.warning(f"Position closed: {position}")
        
        elif status == "Cancelled":
            logger.debug(f"Order {orderId} cancelled (will be updated)")
    
    def error(self, reqId, errorCode: int, errorString: str):
        """Callback: Error occurred."""
        if errorCode in [2104, 2158]:  # Market data subscriptions
            logger.info(f"Market Data: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")
    
    def connectionClosed(self):
        """Callback: Connection lost."""
        logger.error("Connection to TWS/IB Gateway closed!")
        self.monitoring = False
    
    # ============= Core Methods =============
    
    def request_market_data(self, conid: int, position: OptionPosition):
        """Subscribe to market data for an option position."""
        contract = Contract()
        contract.conId = conid
        contract.symbol = position.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = position.expiry
        contract.strike = position.strike
        contract.right = position.right
        
        self.reqMktData(conid, contract, "", False, False, [])
        logger.info(f"Subscribed to market data: {position}")
    
    def place_stop_loss_order(self, position: OptionPosition) -> bool:
        """
        Place initial stop-loss order at 10% below current bid.
        Returns True if successful.
        """
        if position.current_bid is None:
            logger.warning(f"No bid price yet for {position.symbol}")
            return False
        
        stop_price = position.current_bid * (1 - self.trail_percent)
        
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
        
        # Build order
        order = Order()
        order.orderId = self.order_id_counter
        order.clientId = 0
        order.action = "SELL"  # Always sell to close
        order.orderType = "STP"  # Stop order
        order.totalQuantity = position.quantity
        order.auxPrice = stop_price  # Stop price
        order.tif = "GTC"  # Good till canceled
        order.transmit = True
        
        position.stop_order_id = order.orderId
        position.stop_price = stop_price
        
        with self.lock:
            self.stop_orders[order.orderId] = position.conid
        
        self.placeOrder(order.orderId, contract, order)
        self.order_id_counter += 1
        
        logger.info(f"PLACED STOP-LOSS: {position.symbol} @ ${stop_price:.2f} "
                   f"(Bid: ${position.current_bid:.2f})")
        
        self.last_order_time[position.conid] = time.time()
        return True
    
    def update_stop_loss_order(self, position: OptionPosition) -> bool:
        """
        Update stop-loss if price has moved up significantly.
        - Only moves UP (trail up)
        - Stays fixed if price drops
        Returns True if order was updated.
        """
        if position.current_bid is None or position.stop_order_id is None:
            return False
        
        # Check time throttle
        last_update_time = self.last_order_time.get(position.conid, 0)
        if time.time() - last_update_time < self.min_update_interval:
            return False
        
        new_stop_price = position.current_bid * (1 - self.trail_percent)
        current_stop = position.stop_price
        
        # Only update if new stop is higher (trailing up)
        if new_stop_price <= current_stop:
            return False  # Price dropped or stayed same, keep order
        
        price_diff = new_stop_price - current_stop
        logger.info(f"{position.symbol}: Price moved up ${price_diff:.2f}. "
                   f"Updating stop: ${current_stop:.2f} -> ${new_stop_price:.2f}")
        
        # Cancel old order
        self.cancelOrder(position.stop_order_id)
        
        # Place new order
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
        order.clientId = 0
        order.action = "SELL"
        order.orderType = "STP"
        order.totalQuantity = position.quantity
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
    
    def monitor_positions(self):
        """
        Main monitoring loop.
        Runs throughout trading day to update stop losses as prices move.
        """
        logger.info("Starting position monitoring...")
        self.monitoring = True
        check_count = 0
        
        while self.monitoring and self.isConnected():
            try:
                check_count += 1
                
                with self.lock:
                    for conid, position in self.positions.items():
                        # Skip if no bid price yet
                        if position.current_bid is None:
                            continue
                        
                        # Place initial stop if not yet placed
                        if position.stop_order_id is None:
                            self.place_stop_loss_order(position)
                        else:
                            # Update existing stop if price moved up
                            self.update_stop_loss_order(position)
                
                # Log status every 60 checks
                if check_count % 60 == 0:
                    logger.info(f"Monitoring check #{check_count}: "
                               f"{len(self.positions)} positions tracked, "
                               f"{len([p for p in self.positions.values() if p.stop_order_id])} stops active")
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5)
    
    def start(self, host="127.0.0.1", port=7497, client_id=100):
        """Connect to TWS/IB Gateway and start monitoring."""
        logger.info(f"Connecting to {host}:{port} (client ID: {client_id})...")
        
        self.connect(host, port, client_id)
        
        # Start API message loop in separate thread
        api_thread = Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(1)  # Wait for connection
        
        if self.isConnected():
            logger.info("Connected successfully!")
            
            # Request all positions
            self.reqPositions()
            
            time.sleep(2)  # Wait for positions to load
            
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
        
        # Cancel all orders
        for order_id in list(self.stop_orders.keys()):
            self.cancelOrder(order_id)
        
        time.sleep(1)
        self.disconnect()
        logger.info("Disconnected and stopped.")


# ============= Main Entry Point =============

def main():
    """Main execution function."""
    
    logger.info("="*70)
    logger.info("OPTIONS TRAILING STOP-LOSS MANAGER")
    logger.info("="*70)
    
    # Create manager
    manager = TrailingStopManager()
    
    try:
        # Connect to IB (default: localhost:7497 for TWS)
        # For IB Gateway, use port 4002 or 4001
        if not manager.start(host="127.0.0.1", port=7497, client_id=100):
            logger.error("Failed to start manager")
            return
        
        logger.info("\nMonitoring is now active. Press Ctrl+C to stop.\n")
        
        # Keep main thread alive
        while manager.monitoring:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt detected. Shutting down...")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    
    finally:
        manager.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
