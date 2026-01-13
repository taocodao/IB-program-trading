"""
Advanced Volatility-Aware Options Stop-Loss Manager
====================================================

STRATEGY:
---------
This system uses volatility-aware stop sizing:

    stop_distance = k × β × σ_index

Where:
- k = aggression factor (0.7-1.5)
- β = stock beta vs S&P 500
- σ_index = daily index volatility (from VIX)

Key Features:
1. Risk-sized stops: Wider in volatile markets, tighter in calm
2. Underlying-driven triggers: Stop based on underlying price, not option bid
3. Smart execution: Limit orders with theoretical pricing guardrails
4. Adaptive re-pricing: Walk down limit if order doesn't fill

Author: Volatility Trading Bot
License: MIT
"""

import sys
import time
import math
import threading
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import OrderId, TickAttrib

# Local imports
from models import (
    OptionPosition, PositionStatus, VolatilityTracker, get_beta
)
from stop_calculator import (
    StopCalculator, compute_theoretical_price, compute_smart_limit_price
)
from config_advanced import (
    IB_HOST, IB_PORT, IB_CLIENT_ID,
    K_AGGRESSION, MIN_TRAIL_PCT, MAX_TRAIL_PCT,
    DTE_7_30_MULTIPLIER, DTE_UNDER_7_MULTIPLIER,
    EXIT_SPREAD_PARTICIPATION, EXIT_MAX_REPRICES, EXIT_REPRICE_INTERVAL,
    EXIT_ALLOWED_SLIPPAGE_PCT, RISK_FREE_RATE,
    LOG_FILE, LOG_LEVEL,
    ALLOWED_SYMBOLS, EXCLUDED_SYMBOLS, MIN_POSITION_QUANTITY,
    RECONNECT_ATTEMPTS, RECONNECT_DELAY, VIX_REQ_ID
)

# ============= LOGGING SETUP =============

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


class VolatilityAwareStopManager(EClient, EWrapper):
    """
    Advanced stop-loss manager using beta × index volatility sizing.
    
    Key innovations:
    1. Stop sized by: k × β × σ_index
    2. Trigger based on underlying price movement
    3. Smart limit order execution with theoretical pricing
    4. Adaptive re-pricing if order doesn't fill
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Position tracking
        self.positions: Dict[int, OptionPosition] = {}
        self.next_order_id = 1000
        
        # Volatility tracking
        self.vol_tracker = VolatilityTracker()
        
        # Stop calculator
        self.stop_calc = StopCalculator(
            k_aggression=K_AGGRESSION,
            min_trail_pct=MIN_TRAIL_PCT,
            max_trail_pct=MAX_TRAIL_PCT,
            dte_7_30_mult=DTE_7_30_MULTIPLIER,
            dte_under_7_mult=DTE_UNDER_7_MULTIPLIER
        )
        
        # Thread safety
        self.lock = threading.RLock()
        
        # State
        self.monitoring = False
        self.connected = False
        
        # Exit order tracking
        self.exit_orders: Dict[int, int] = {}  # order_id -> conid
    
    # ============= CONNECTION CALLBACKS =============
    
    def nextValidId(self, orderId: int):
        """Called when connection established."""
        logger.info(f"Connected to IB. Next order ID: {orderId}")
        self.next_order_id = orderId
        self.connected = True
    
    def connectionClosed(self):
        """Called when connection lost."""
        logger.error("Connection to TWS closed!")
        self.connected = False
        self.monitoring = False
    
    def error(self, reqId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Handle errors and warnings."""
        # Market data info (not errors)
        if errorCode in [2104, 2106, 2158]:
            logger.info(f"Market Data: {errorString}")
        elif errorCode == 1100:
            logger.error("Connection lost!")
            self.connected = False
        elif errorCode == 1102:
            logger.info("Connection restored")
            self.connected = True
        elif errorCode == 10168:
            # Requested contract not found (VIX issue sometimes)
            logger.warning(f"Contract issue: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")
    
    # ============= POSITION LOADING =============
    
    def position(self, account: str, contract: Contract, quantity: Decimal, avgCost: float):
        """Receive portfolio position."""
        # Only options
        if contract.secType != "OPT":
            return
        
        if quantity == 0:
            return
        
        if abs(int(quantity)) < MIN_POSITION_QUANTITY:
            return
        
        # Symbol filtering
        if ALLOWED_SYMBOLS and contract.symbol not in ALLOWED_SYMBOLS:
            return
        if contract.symbol in EXCLUDED_SYMBOLS:
            return
        
        # Get beta for underlying
        beta = get_beta(contract.symbol)
        
        # Create position
        pos = OptionPosition(
            conid=contract.conId,
            symbol=contract.symbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=contract.strike,
            right=contract.right,
            quantity=int(quantity),
            avg_entry_price=float(avgCost),
            underlying_symbol=contract.symbol,
            underlying_beta=beta,
        )
        
        with self.lock:
            self.positions[contract.conId] = pos
        
        logger.info(f"Position loaded: {pos} | Beta: {beta:.2f}")
    
    def positionEnd(self):
        """All positions received."""
        count = len(self.positions)
        logger.info(f"Position loading complete. Found {count} option positions")
        
        # Subscribe to market data
        self._subscribe_all_positions()
        
        # Subscribe to VIX
        self._subscribe_to_vix()
    
    # ============= MARKET DATA =============
    
    def _subscribe_all_positions(self):
        """Subscribe to market data for all positions."""
        with self.lock:
            for conid, pos in self.positions.items():
                self._subscribe_option(conid, pos)
    
    def _subscribe_option(self, conid: int, pos: OptionPosition):
        """Subscribe to market data and Greeks for one option."""
        contract = Contract()
        contract.conId = conid
        contract.symbol = pos.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = pos.expiry
        contract.strike = pos.strike
        contract.right = pos.right
        
        # Use conid as request ID
        self.reqMktData(conid, contract, "", False, False, [])
        logger.info(f"Subscribed to market data: {pos.symbol} {pos.strike} {pos.right}")
    
    def _subscribe_to_vix(self):
        """Subscribe to VIX for volatility tracking."""
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        
        self.reqMktData(VIX_REQ_ID, contract, "", False, False, [])
        logger.info("Subscribed to VIX for volatility tracking")
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib):
        """Receive price updates."""
        if price <= 0:
            return
        
        # VIX update
        if reqId == VIX_REQ_ID:
            if tickType == 4:  # LAST
                self.vol_tracker.update_vix(price)
                logger.debug(f"VIX updated: {price:.2f}")
            return
        
        # Option price update
        with self.lock:
            pos = self.positions.get(reqId)
            if not pos:
                return
            
            if tickType == 1:  # BID
                pos.current_bid = price
            elif tickType == 2:  # ASK
                pos.current_ask = price
            elif tickType == 4:  # LAST
                pos.current_last = price
            
            pos.last_update = datetime.now()
    
    def tickOptionComputation(
        self,
        reqId: int,
        tickType: int,
        tickAttrib: int,
        impliedVol: float,
        delta: float,
        optPrice: float,
        pvDividend: float,
        gamma: float,
        vega: float,
        theta: float,
        undPrice: float
    ):
        """Receive Greeks from IB model."""
        # Only use tick type 13 (Model) - most stable
        if tickType != 13:
            return
        
        with self.lock:
            pos = self.positions.get(reqId)
            if not pos:
                return
            
            pos.delta = delta
            pos.gamma = gamma
            pos.vega = vega
            pos.theta = theta
            pos.implied_vol = impliedVol
            pos.model_price = optPrice
            pos.underlying_price = undPrice
            
            # Set entry price if not set
            if pos.underlying_entry_price == 0 and undPrice > 0:
                pos.underlying_entry_price = undPrice
                pos.underlying_high = undPrice
            
            # Update trailing high
            pos.update_underlying_high()
    
    # ============= STOP LOGIC (CORE) =============
    
    def _update_all_stops(self):
        """Update stop levels for all positions."""
        index_vol = self.vol_tracker.get_daily_vol_pct()
        
        with self.lock:
            for conid, pos in self.positions.items():
                if pos.status in [PositionStatus.CLOSED, PositionStatus.EXIT_FILLED]:
                    continue
                
                if pos.underlying_price is None:
                    continue
                
                # Compute stop from trailing high
                dte = pos.days_to_expiry()
                
                new_stop = self.stop_calc.compute_trail_from_high(
                    underlying_high=pos.underlying_high,
                    beta=pos.underlying_beta,
                    index_vol_pct=index_vol,
                    days_to_expiry=dte
                )
                
                # Trailing: only move UP (ratchet)
                if pos.underlying_stop_level is None:
                    pos.underlying_stop_level = new_stop
                    trail_pct = self.stop_calc.get_trail_percentage(
                        pos.underlying_beta, index_vol, dte
                    )
                    logger.info(
                        f"[STOP SET] {pos.symbol}: Entry=${pos.underlying_entry_price:.2f}, "
                        f"Stop=${new_stop:.2f} ({trail_pct*100:.1f}% trail, β={pos.underlying_beta:.2f})"
                    )
                else:
                    if new_stop > pos.underlying_stop_level:
                        old_stop = pos.underlying_stop_level
                        pos.underlying_stop_level = new_stop
                        logger.info(
                            f"[STOP TRAIL] {pos.symbol}: Stop raised "
                            f"${old_stop:.2f} → ${new_stop:.2f}"
                        )
                
                # Track for logging
                pos.trail_distance_dollars = pos.underlying_high - pos.underlying_stop_level
                pos.trail_distance_pct = pos.trail_distance_dollars / pos.underlying_high
                
                # Check trigger
                if pos.underlying_price <= pos.underlying_stop_level and not pos.exit_triggered:
                    logger.warning(
                        f"*** STOP TRIGGERED *** {pos.symbol}: Underlying ${pos.underlying_price:.2f} "
                        f"<= Stop ${pos.underlying_stop_level:.2f}"
                    )
                    pos.exit_triggered = True
                    pos.exit_triggered_time = datetime.now()
                    pos.status = PositionStatus.EXIT_TRIGGERED
    
    def _handle_triggered_exits(self):
        """Execute smart exits for triggered positions."""
        with self.lock:
            triggered = [
                (conid, pos) for conid, pos in self.positions.items()
                if pos.status == PositionStatus.EXIT_TRIGGERED
            ]
        
        for conid, pos in triggered:
            self._execute_smart_exit(conid, pos)
    
    def _execute_smart_exit(self, conid: int, pos: OptionPosition):
        """Place smart limit order to exit position."""
        if pos.current_bid is None or pos.current_ask is None:
            logger.warning(f"{pos.symbol}: No bid/ask available, skipping exit")
            return
        
        if pos.underlying_stop_level is None:
            return
        
        # Compute theoretical at stop level
        theo = compute_theoretical_price(
            underlying_price=pos.underlying_stop_level,
            strike=pos.strike,
            days_to_expiry=pos.days_to_expiry(),
            implied_vol=pos.implied_vol or 0.30,
            right=pos.right,
            rate=RISK_FREE_RATE
        )
        
        # Compute smart limit
        limit = compute_smart_limit_price(
            current_bid=pos.current_bid,
            current_ask=pos.current_ask,
            theoretical_price=theo,
            spread_participation=EXIT_SPREAD_PARTICIPATION,
            allowed_slippage_pct=EXIT_ALLOWED_SLIPPAGE_PCT
        )
        
        # Build contract
        contract = Contract()
        contract.conId = conid
        contract.symbol = pos.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = pos.expiry
        contract.strike = pos.strike
        contract.right = pos.right
        
        # Build limit order
        order = Order()
        order.orderId = self.next_order_id
        order.clientId = IB_CLIENT_ID
        order.action = "SELL"
        order.orderType = "LMT"
        order.totalQuantity = abs(pos.quantity)
        order.lmtPrice = limit
        order.tif = "GTC"
        order.transmit = True
        
        self.placeOrder(order.orderId, contract, order)
        
        with self.lock:
            pos.exit_order_id = order.orderId
            pos.exit_limit_price = limit
            pos.status = PositionStatus.EXIT_ORDER_PLACED
            self.exit_orders[order.orderId] = conid
        
        self.next_order_id += 1
        
        logger.info(
            f"[EXIT ORDER] {pos.symbol}: SELL {pos.quantity} @ ${limit:.2f} "
            f"(bid ${pos.current_bid:.2f}, ask ${pos.current_ask:.2f}, theo ${theo:.2f})"
        )
    
    def _reprice_unfilled_exits(self):
        """Re-price unfilled exit orders if needed."""
        with self.lock:
            unfilled = [
                (conid, pos) for conid, pos in self.positions.items()
                if pos.status == PositionStatus.EXIT_ORDER_PLACED
                and pos.exit_order_id is not None
                and pos.exit_reprices < EXIT_MAX_REPRICES
            ]
        
        for conid, pos in unfilled:
            # Check time since last reprice
            if pos.exit_last_reprice_time:
                elapsed = (datetime.now() - pos.exit_last_reprice_time).total_seconds()
                if elapsed < EXIT_REPRICE_INTERVAL:
                    continue
            elif pos.exit_triggered_time:
                elapsed = (datetime.now() - pos.exit_triggered_time).total_seconds()
                if elapsed < EXIT_REPRICE_INTERVAL:
                    continue
            
            # Check if underlying recovered
            if pos.underlying_price and pos.underlying_stop_level:
                if pos.underlying_price > pos.underlying_stop_level * 1.02:
                    logger.info(f"{pos.symbol}: Underlying recovered, cancelling exit")
                    self.cancelOrder(pos.exit_order_id, "")
                    with self.lock:
                        pos.status = PositionStatus.TRACKING
                        pos.exit_triggered = False
                        pos.exit_order_id = None
                    continue
            
            # Recompute limit (deeper into spread)
            if pos.current_bid and pos.current_ask and pos.implied_vol:
                theo = compute_theoretical_price(
                    underlying_price=pos.underlying_stop_level or pos.underlying_price,
                    strike=pos.strike,
                    days_to_expiry=pos.days_to_expiry(),
                    implied_vol=pos.implied_vol,
                    right=pos.right
                )
                
                # Go deeper into spread on each reprice
                participation = EXIT_SPREAD_PARTICIPATION + (0.1 * pos.exit_reprices)
                participation = min(participation, 0.9)
                
                new_limit = compute_smart_limit_price(
                    pos.current_bid, pos.current_ask, theo,
                    spread_participation=participation
                )
                
                if new_limit < (pos.exit_limit_price or float('inf')):
                    # Cancel and replace
                    self.cancelOrder(pos.exit_order_id, "")
                    
                    # Place new order
                    contract = Contract()
                    contract.conId = conid
                    contract.symbol = pos.symbol
                    contract.secType = "OPT"
                    contract.exchange = "SMART"
                    contract.currency = "USD"
                    contract.lastTradeDateOrContractMonth = pos.expiry
                    contract.strike = pos.strike
                    contract.right = pos.right
                    
                    order = Order()
                    order.orderId = self.next_order_id
                    order.action = "SELL"
                    order.orderType = "LMT"
                    order.totalQuantity = abs(pos.quantity)
                    order.lmtPrice = new_limit
                    order.tif = "GTC"
                    order.transmit = True
                    
                    self.placeOrder(order.orderId, contract, order)
                    
                    with self.lock:
                        pos.exit_order_id = order.orderId
                        pos.exit_limit_price = new_limit
                        pos.exit_reprices += 1
                        pos.exit_last_reprice_time = datetime.now()
                        self.exit_orders[order.orderId] = conid
                    
                    self.next_order_id += 1
                    
                    logger.info(
                        f"[REPRICE #{pos.exit_reprices}] {pos.symbol}: "
                        f"New limit ${new_limit:.2f}"
                    )
    
    # ============= ORDER CALLBACKS =============
    
    def orderStatus(
        self, orderId: int, status: str, filled: Decimal,
        remaining: Decimal, avgFillPrice: float, permId: int,
        parentId: int, lastFillPrice: float, clientId: int,
        whyHeld: str, mktCapPrice: float
    ):
        """Handle order status updates."""
        if status == "Filled":
            with self.lock:
                conid = self.exit_orders.get(orderId)
                if conid and conid in self.positions:
                    pos = self.positions[conid]
                    pos.status = PositionStatus.EXIT_FILLED
                    pos.closed_price = avgFillPrice
                    pos.closed_time = datetime.now()
                    pos.realized_pnl = (avgFillPrice - pos.avg_entry_price) * abs(pos.quantity) * 100
                    
                    logger.warning(
                        f"*** POSITION CLOSED *** {pos.symbol}: "
                        f"Filled @ ${avgFillPrice:.2f}, P&L: ${pos.realized_pnl:.2f}"
                    )
        
        elif status == "Cancelled":
            logger.debug(f"Order {orderId} cancelled")
    
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
        """Handle order placement confirmation."""
        logger.debug(f"Order {orderId}: {order.action} {order.totalQuantity} @ {order.lmtPrice}")
    
    # ============= MONITORING LOOP =============
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting volatility-aware monitoring loop...")
        self.monitoring = True
        check_count = 0
        
        while self.monitoring and self.isConnected():
            try:
                check_count += 1
                
                # Update stop levels
                self._update_all_stops()
                
                # Handle triggered exits
                self._handle_triggered_exits()
                
                # Reprice unfilled exits
                self._reprice_unfilled_exits()
                
                # Status log every 60 iterations (~1 min)
                if check_count % 60 == 0:
                    active = len([p for p in self.positions.values() 
                                 if p.status == PositionStatus.TRACKING])
                    triggered = len([p for p in self.positions.values() 
                                    if p.status in [PositionStatus.EXIT_TRIGGERED, 
                                                   PositionStatus.EXIT_ORDER_PLACED]])
                    vol = self.vol_tracker.get_daily_vol_pct() * 100
                    
                    logger.info(
                        f"[CHECK #{check_count}] {active} tracking, {triggered} triggered | "
                        f"VIX: {self.vol_tracker.vix_level or 'N/A'}, Vol: {vol:.2f}%"
                    )
                
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Monitoring loop ended.")
    
    # ============= CONTROL =============
    
    def start(self, host: str = None, port: int = None, client_id: int = None):
        """Start the manager."""
        host = host or IB_HOST
        port = port or IB_PORT
        client_id = client_id or IB_CLIENT_ID
        
        logger.info(f"Connecting to {host}:{port} (client ID: {client_id})...")
        
        self.connect(host, port, client_id)
        
        # Start API thread
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect")
            return False
        
        logger.info("Connected successfully!")
        
        # Request positions
        logger.info("Requesting positions...")
        self.reqPositions()
        
        time.sleep(3)
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=False)
        monitor_thread.start()
        
        return True
    
    def stop(self):
        """Stop the manager."""
        logger.info("Stopping...")
        self.monitoring = False
        time.sleep(1)
        
        # Cancel exit orders
        for order_id in list(self.exit_orders.keys()):
            try:
                self.cancelOrder(order_id, "")
            except:
                pass
        
        time.sleep(1)
        self.disconnect()
        logger.info("Stopped.")


# ============= MAIN =============

def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("VOLATILITY-AWARE OPTIONS STOP-LOSS MANAGER")
    logger.info("=" * 70)
    logger.info(f"Config: k={K_AGGRESSION}, trail=[{MIN_TRAIL_PCT*100:.0f}%-{MAX_TRAIL_PCT*100:.0f}%]")
    logger.info("")
    
    manager = VolatilityAwareStopManager()
    
    try:
        if not manager.start():
            logger.error("Failed to start")
            return 1
        
        logger.info("")
        logger.info("Monitoring ACTIVE. Press Ctrl+C to stop.")
        logger.info("")
        
        while manager.monitoring:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        manager.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
