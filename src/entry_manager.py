"""
Trailing Entry Manager
======================

Monitors underlyings and triggers BUY orders when price rises above trailing level.

Opposite of trailing stop:
- Trailing Stop: follows price UP, triggers SELL when drops
- Trailing Entry: follows price DOWN, triggers BUY when rises

Usage:
    manager = TrailingEntryManager()
    manager.add_symbol("AAPL", current_price=240.0)
    
    # On each price update:
    for price in prices:
        triggered = manager.update("AAPL", price)
        if triggered:
            buy_option(...)
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
from decimal import Decimal

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

from models import get_beta
from config_advanced import (
    IB_HOST, IB_PORT, IB_CLIENT_ID,
    PORTFOLIO_SIZE, MAX_POSITIONS
)

logger = logging.getLogger(__name__)


# ============= Configuration =============

ENTRY_TRAIL_PCT = 0.02          # 2% above low triggers buy
ENTRY_EXPIRY_DAYS = 14          # Target 2 weeks out
ENTRY_STRIKE_OTM_PCT = 0.01     # 1% OTM strike (slightly above for calls)
WATCHLIST_PATH = "watchlist.csv"


@dataclass
class TrailingEntry:
    """Tracks a symbol for trailing entry."""
    
    symbol: str
    beta: float = 1.0
    
    # Price tracking
    current_price: float = 0.0
    lowest_price: float = float('inf')
    trail_level: float = float('inf')
    
    # Entry parameters
    trail_pct: float = ENTRY_TRAIL_PCT
    
    # State
    entry_triggered: bool = False
    trigger_time: Optional[datetime] = None
    trigger_price: Optional[float] = None
    
    # Order tracking
    order_placed: bool = False
    order_id: Optional[int] = None
    
    def update_price(self, price: float) -> bool:
        """
        Update with new price. Returns True if entry triggered.
        """
        self.current_price = price
        
        # Update low and trail level
        if price < self.lowest_price:
            self.lowest_price = price
            self.trail_level = price * (1 + self.trail_pct)
            logger.debug(f"{self.symbol}: New low ${price:.2f}, trail level ${self.trail_level:.2f}")
        
        # Check trigger
        if price >= self.trail_level and not self.entry_triggered:
            self.entry_triggered = True
            self.trigger_time = datetime.now()
            self.trigger_price = price
            logger.warning(
                f"*** ENTRY TRIGGERED *** {self.symbol}: "
                f"${price:.2f} >= trail ${self.trail_level:.2f}"
            )
            return True
        
        return False
    
    def reset(self):
        """Reset for new entry opportunity."""
        self.lowest_price = float('inf')
        self.trail_level = float('inf')
        self.entry_triggered = False
        self.trigger_time = None
        self.trigger_price = None
        self.order_placed = False
        self.order_id = None


def load_watchlist(filepath: str = WATCHLIST_PATH) -> List[str]:
    """Load symbols from watchlist CSV."""
    symbols = []
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"Watchlist not found: {filepath}")
        return symbols
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get('Symbol', '').strip()
            if symbol and symbol != '':
                symbols.append(symbol)
    
    logger.info(f"Loaded {len(symbols)} symbols from watchlist")
    return symbols


def find_near_expiry(target_days: int = 14) -> str:
    """
    Find expiry date approximately target_days out.
    Options expire on Fridays, so find the nearest Friday.
    
    Returns: YYYYMMDD format string
    """
    today = datetime.now()
    target = today + timedelta(days=target_days)
    
    # Find next Friday (weekday 4)
    days_until_friday = (4 - target.weekday()) % 7
    if days_until_friday == 0 and target.hour > 16:
        days_until_friday = 7
    
    expiry_date = target + timedelta(days=days_until_friday)
    return expiry_date.strftime("%Y%m%d")


def find_atm_strike(underlying_price: float, otm_pct: float = 0.01) -> float:
    """
    Find strike price slightly OTM.
    
    For calls, strike = price * (1 + otm_pct), rounded to nearest $5
    """
    target_strike = underlying_price * (1 + otm_pct)
    
    # Round to nearest $5 for most stocks
    if underlying_price >= 100:
        rounded = round(target_strike / 5) * 5
    elif underlying_price >= 50:
        rounded = round(target_strike / 2.5) * 2.5
    else:
        rounded = round(target_strike)
    
    return float(rounded)


class TrailingEntryManager(EClient, EWrapper):
    """
    Monitors underlyings and triggers BUY orders when trailing entry is hit.
    """
    
    def __init__(self, watchlist_path: str = WATCHLIST_PATH):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        self.next_order_id = 0
        
        # Symbols to monitor
        self.entries: Dict[str, TrailingEntry] = {}
        
        # Request ID mappings
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 5000
        
        # Triggered orders
        self.pending_orders: Dict[int, TrailingEntry] = {}
        self.filled_orders: List[Dict] = []
        
        # Load watchlist
        symbols = load_watchlist(watchlist_path)
        for symbol in symbols:
            beta = get_beta(symbol)
            self.entries[symbol] = TrailingEntry(
                symbol=symbol,
                beta=beta,
                trail_pct=ENTRY_TRAIL_PCT
            )
    
    def connect_and_start(self, host: str = IB_HOST, port: int = IB_PORT, 
                         client_id: int = IB_CLIENT_ID + 10):
        """Connect to IB and start monitoring."""
        logger.info(f"Connecting to {host}:{port}...")
        self.connect(host, port, client_id)
        
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        import time
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect to IB")
            return False
        
        self.connected = True
        logger.info(f"Connected! Monitoring {len(self.entries)} symbols")
        return True
    
    def subscribe_market_data(self):
        """Subscribe to market data for all symbols."""
        for symbol, entry in self.entries.items():
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            req_id = self.next_req_id
            self.next_req_id += 1
            self.req_to_symbol[req_id] = symbol
            
            self.reqMktData(req_id, contract, "", False, False, [])
            logger.debug(f"Subscribed to {symbol} (req_id={req_id})")
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """Handle price updates."""
        # tickType 4 = last price, 1 = bid, 2 = ask
        if tickType not in [1, 2, 4]:
            return
        
        if price <= 0:
            return
        
        symbol = self.req_to_symbol.get(reqId)
        if not symbol or symbol not in self.entries:
            return
        
        entry = self.entries[symbol]
        
        # Skip if already triggered
        if entry.entry_triggered:
            return
        
        triggered = entry.update_price(price)
        
        if triggered:
            self._place_option_order(entry)
    
    def _place_option_order(self, entry: TrailingEntry):
        """Place option BUY order when entry triggered."""
        if entry.order_placed:
            return
        
        symbol = entry.symbol
        underlying_price = entry.trigger_price or entry.current_price
        
        # Find strike and expiry
        strike = find_atm_strike(underlying_price, ENTRY_STRIKE_OTM_PCT)
        expiry = find_near_expiry(ENTRY_EXPIRY_DAYS)
        
        logger.info(
            f"Placing order: BUY 1 {symbol} {expiry} ${strike} CALL "
            f"(underlying ${underlying_price:.2f})"
        )
        
        # Build contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = "C"  # CALL
        contract.multiplier = "100"
        
        # Build order - Market order for quick fill
        order = Order()
        order.orderId = self.next_order_id
        order.action = "BUY"
        order.orderType = "MKT"
        order.totalQuantity = 1
        order.tif = "DAY"
        order.transmit = True
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        # Place order
        self.placeOrder(self.next_order_id, contract, order)
        
        entry.order_placed = True
        entry.order_id = self.next_order_id
        self.pending_orders[self.next_order_id] = entry
        
        self.next_order_id += 1
    
    def orderStatus(self, orderId: int, status: str, filled: Decimal, 
                   remaining: Decimal, avgFillPrice: float, *args):
        """Handle order status updates."""
        if status == "Filled":
            entry = self.pending_orders.pop(orderId, None)
            if entry:
                self.filled_orders.append({
                    'symbol': entry.symbol,
                    'trigger_price': entry.trigger_price,
                    'fill_price': avgFillPrice,
                    'time': datetime.now()
                })
                logger.info(
                    f"*** ORDER FILLED *** {entry.symbol} CALL @ ${avgFillPrice:.2f}"
                )
    
    def nextValidId(self, orderId: int):
        """Callback: next valid order ID."""
        self.next_order_id = orderId
        logger.info(f"Next order ID: {orderId}")
    
    def error(self, reqId: int, errorCode: int, errorString: str, *args):
        """Handle errors."""
        if errorCode in [2104, 2106, 2158]:
            logger.debug(f"Info: {errorString}")
        elif errorCode == 10168:
            logger.warning(f"Contract issue: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")
    
    def get_summary(self) -> str:
        """Get summary of filled orders."""
        if not self.filled_orders:
            return "No orders filled yet"
        
        lines = [
            "=" * 60,
            "FILLED ORDERS",
            "=" * 60,
        ]
        
        for order in self.filled_orders:
            lines.append(
                f"  {order['symbol']}: triggered @ ${order['trigger_price']:.2f}, "
                f"filled @ ${order['fill_price']:.2f}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


def run_entry_monitor():
    """Main entry point to run the entry monitor."""
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("TRAILING ENTRY MONITOR")
    logger.info("=" * 60)
    logger.info(f"Trail percentage: {ENTRY_TRAIL_PCT*100:.1f}%")
    logger.info(f"Target expiry: ~{ENTRY_EXPIRY_DAYS} days")
    logger.info(f"Strike OTM: {ENTRY_STRIKE_OTM_PCT*100:.1f}%")
    
    manager = TrailingEntryManager()
    
    if not manager.connect_and_start():
        return
    
    # Subscribe to market data
    manager.subscribe_market_data()
    
    logger.info("\nMonitoring started. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(60)
            
            # Status update
            triggered = sum(1 for e in manager.entries.values() if e.entry_triggered)
            filled = len(manager.filled_orders)
            
            logger.info(f"[Status] Monitoring {len(manager.entries)} symbols, "
                       f"{triggered} triggered, {filled} filled")
    
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    
    finally:
        print(manager.get_summary())
        manager.disconnect()


if __name__ == "__main__":
    run_entry_monitor()
