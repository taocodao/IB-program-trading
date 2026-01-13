"""
Buy Test Options for Paper Trading
===================================

Places orders for test options with different betas to test the stop system.
Connects to paper trading account via IB API.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order


class TestOrderPlacer(EClient, EWrapper):
    """Places test option orders."""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.next_order_id = None
        self.connected = False
        self.orders_placed = 0
    
    def nextValidId(self, orderId: int):
        print(f"✓ Connected! Next order ID: {orderId}")
        self.next_order_id = orderId
        self.connected = True
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        if status == "Filled":
            print(f"  ✓ Order {orderId} FILLED at ${avgFillPrice:.2f}")
        elif status == "Submitted":
            print(f"  → Order {orderId} submitted")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]:
            pass  # Market data info
        elif errorCode == 201:
            print(f"  ✗ Order rejected: {errorString}")
        elif errorCode == 10147:
            print(f"  ⚠ Order modification warning: {errorString}")
        else:
            print(f"Error {errorCode}: {errorString}")
    
    def place_option_order(self, symbol: str, expiry: str, strike: float, 
                           right: str, quantity: int = 1):
        """Place a BUY order for an option."""
        
        # Build option contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right  # "C" or "P"
        contract.multiplier = "100"
        
        # Build order - Market order for quick fill
        order = Order()
        order.orderId = self.next_order_id
        order.action = "BUY"
        order.orderType = "MKT"  # Market order
        order.totalQuantity = quantity
        order.tif = "DAY"  # Day order instead of GTC
        order.transmit = True
        
        # Explicitly disable problematic attributes
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        right_str = "CALL" if right == "C" else "PUT"
        print(f"\nPlacing order: BUY {quantity} {symbol} {expiry} {strike} {right_str}")
        
        self.placeOrder(self.next_order_id, contract, order)
        self.next_order_id += 1
        self.orders_placed += 1
        
        time.sleep(2)  # Wait for order processing


def main():
    print("=" * 60)
    print("TEST OPTION ORDER PLACER - Paper Trading")
    print("=" * 60)
    print("\nThis will buy small test options to test the stop system.")
    print("Make sure TWS is connected to your PAPER trading account!\n")
    
    app = TestOrderPlacer()
    
    # Connect to TWS
    print("Connecting to TWS on port 7497...")
    app.connect("127.0.0.1", 7497, clientId=200)
    
    # Start message loop
    import threading
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    
    # Wait for connection
    timeout = 10
    start = time.time()
    while not app.connected and time.time() - start < timeout:
        time.sleep(0.1)
    
    if not app.connected:
        print("✗ Failed to connect. Is TWS running?")
        return 1
    
    time.sleep(1)
    
    # Get a valid expiry date (3rd Friday of next month or February 2026)
    from datetime import datetime, timedelta
    
    # Use a known valid monthly expiry - February 21, 2026 (3rd Friday)
    # Or March 20, 2026
    expiry = "20260220"  # Feb 20, 2026 (Friday)
    
    print(f"\nTarget expiry: {expiry} (Feb 2026 monthly)")
    print("\nPlacing test orders for different beta stocks:")
    print("-" * 60)
    
    # Define test options with different betas
    # Using ETFs and stocks (not index options which require special permissions)
    # Strikes based on approximate current prices
    test_options = [
        # (symbol, strike, right, beta_note)
        ("AAPL", 240.0, "C", "β=1.25 - Apple"),      # AAPL ~$240
        ("MSFT", 420.0, "C", "β=1.10 - Microsoft"),  # MSFT ~$420  
        ("TSLA", 400.0, "C", "β=2.00 - Tesla"),      # TSLA ~$400
    ]
    
    print("\nWhich options do you want to buy?")
    print("1. AAPL only (β=1.25)")
    print("2. All three (AAPL, MSFT, TSLA - different betas)")
    print("3. Cancel")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "3":
        print("Cancelled.")
        app.disconnect()
        return 0
    
    if choice == "2":
        options_to_buy = test_options
    else:
        options_to_buy = [test_options[0]]  # Just SPY
    
    print(f"\nPlacing {len(options_to_buy)} order(s)...")
    
    for symbol, strike, right, note in options_to_buy:
        print(f"\n{note}")
        app.place_option_order(symbol, expiry, strike, right, quantity=1)
    
    # Wait for orders to process
    print("\nWaiting for orders to fill...")
    time.sleep(5)
    
    print("\n" + "=" * 60)
    print(f"Done! Placed {app.orders_placed} order(s).")
    print("=" * 60)
    print("\nNow run the stop manager to test:")
    print("  python advanced_volatility_stops.py")
    print()
    
    app.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
