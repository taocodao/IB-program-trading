"""
IB Connection Test Script
=========================

Run this script to verify connectivity to TWS/IB Gateway.

Usage:
    python tests/test_connection.py

Expected output:
    ✓ Connected! Next Order ID: XXXX
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class TestConnection(EClient, EWrapper):
    """Simple connection test client."""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
    
    def nextValidId(self, orderId: int):
        """Called when connection is established."""
        self.connected = True
        self.next_order_id = orderId
        print(f"✓ Connected! Next Order ID: {orderId}")
        self.disconnect()
    
    def error(self, reqId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Handle errors."""
        if errorCode in [2104, 2106, 2158]:
            # Market data info, not errors
            pass
        elif errorCode == 504:
            print(f"✗ Not connected - TWS/IB Gateway is not running or API not enabled")
        else:
            print(f"Error {errorCode}: {errorString}")


def test_connection(host: str = "127.0.0.1", port: int = 7497, client_id: int = 999):
    """
    Test connection to Interactive Brokers.
    
    Args:
        host: IB API host (default: localhost)
        port: IB API port (7497 for TWS, 4002 for Gateway)
        client_id: Client ID for connection
    """
    print(f"\nTesting connection to {host}:{port}...")
    print("-" * 40)
    
    app = TestConnection()
    
    try:
        app.connect(host, port, client_id)
        
        # Run for a few seconds to receive callback
        timeout = 5
        start = time.time()
        while not app.connected and time.time() - start < timeout:
            time.sleep(0.1)
            if app.isConnected():
                app.run()
        
        if not app.connected:
            print(f"✗ Connection timeout after {timeout}s")
            print("\nTroubleshooting:")
            print("  1. Is TWS or IB Gateway running?")
            print("  2. Is API enabled? (File → Global Config → API → Settings)")
            print("  3. Is socket port correct? (TWS: 7497, Gateway: 4002)")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    
    finally:
        if app.isConnected():
            app.disconnect()


if __name__ == "__main__":
    # Try TWS port first, then Gateway port
    from config import IB_HOST, IB_PORT
    
    success = test_connection(IB_HOST, IB_PORT)
    
    if not success and IB_PORT == 7497:
        print("\nTrying IB Gateway port (4002)...")
        success = test_connection(IB_HOST, 4002)
    
    print("")
    sys.exit(0 if success else 1)
