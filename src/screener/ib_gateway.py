"""
IB Gateway Integration for Screener
====================================

Handles:
- Connection to IB Gateway/TWS
- Real-time price subscriptions
- 1-minute bar history for indicators
- VIX subscription
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Real-time tick data for a symbol."""
    symbol: str
    last: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    prev_close: float = 0.0
    timestamp: Optional[datetime] = None


class ScreenerGateway(EClient, EWrapper):
    """
    IB Gateway connection for the stock screener.
    
    Features:
    - Subscribe to real-time prices
    - Request 1-min historical bars for indicators
    - Subscribe to VIX for volatility
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 200):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self.lock = threading.RLock()
        self.connected = False
        
        # Data storage
        self.tick_data: Dict[str, TickData] = {}
        self.bar_data: Dict[int, List[BarData]] = {}
        self.bar_complete: Dict[int, bool] = {}
        
        # Request ID mappings
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 10000
        
        # VIX tracking
        self.vix_level = 20.0
        self.vix_req_id = None
        
        # Callbacks
        self.on_tick_callback: Optional[Callable] = None
    
    def connect_and_start(self) -> bool:
        """Connect to IB and start message processing."""
        try:
            logger.info(f"Connecting to IB Gateway at {self.host}:{self.port}...")
            self.connect(self.host, self.port, self.client_id)
            
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            time.sleep(2)
            
            if not self.isConnected():
                logger.error("Failed to connect to IB Gateway")
                return False
            
            self.connected = True
            logger.info("✓ Connected to IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def subscribe_stock(self, symbol: str):
        """Subscribe to real-time data for a stock."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.req_to_symbol[req_id] = symbol
        self.tick_data[symbol] = TickData(symbol=symbol)
        
        # Request market data: last, bid, ask, volume
        self.reqMktData(req_id, contract, "", False, False, [])
        logger.debug(f"Subscribed to {symbol} (req_id={req_id})")
    
    def subscribe_vix(self):
        """Subscribe to VIX for volatility reference."""
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        
        self.vix_req_id = self.next_req_id
        self.next_req_id += 1
        
        self.reqMktData(self.vix_req_id, contract, "", False, False, [])
        logger.info("Subscribed to VIX")
    
    def request_historical_bars(
        self, 
        symbol: str, 
        bar_size: str = "1 min",
        duration: str = "1 D"
    ) -> Optional[pd.DataFrame]:
        """
        Request historical bars for indicator calculation.
        
        Args:
            symbol: Stock symbol
            bar_size: Bar size (e.g., "1 min", "5 mins")
            duration: How far back (e.g., "1 D", "2 D")
        
        Returns:
            DataFrame with OHLCV data or None
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.bar_data[req_id] = []
        self.bar_complete[req_id] = False
        
        self.reqHistoricalData(
            req_id,
            contract,
            "",  # endDateTime = now
            duration,
            bar_size,
            "TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Wait for data
        timeout = 15
        start = time.time()
        while not self.bar_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.warning(f"Timeout waiting for {symbol} bars")
                return None
            time.sleep(0.1)
        
        bars = self.bar_data.get(req_id, [])
        if not bars:
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            for bar in bars
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        tick = self.tick_data.get(symbol)
        if tick:
            return tick.last
        return 0.0
    
    def get_prev_close(self, symbol: str) -> float:
        """Get previous close for a symbol."""
        tick = self.tick_data.get(symbol)
        if tick:
            return tick.prev_close
        return 0.0
    
    def get_vix(self) -> float:
        """Get current VIX level."""
        return self.vix_level
    
    # ========== EWrapper Callbacks ==========
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """Handle price ticks."""
        if price <= 0:
            return
        
        # Check VIX
        if reqId == self.vix_req_id:
            if tickType in [1, 2, 4]:  # Bid, Ask, Last
                self.vix_level = price
            return
        
        symbol = self.req_to_symbol.get(reqId)
        if not symbol:
            return
        
        with self.lock:
            if symbol not in self.tick_data:
                self.tick_data[symbol] = TickData(symbol=symbol)
            
            tick = self.tick_data[symbol]
            tick.timestamp = datetime.now()
            
            if tickType == 1:  # Bid
                tick.bid = price
            elif tickType == 2:  # Ask
                tick.ask = price
            elif tickType == 4:  # Last
                tick.last = price
            elif tickType == 9:  # Close (prev close)
                tick.prev_close = price
        
        # Notify callback
        if self.on_tick_callback and tickType == 4:
            self.on_tick_callback(symbol, price)
    
    def tickSize(self, reqId: int, tickType: int, size: int):
        """Handle size ticks."""
        if tickType == 8:  # Volume
            symbol = self.req_to_symbol.get(reqId)
            if symbol and symbol in self.tick_data:
                self.tick_data[symbol].volume = size
    
    def historicalData(self, reqId: int, bar: BarData):
        """Handle historical bar data."""
        with self.lock:
            if reqId not in self.bar_data:
                self.bar_data[reqId] = []
            self.bar_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Historical data complete."""
        with self.lock:
            self.bar_complete[reqId] = True
    
    def nextValidId(self, orderId: int):
        """Next valid order ID received."""
        logger.info(f"Connected. Next order ID: {orderId}")
    
    def error(self, reqId: int, errorCode: int, errorString: str, *args):
        """Handle errors."""
        if errorCode in [2104, 2106, 2158]:
            logger.debug(f"Info: {errorString}")
        elif errorCode in [10167, 10168]:
            logger.warning(f"Data warning: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")
