"""
End-to-End Trading System
=========================

Complete integration test that:
1. Connects to IB Gateway
2. Loads watchlist and monitors for signals
3. When signal detected → Buy option (slightly ATM, 2-week expiry)
4. Monitor position with trailing stop
5. Sell when stop triggered

This is a SIMULATION using historical data - no real orders placed.
Set LIVE_MODE = True to place real orders (paper trading only!)
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import threading

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from decimal import Decimal

from models import get_beta, VolatilityTracker
from stop_calculator import StopCalculator
from screener.formulas import (
    expected_move, abnormality_score, enhanced_score, 
    classify_signal, get_direction
)
from screener.indicators import get_all_indicators

# Import AI signal modules
try:
    from ai_signal_generator import AISignalGenerator, SignalType
    from iv_skew_analyzer import IVSkewAnalyzer
    from signal_validator import SignalValidator
    from dashboard import get_settings, add_signal, add_pending_trade, check_trade_approved
    AI_SIGNALS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI signals not available: {e}")
    AI_SIGNALS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============= Configuration =============

LIVE_MODE = True            # Paper trading enabled!
IB_HOST = "127.0.0.1"
IB_PORT = 7497              # Paper trading port
IB_CLIENT_ID = 300

# Screener settings
ABN_THRESHOLD = 1.5         # Abnormality threshold
MIN_SCORE = 60              # Minimum score to trigger (can be overridden by dashboard)
SCAN_INTERVAL = 10          # Seconds between scans

# AI Signal settings (defaults - can be overridden by dashboard)
AI_MIN_SCORE = 60           # Minimum AI consensus score
AI_AUTO_EXECUTE = 80        # Auto-execute threshold (raised to reduce trades)
AI_TIMEFRAME = "5m"         # Default timeframe

# Entry settings
ENTRY_TRAIL_PCT = 0.02      # 2% above low triggers entry
STRIKE_OTM_PCT = 0.01       # 1% OTM strike
EXPIRY_DAYS = 14            # ~2 weeks

# Exit settings  
EXIT_TRAIL_PCT = 0.06       # 6% trailing stop
K_AGGRESSION = 1.0

# Position limits
MAX_OPEN_POSITIONS = 5      # Maximum simultaneous positions

# Watchlist
WATCHLIST_PATH = "watchlist.csv"

# ============= Risk & EOD Imports =============
try:
    from risk_config import (
        get_risk_config, calculate_position_size, filter_signal_by_direction,
        UserPreferences, RiskConfig, TradeFrequency, DirectionalBias, EODStrategy
    )
    from eod_manager import EODManager, Position as EODPosition
    RISK_SYSTEM_AVAILABLE = True
    logger.info("Risk & EOD system loaded")
except ImportError as e:
    logger.warning(f"Risk system not available: {e}")
    RISK_SYSTEM_AVAILABLE = False

# Default user preferences (can be overridden per user)
DEFAULT_USER_PREFS = {
    "risk_tolerance": 5,
    "trade_frequency": "moderate",
    "directional_bias": "both",
    "eod_strategy": "friday_close",
    "avoid_earnings": True,
}


@dataclass
class TrackedPosition:
    """Position being monitored."""
    symbol: str
    option_symbol: str
    strike: float
    expiry: str
    right: str
    
    entry_price: float
    entry_underlying: float
    entry_time: datetime
    quantity: int = 1
    
    # Stop tracking
    underlying_high: float = 0.0
    stop_level: float = 0.0
    beta: float = 1.0
    
    # State
    exit_triggered: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0


class TradingSystem(EClient, EWrapper):
    """
    Complete end-to-end trading system.
    
    Combines:
    - Screener for signal detection
    - Entry manager for option buying
    - Stop manager for position exits
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        self.next_order_id = 0
        
        # Watchlist with betas
        self.watchlist: List[dict] = []
        
        # Market data
        self.prices: Dict[str, float] = {}
        self.prev_close: Dict[str, float] = {}
        self.req_to_symbol: Dict[int, str] = {}
        self.next_req_id = 1000
        
        # Historical bars cache
        self.bar_data: Dict[int, List] = {}
        self.bar_complete: Dict[int, bool] = {}
        
        # VIX
        self.vix_level = 20.0
        self.vix_req_id = None
        
        # Trailing entry tracking
        self.entry_lows: Dict[str, float] = {}
        self.entry_triggered: Dict[str, bool] = {}
        
        # Positions
        self.positions: Dict[str, TrackedPosition] = {}
        self.closed_trades: List[TrackedPosition] = []
        
        # Stop calculator
        self.stop_calc = StopCalculator(
            k_aggression=K_AGGRESSION,
            min_trail_pct=EXIT_TRAIL_PCT
        )
        self.vol_tracker = VolatilityTracker()
        
        # AI Signal Generator
        if AI_SIGNALS_AVAILABLE:
            self.ai_generator = AISignalGenerator()
            self.iv_analyzer = IVSkewAnalyzer(simulation_mode=True)
            self.signal_validator = SignalValidator()
            logger.info("AI Signal Generator initialized")
        else:
            self.ai_generator = None
            self.iv_analyzer = None
            self.signal_validator = None
        
        # Statistics
        self.signals_detected = 0
        self.orders_placed = 0
        self.positions_closed = 0
        
        # Signal cooldown - prevent duplicate signals
        self.signal_cooldown: Dict[str, datetime] = {}  # symbol -> last signal time
        self.SIGNAL_COOLDOWN_MINUTES = 30  # Only signal same symbol every 30 min
        
        # Track symbols with active trades (prevents multiple orders per symbol)
        self.traded_symbols: set = set()  # Symbols with open positions
    
    def load_watchlist(self, filepath: str):
        """Load watchlist from CSV."""
        import csv
        
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Watchlist not found: {filepath}")
            return
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('Symbol', '').strip()
                if symbol:
                    beta = get_beta(symbol)
                    self.watchlist.append({
                        'symbol': symbol,
                        'beta': beta
                    })
        
        logger.info(f"Loaded {len(self.watchlist)} symbols from watchlist")
    
    def connect_and_start(self) -> bool:
        """Connect to IB Gateway."""
        logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT}...")
        self.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
        
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect")
            return False
        
        self.connected = True
        logger.info("✓ Connected to IB Gateway")
        return True
    
    def subscribe_all(self):
        """Subscribe to market data for all symbols."""
        for item in self.watchlist:
            symbol = item['symbol']
            self._subscribe_stock(symbol)
            self.entry_lows[symbol] = float('inf')
            self.entry_triggered[symbol] = False
        
        self._subscribe_vix()
        logger.info(f"Subscribed to {len(self.watchlist)} symbols + VIX")
    
    def _subscribe_stock(self, symbol: str):
        """Subscribe to stock market data."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        self.req_to_symbol[req_id] = symbol
        
        self.reqMktData(req_id, contract, "", False, False, [])
    
    def _subscribe_vix(self):
        """Subscribe to VIX."""
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        
        self.vix_req_id = self.next_req_id
        self.next_req_id += 1
        self.reqMktData(self.vix_req_id, contract, "", False, False, [])
    
    def get_historical_bars(self, symbol: str, duration: str = "5 D", bar_size: str = "5 mins") -> Optional[pd.DataFrame]:
        """
        Get historical bars for indicator calculation.
        
        Args:
            symbol: Stock symbol
            duration: IB duration string (e.g., "5 D" for 5 days, "1 W" for 1 week)
            bar_size: Bar size (e.g., "5 mins", "1 min", "1 hour")
        
        Returns:
            DataFrame with OHLCV data for AI signal generation
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
            req_id, contract, "", duration, bar_size,
            "TRADES", 1, 1, False, []
        )
        
        # Wait for data
        start = time.time()
        while not self.bar_complete.get(req_id, False):
            if time.time() - start > 30:  # Increased timeout for more data
                logger.warning(f"Timeout fetching historical data for {symbol}")
                return None
            time.sleep(0.1)
        
        bars = self.bar_data.get(req_id, [])
        if not bars:
            return None
        
        df = pd.DataFrame([{
            'timestamp': b.date,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume
        } for b in bars])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def screen_symbol(self, symbol: str, beta: float) -> Optional[dict]:
        """Screen a symbol for opportunity."""
        price = self.prices.get(symbol, 0)
        prev = self.prev_close.get(symbol, 0)
        
        if not price or not prev:
            return None
        
        actual_pct = (price - prev) / prev * 100
        direction = get_direction(actual_pct)
        exp_pct, _ = expected_move(beta, self.vix_level, price)
        
        if exp_pct == 0:
            return None
        
        abn = abnormality_score(actual_pct, exp_pct)
        
        # Get indicators
        df = self.get_historical_bars(symbol)
        if df is not None and len(df) > 26:
            indicators = get_all_indicators(df)
        else:
            indicators = {
                'macd_state': 'neutral', 'rsi': 50,
                'bb_pos': 'NORMAL', 'volume_ratio': 1.0
            }
        
        score = enhanced_score(
            actual_pct, exp_pct,
            indicators['volume_ratio'],
            indicators['macd_state'],
            indicators['rsi'],
            indicators['bb_pos'],
            direction
        )
        
        return {
            'symbol': symbol,
            'price': price,
            'actual_pct': actual_pct,
            'expected_pct': exp_pct,
            'abnormality': abn,
            'score': score,
            'signal': classify_signal(score),
            'direction': direction,
            'beta': beta,
            **indicators
        }
    
    def check_entry_trigger(self, symbol: str, price: float, beta: float) -> bool:
        """Check if trailing entry is triggered."""
        if self.entry_triggered.get(symbol, False):
            return False
        
        # Update low
        if price < self.entry_lows.get(symbol, float('inf')):
            self.entry_lows[symbol] = price
        
        # Check trigger
        low = self.entry_lows[symbol]
        trail_level = low * (1 + ENTRY_TRAIL_PCT)
        
        if price >= trail_level:
            self.entry_triggered[symbol] = True
            logger.warning(f"*** ENTRY TRIGGER *** {symbol}: ${price:.2f} >= ${trail_level:.2f}")
            return True
        
        return False
    
    def _check_ai_signal(self, symbol: str, beta: float) -> dict:
        """
        Check AI signal for a symbol.
        
        Returns dict with:
        - score: consensus score (0-100)
        - approved: whether trade is approved (auto-execute or user approved)
        - signal_type: BUY_CALL, SELL_CALL, etc.
        - reasons: list of reasons
        """
        if not self.ai_generator:
            return None
        
        # Get settings from dashboard
        settings = get_settings() if AI_SIGNALS_AVAILABLE else {}
        min_score = settings.get('min_score_threshold', AI_MIN_SCORE)
        auto_threshold = settings.get('auto_execute_threshold', AI_AUTO_EXECUTE)
        auto_enabled = settings.get('auto_execute_enabled', True)
        
        # Get historical data for AI analysis
        df = self.get_historical_bars(symbol)
        if df is None or len(df) < 30:
            return None
        
        # Add required columns if missing
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.001
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.999
        
        try:
            signal = self.ai_generator.generate_signal_from_data(df, symbol)
            
            # Check if score meets threshold
            if signal.consensus_score < min_score:
                return {
                    'score': signal.consensus_score,
                    'approved': False,
                    'signal_type': signal.signal_type.value,
                    'reasons': ['Score below minimum threshold']
                }
            
            # Check auto-execute
            if auto_enabled and signal.consensus_score >= auto_threshold:
                return {
                    'score': signal.consensus_score,
                    'approved': True,
                    'signal_type': signal.signal_type.value,
                    'reasons': signal.reasons
                }
            
            # Requires manual approval
            trade_id = add_pending_trade(
                symbol,
                signal.signal_type.value,
                signal.consensus_score,
                signal.reasons
            )
            
            # Check if already approved via dashboard
            approval = check_trade_approved(trade_id)
            
            return {
                'score': signal.consensus_score,
                'approved': approval == True,  # True if approved, False/None otherwise
                'signal_type': signal.signal_type.value,
                'reasons': signal.reasons,
                'trade_id': trade_id
            }
            
        except Exception as e:
            logger.warning(f"AI signal error for {symbol}: {e}")
            return None
    
    def place_option_order(self, symbol: str, underlying_price: float, beta: float, ai_score: float = None):
        """Place option buy order."""
        # Calculate strike and expiry
        strike = round(underlying_price * (1 + STRIKE_OTM_PCT) / 5) * 5
        
        today = datetime.now()
        target = today + timedelta(days=EXPIRY_DAYS)
        days_to_friday = (4 - target.weekday()) % 7
        expiry_date = target + timedelta(days=days_to_friday)
        expiry = expiry_date.strftime("%Y%m%d")
        
        ai_note = f" [AI Score: {ai_score:.0f}]" if ai_score else ""
        logger.info(f"Placing order: BUY 1 {symbol} {expiry} ${strike} CALL{ai_note}")
        
        if not LIVE_MODE:
            # Simulate order fill
            fake_fill_price = underlying_price * 0.02  # ~2% of underlying
            self._record_position(
                symbol, strike, expiry, "C",
                fake_fill_price, underlying_price, beta
            )
            logger.info(f"[SIMULATED] Order filled at ${fake_fill_price:.2f}")
            return
        
        # Build contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = "C"
        contract.multiplier = "100"
        
        # Build order
        order = Order()
        order.orderId = self.next_order_id
        order.action = "BUY"
        order.orderType = "MKT"
        order.totalQuantity = 1
        order.tif = "DAY"
        order.transmit = True
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        self.placeOrder(self.next_order_id, contract, order)
        self.next_order_id += 1
        self.orders_placed += 1
    
    def _record_position(self, symbol: str, strike: float, expiry: str, 
                        right: str, fill_price: float, underlying: float, beta: float):
        """Record a new position."""
        pos = TrackedPosition(
            symbol=symbol,
            option_symbol=f"{symbol}_{expiry}_{strike}_{right}",
            strike=strike,
            expiry=expiry,
            right=right,
            entry_price=fill_price,
            entry_underlying=underlying,
            entry_time=datetime.now(),
            underlying_high=underlying,
            beta=beta
        )
        
        # Set initial stop
        index_vol = self.vol_tracker.get_daily_vol_pct()
        dte = 14  # Approximate
        pos.stop_level = self.stop_calc.compute_underlying_stop(
            underlying, beta, index_vol, dte
        )
        
        self.positions[pos.option_symbol] = pos
        logger.info(f"Position opened: {pos.option_symbol} @ ${fill_price:.2f}, stop=${pos.stop_level:.2f}")
    
    def update_positions(self):
        """Update all positions with current prices and check stops."""
        index_vol = self.vol_tracker.get_daily_vol_pct()
        
        for key, pos in list(self.positions.items()):
            current_price = self.prices.get(pos.symbol, 0)
            if not current_price:
                continue
            
            # Update high and trailing stop
            if current_price > pos.underlying_high:
                pos.underlying_high = current_price
                new_stop = self.stop_calc.compute_trail_from_high(
                    pos.underlying_high, pos.beta, index_vol, 14
                )
                pos.stop_level = max(pos.stop_level, new_stop)
            
            # Check stop trigger
            if current_price <= pos.stop_level and not pos.exit_triggered:
                pos.exit_triggered = True
                pos.exit_time = datetime.now()
                
                # Estimate exit price
                entry_move = pos.entry_underlying - pos.stop_level
                exit_price = pos.entry_price - (entry_move * 0.5 * 0.01)  # Delta approx
                exit_price = max(exit_price, 0.05)
                pos.exit_price = exit_price
                
                pos.pnl = (exit_price - pos.entry_price) * 100
                
                logger.warning(
                    f"*** STOP TRIGGERED *** {pos.symbol}: "
                    f"${current_price:.2f} <= ${pos.stop_level:.2f} | "
                    f"P&L: ${pos.pnl:+.2f}"
                )
                
                self.closed_trades.append(pos)
                del self.positions[key]
                if pos.symbol in self.traded_symbols:
                    self.traded_symbols.discard(pos.symbol)
                self.positions_closed += 1
    
    def run_loop(self, max_iterations: int = 100):
        """Main trading loop."""
        logger.info("\n" + "=" * 60)
        logger.info("TRADING SYSTEM STARTED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'LIVE' if LIVE_MODE else 'SIMULATION'}")
        logger.info(f"Watchlist: {len(self.watchlist)} symbols")
        logger.info(f"Entry trail: {ENTRY_TRAIL_PCT*100:.1f}%")
        logger.info(f"Exit trail: {EXIT_TRAIL_PCT*100:.1f}%")
        logger.info("=" * 60 + "\n")
        
        iteration = 0
        
        while iteration < max_iterations:
            try:
                iteration += 1
                
                # Update VIX tracker
                self.vol_tracker.update_vix(self.vix_level)
                
                # Screen all symbols
                for item in self.watchlist:
                    symbol = item['symbol']
                    beta = item['beta']
                    
                    # Skip if already have position
                    if any(p.symbol == symbol for p in self.positions.values()):
                        continue
                    
                    # Skip if symbol already traded (one contract per symbol at a time)
                    if symbol in self.traded_symbols:
                        continue
                    
                    # Skip if at max positions
                    if len(self.positions) >= MAX_OPEN_POSITIONS:
                        break  # Stop scanning, at limit
                    
                    # Skip if in signal cooldown (prevent duplicate signals)
                    if symbol in self.signal_cooldown:
                        cooldown_elapsed = (datetime.now() - self.signal_cooldown[symbol]).total_seconds() / 60
                        if cooldown_elapsed < self.SIGNAL_COOLDOWN_MINUTES:
                            continue  # Still in cooldown
                    
                    # === AI-FIRST PATH: Check AI signals directly from historical data ===
                    if self.ai_generator:
                        ai_result = self._check_ai_signal(symbol, beta)
                        if ai_result and ai_result.get('score', 0) >= AI_MIN_SCORE:
                            ai_approved = ai_result.get('approved', False)
                            ai_score = ai_result['score']
                            signal_type = ai_result.get('signal_type', 'NO_SIGNAL')
                            
                            if signal_type != 'NO_SIGNAL':
                                self.signals_detected += 1
                                self.signal_cooldown[symbol] = datetime.now()  # Set cooldown
                                logger.info(f">>> AI Signal: {symbol} {signal_type} (score={ai_score:.0f})")
                                
                                # Add to dashboard
                                if AI_SIGNALS_AVAILABLE:
                                    add_signal(symbol, signal_type, ai_score, ai_result.get('reasons', []))
                                
                                # Get current price from historical data if not streaming
                                price = self.prices.get(symbol)
                                if not price:
                                    # Get last close from historical data
                                    df = self.get_historical_bars(symbol)
                                    if df is not None and len(df) > 0:
                                        price = df['close'].iloc[-1]
                                
                                if price and ai_approved:
                                    logger.info(f"  [OK] Auto-executing {symbol} @ ${price:.2f}")
                                    self.traded_symbols.add(symbol)  # Mark as traded
                                    self.place_option_order(symbol, price, beta, ai_score=ai_score)
                                elif price:
                                    logger.info(f"  [PENDING] {symbol} awaiting manual approval (score={ai_score:.0f})")
                            continue
                    
                    # === FALLBACK: Traditional screener path ===
                    # Screen for opportunity (requires streaming price)
                    result = self.screen_symbol(symbol, beta)
                    if not result:
                        continue
                    
                    # Check if signal strength sufficient
                    if result['abnormality'] >= ABN_THRESHOLD and result['score'] >= MIN_SCORE:
                        self.signals_detected += 1
                        
                        # Get AI signal if available
                        ai_approved = True  # Default to traditional behavior
                        ai_score = result['score']
                        
                        if self.ai_generator:
                            ai_result = self._check_ai_signal(symbol, beta)
                            if ai_result:
                                ai_score = ai_result.get('score', result['score'])
                                ai_approved = ai_result.get('approved', True)
                                
                                # Add to dashboard
                                if AI_SIGNALS_AVAILABLE:
                                    add_signal(
                                        symbol, 
                                        ai_result.get('signal_type', 'BUY_CALL'),
                                        ai_score,
                                        ai_result.get('reasons', [])
                                    )
                        
                        if not ai_approved:
                            logger.info(f"{symbol}: AI signal requires approval (score={ai_score:.0f})")
                            continue
                        
                        # Check entry trigger
                        price = self.prices.get(symbol, 0)
                        if self.check_entry_trigger(symbol, price, beta):
                            self.place_option_order(symbol, price, beta, ai_score=ai_score)
                
                # Update existing positions
                self.update_positions()
                
                # Status update
                if iteration % 10 == 0:
                    logger.info(
                        f"[Iter {iteration}] Signals: {self.signals_detected} | "
                        f"Orders: {self.orders_placed} | "
                        f"Open: {len(self.positions)} | "
                        f"Closed: {self.positions_closed} | "
                        f"VIX: {self.vix_level:.1f}"
                    )
                
                time.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("\nStopping...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(SCAN_INTERVAL)
        
        self.print_summary()
    
    def print_summary(self):
        """Print trading summary."""
        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"  Signals Detected:  {self.signals_detected}")
        print(f"  Orders Placed:     {self.orders_placed}")
        print(f"  Positions Opened:  {self.orders_placed}")
        print(f"  Positions Closed:  {self.positions_closed}")
        print(f"  Still Open:        {len(self.positions)}")
        
        if self.closed_trades:
            pnls = [t.pnl for t in self.closed_trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            
            print("\n  Closed Trade P&L:")
            for t in self.closed_trades:
                print(f"    {t.symbol}: ${t.pnl:+.2f}")
            
            print(f"\n  Total P&L: ${sum(pnls):+.2f}")
            print(f"  Win Rate: {len(winners)}/{len(pnls)} ({len(winners)/len(pnls)*100:.1f}%)")
        
        print("=" * 60 + "\n")
    
    # ========== EWrapper Callbacks ==========
    
    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return
        
        if reqId == self.vix_req_id:
            if tickType in [1, 2, 4]:
                self.vix_level = price
            return
        
        symbol = self.req_to_symbol.get(reqId)
        if symbol:
            if tickType == 4:  # Last
                self.prices[symbol] = price
            elif tickType == 9:  # Close
                self.prev_close[symbol] = price
    
    def historicalData(self, reqId, bar):
        if reqId not in self.bar_data:
            self.bar_data[reqId] = []
        self.bar_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId, start, end):
        self.bar_complete[reqId] = True
    
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        logger.info(f"Next order ID: {orderId}")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, *args):
        if status == "Filled":
            logger.info(f"Order {orderId} filled at ${avgFillPrice:.2f}")
    
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode in [2104, 2106, 2158]:
            pass
        elif errorCode in [10167, 10168]:
            logger.debug(f"Data: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_system.log')
        ]
    )
    
    print("\n" + "=" * 60)
    print("END-TO-END TRADING SYSTEM TEST")
    print("=" * 60)
    print(f"Mode: {'LIVE' if LIVE_MODE else 'SIMULATION (no real orders)'}")
    print("=" * 60 + "\n")
    
    system = TradingSystem()
    
    # Load watchlist
    system.load_watchlist(WATCHLIST_PATH)
    
    if not system.watchlist:
        logger.error("No symbols loaded!")
        return
    
    # Connect
    if not system.connect_and_start():
        logger.error("Failed to connect")
        return
    
    # Subscribe to data
    system.subscribe_all()
    
    # Wait for initial data
    logger.info("Waiting for market data...")
    time.sleep(5)
    
    # Run trading loop
    try:
        system.run_loop(max_iterations=100)  # Run for ~15 minutes
    finally:
        system.disconnect()
        logger.info("System stopped.")


if __name__ == "__main__":
    main()
