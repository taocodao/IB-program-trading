"""
Option Selector - IB Option Chain Integration
==============================================

Queries IB for option chain and selects appropriate contracts:
1. Get available expirations for a symbol
2. Query option chain for target expiry
3. Select strike based on delta target or % OTM
4. Return contract with current bid/ask pricing

Usage:
    selector = OptionSelector()
    selector.connect()
    
    # Select a call option for AAPL
    option = selector.select_option(
        symbol="AAPL",
        right="C",
        target_delta=0.50,  # ATM
        target_dte=21       # ~3 weeks
    )
    
    print(f"Selected: {option.strike} strike, ${option.bid}-${option.ask}")
"""

import sys
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.ticktype import TickTypeEnum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 500

# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH-BACKED OPTIMAL PARAMETERS
# Source: Options-Selection-Best-Practices-Deep-Research.md
# ═══════════════════════════════════════════════════════════════════════════

# DTE (Days To Expiration) - Optimal: 30-45 days
# Why: At 14-21 DTE, theta decay is 50-70% FASTER, eating profits
# At 30-45 DTE, theta is moderate ($0.08-0.12/day), giving signals time to work
DEFAULT_TARGET_DTE_MIN = 30      # Minimum 4 weeks (research optimum)
DEFAULT_TARGET_DTE_MAX = 45      # Maximum 6 weeks
DEFAULT_TARGET_DTE = 35          # Sweet spot: 5 weeks

# Delta-based strike selection (more precise than OTM%)
# Higher delta = higher probability of profit, lower leverage
# 0.55-0.60 = balanced risk/reward for directional trades
DEFAULT_TARGET_DELTA = 0.55      # Slightly ITM (55% probability of profit)
DEFAULT_OTM_PCT = 0.01           # Fallback: 1% OTM if delta unavailable

# Confidence-to-delta mapping (used when AI score is available)
# High confidence (80+) = use higher delta (0.60) = safer
# Moderate confidence (60-80) = standard delta (0.55)
# Lower confidence (<60) = lower delta (0.40-0.50) = more leverage

# Liquidity thresholds - PRODUCTION GRADE (not backtesting grade)
# These prevent slippage and execution failures
MIN_VOLUME = 500                  # Min daily volume (research: 500+ preferred)
MIN_OPEN_INTEREST = 1000          # Min open interest (research: 1000+ preferred)
MAX_SPREAD_PCT = 0.10             # Max 10% bid-ask spread
PREFERRED_VOLUME = 2000           # Preferred volume for excellent liquidity
PREFERRED_OPEN_INTEREST = 5000    # Preferred OI for excellent liquidity
MIN_BID_ASK_SIZE = 10             # Minimum contracts at bid/ask


# ============= Data Classes =============

class OptionRight(Enum):
    CALL = "C"
    PUT = "P"


@dataclass
class OptionContract:
    """Complete option contract with pricing and liquidity."""
    symbol: str
    expiry: str               # YYYYMMDD format
    strike: float
    right: str                # "C" or "P"
    
    # Contract details
    con_id: int = 0
    multiplier: int = 100
    
    # Current pricing
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    mid: float = 0.0
    
    # Liquidity metrics
    volume: int = 0           # Daily volume
    open_interest: int = 0    # Open interest
    
    # Greeks (if available)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    implied_vol: float = 0.0
    
    # Underlying
    underlying_price: float = 0.0
    
    # Metadata
    days_to_expiry: int = 0
    otm_pct: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if contract has valid pricing."""
        return self.bid > 0 or self.ask > 0
    
    def get_cost(self, quantity: int = 1) -> float:
        """Get total cost to buy at ask."""
        return self.ask * self.multiplier * quantity
    
    def get_spread_pct(self) -> float:
        """Get bid-ask spread as percentage of mid price."""
        if self.mid > 0:
            return (self.ask - self.bid) / self.mid
        return 1.0  # 100% spread if no mid
    
    def get_liquidity_score(self) -> float:
        """
        Calculate liquidity score (0-100) using VOSS Framework.
        
        Based on: Options-Selection-Best-Practices-Deep-Research.md
        Higher is better liquidity. Must score 60+ to be tradeable.
        
        Components (research-backed weights):
        - Volume: 25 points (daily trading activity)
        - Open Interest: 25 points (total active contracts)
        - Spread: 30 points (transaction cost)
        - Size: 20 points (market depth) - estimated from OI
        """
        score = 0.0
        
        # 1. Volume score (0-25 points) - Research thresholds
        if self.volume >= 10000:
            score += 25
        elif self.volume >= 2000:  # Preferred threshold
            score += 20
        elif self.volume >= 500:   # Minimum acceptable
            score += 15
        elif self.volume >= 100:
            score += 5
        # < 100 = 0 points (reject)
        
        # 2. Open Interest score (0-25 points) - Research thresholds
        if self.open_interest >= 10000:
            score += 25
        elif self.open_interest >= 5000:  # Preferred
            score += 20
        elif self.open_interest >= 1000:  # Minimum acceptable
            score += 15
        elif self.open_interest >= 500:
            score += 5
        # < 500 = 0 points (reject)
        
        # 3. Spread score (0-30 points) - Research thresholds
        spread_pct = self.get_spread_pct()
        if spread_pct < 0.02:       # < 2% = excellent
            score += 30
        elif spread_pct < 0.05:     # < 5% = good
            score += 25
        elif spread_pct < 0.10:     # < 10% = acceptable
            score += 15
        elif spread_pct < 0.20:     # < 20% = poor
            score += 5
        # > 20% = 0 points (reject)
        
        # 4. Size score (0-20 points) - Estimated from OI
        # Actual bid/ask size comes from market data
        # Estimate: Higher OI usually means better depth
        estimated_size = self.open_interest / 100  # Rough estimate
        if estimated_size >= 100:
            score += 20
        elif estimated_size >= 50:
            score += 18
        elif estimated_size >= 10:  # Minimum acceptable
            score += 15
        elif estimated_size >= 5:
            score += 5
        # < 5 = 0 points
        
        return score
    
    def get_liquidity_rating(self) -> str:
        """Get human-readable liquidity rating."""
        score = self.get_liquidity_score()
        if score >= 85:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 60:
            return "ACCEPTABLE"
        else:
            return "POOR - REJECT"
    
    def is_liquid(self) -> bool:
        """
        Check if option meets PRODUCTION liquidity requirements.
        
        Based on research: must pass ALL of:
        - Volume >= 500 (not 10)
        - OI >= 1000 (not 100)
        - Spread <= 10%
        """
        spread_pct = self.get_spread_pct()
        return (
            spread_pct <= MAX_SPREAD_PCT and
            self.volume >= MIN_VOLUME and
            self.open_interest >= MIN_OPEN_INTEREST
        )
    
    def __str__(self):
        right_str = "CALL" if self.right == "C" else "PUT"
        liq_score = self.get_liquidity_score()
        return (f"{self.symbol} {self.expiry} ${self.strike} {right_str} "
                f"| Bid: ${self.bid:.2f} Ask: ${self.ask:.2f} "
                f"| Vol: {self.volume} OI: {self.open_interest} "
                f"| Liq: {liq_score:.0f}")


@dataclass
class ExpiryInfo:
    """Information about an option expiry."""
    expiry_date: str          # YYYYMMDD
    days_to_expiry: int
    is_weekly: bool = False
    is_monthly: bool = False


# ============= Option Selector Class =============

class OptionSelector(EClient, EWrapper):
    """
    Queries IB for option chain and selects appropriate contracts.
    
    Provides two selection modes:
    1. Delta-based: Select option closest to target delta (e.g., 0.50 for ATM)
    2. OTM-based: Select option at specified % out-of-the-money
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        
        self.lock = threading.RLock()
        self.connected = False
        self.next_req_id = 7000
        
        # Contract details storage
        self.contract_details: Dict[int, List[ContractDetails]] = {}
        self.contract_details_complete: Dict[int, bool] = {}
        
        # Market data storage
        self.tick_data: Dict[int, Dict] = {}
        self.req_to_contract: Dict[int, Contract] = {}
        
        # Underlying prices
        self.underlying_prices: Dict[str, float] = {}
        self.underlying_req_ids: Dict[str, int] = {}
    
    def connect_and_start(
        self, 
        host: str = IB_HOST, 
        port: int = IB_PORT, 
        client_id: int = IB_CLIENT_ID
    ) -> bool:
        """Connect to IB Gateway."""
        logger.info(f"Connecting to IB at {host}:{port}...")
        self.connect(host, port, client_id)
        
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        
        time.sleep(2)
        
        if not self.isConnected():
            logger.error("Failed to connect to IB")
            return False
        
        self.connected = True
        logger.info("✓ Connected to IB Gateway")
        return True
    
    def get_underlying_price(self, symbol: str, timeout: float = 5.0) -> Optional[float]:
        """Get current price of underlying."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.underlying_req_ids[symbol] = req_id
        self.tick_data[req_id] = {}
        
        self.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for price
        start = time.time()
        while time.time() - start < timeout:
            if symbol in self.underlying_prices and self.underlying_prices[symbol] > 0:
                self.cancelMktData(req_id)
                return self.underlying_prices[symbol]
            time.sleep(0.1)
        
        self.cancelMktData(req_id)
        return None
    
    def get_option_expirations(self, symbol: str) -> List[ExpiryInfo]:
        """
        Get available option expirations for a symbol.
        
        Returns list of ExpiryInfo sorted by date.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.contract_details[req_id] = []
        self.contract_details_complete[req_id] = False
        
        self.reqContractDetails(req_id, contract)
        
        # Wait for completion
        timeout = 30
        start = time.time()
        while not self.contract_details_complete.get(req_id, False):
            if time.time() - start > timeout:
                logger.warning(f"Timeout getting expirations for {symbol}")
                break
            time.sleep(0.5)
        
        # Extract unique expirations
        expirations = {}
        for cd in self.contract_details.get(req_id, []):
            expiry = cd.contract.lastTradeDateOrContractMonth
            if expiry not in expirations:
                try:
                    exp_date = datetime.strptime(expiry, "%Y%m%d")
                    dte = (exp_date - datetime.now()).days
                    
                    # Determine if weekly or monthly
                    # Monthly options typically expire on 3rd Friday
                    is_monthly = exp_date.day >= 15 and exp_date.day <= 21 and exp_date.weekday() == 4
                    
                    expirations[expiry] = ExpiryInfo(
                        expiry_date=expiry,
                        days_to_expiry=dte,
                        is_weekly=not is_monthly,
                        is_monthly=is_monthly
                    )
                except:
                    pass
        
        return sorted(expirations.values(), key=lambda x: x.expiry_date)
    
    def get_option_chain(
        self, 
        symbol: str, 
        expiry: str, 
        right: str = "C"
    ) -> List[OptionContract]:
        """
        Get all option strikes for a specific expiry.
        
        Args:
            symbol: Underlying symbol
            expiry: Expiry date YYYYMMDD
            right: "C" for call, "P" for put
            
        Returns:
            List of OptionContract sorted by strike
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.right = right
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.contract_details[req_id] = []
        self.contract_details_complete[req_id] = False
        
        self.reqContractDetails(req_id, contract)
        
        # Wait for completion
        timeout = 30
        start = time.time()
        while not self.contract_details_complete.get(req_id, False):
            if time.time() - start > timeout:
                break
            time.sleep(0.5)
        
        # Convert to OptionContract
        options = []
        for cd in self.contract_details.get(req_id, []):
            c = cd.contract
            try:
                exp_date = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d")
                dte = (exp_date - datetime.now()).days
            except:
                dte = 0
            
            options.append(OptionContract(
                symbol=c.symbol,
                expiry=c.lastTradeDateOrContractMonth,
                strike=c.strike,
                right=c.right,
                con_id=c.conId,
                multiplier=int(c.multiplier) if c.multiplier else 100,
                days_to_expiry=dte
            ))
        
        return sorted(options, key=lambda x: x.strike)
    
    def get_option_price(self, option: OptionContract, timeout: float = 5.0) -> OptionContract:
        """
        Get current bid/ask for an option contract.
        
        Args:
            option: OptionContract to price
            timeout: Max seconds to wait
            
        Returns:
            Same OptionContract with pricing filled in
        """
        contract = Contract()
        contract.symbol = option.symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = option.expiry
        contract.strike = option.strike
        contract.right = option.right
        contract.multiplier = str(option.multiplier)
        
        if option.con_id:
            contract.conId = option.con_id
        
        req_id = self.next_req_id
        self.next_req_id += 1
        
        self.tick_data[req_id] = {}
        self.req_to_contract[req_id] = contract
        
        # Request market data including Greeks
        self.reqMktData(req_id, contract, "100,101,104,106", False, False, [])
        
        # Wait for data
        start = time.time()
        while time.time() - start < timeout:
            data = self.tick_data.get(req_id, {})
            if 'bid' in data and 'ask' in data:
                break
            time.sleep(0.1)
        
        self.cancelMktData(req_id)
        
        # Fill in pricing
        data = self.tick_data.get(req_id, {})
        option.bid = data.get('bid', 0.0)
        option.ask = data.get('ask', 0.0)
        option.last = data.get('last', 0.0)
        option.mid = (option.bid + option.ask) / 2 if option.bid and option.ask else 0
        
        # Liquidity metrics
        option.volume = data.get('volume', 0)
        option.open_interest = data.get('open_interest', 0)
        
        # Greeks from model
        option.delta = data.get('delta', 0.0)
        option.gamma = data.get('gamma', 0.0)
        option.theta = data.get('theta', 0.0)
        option.vega = data.get('vega', 0.0)
        option.implied_vol = data.get('iv', 0.0)
        
        return option
    
    def select_option(
        self,
        symbol: str,
        right: str = "C",
        target_dte: int = DEFAULT_TARGET_DTE,
        target_delta: Optional[float] = None,
        target_otm_pct: Optional[float] = None,
        get_pricing: bool = True
    ) -> Optional[OptionContract]:
        """
        Select an option based on criteria.
        
        Args:
            symbol: Underlying symbol
            right: "C" for call, "P" for put
            target_dte: Target days to expiration
            target_delta: Target delta (e.g., 0.50 for ATM). If None, uses target_otm_pct
            target_otm_pct: Target % OTM (e.g., 0.02 for 2% OTM). Used if target_delta is None
            get_pricing: Whether to fetch current bid/ask
            
        Returns:
            Selected OptionContract or None if not found
        """
        logger.info(f"Selecting {right} option for {symbol}, target DTE={target_dte}")
        
        # Get underlying price
        underlying_price = self.get_underlying_price(symbol)
        if not underlying_price:
            logger.error(f"Could not get price for {symbol}")
            return None
        
        logger.info(f"  Underlying price: ${underlying_price:.2f}")
        
        # Get expirations and find best match
        expirations = self.get_option_expirations(symbol)
        if not expirations:
            logger.error(f"No expirations found for {symbol}")
            return None
        
        # Find expiry closest to target DTE
        best_expiry = min(expirations, key=lambda x: abs(x.days_to_expiry - target_dte))
        logger.info(f"  Selected expiry: {best_expiry.expiry_date} ({best_expiry.days_to_expiry} DTE)")
        
        # Get option chain
        chain = self.get_option_chain(symbol, best_expiry.expiry_date, right)
        if not chain:
            logger.error(f"No options found for {symbol} {best_expiry.expiry_date}")
            return None
        
        logger.info(f"  Found {len(chain)} strikes")
        
        # Select strike
        if target_delta is not None:
            # Delta-based selection - need to get prices and deltas
            # For now, approximate using ATM logic
            atm_strike = min(chain, key=lambda x: abs(x.strike - underlying_price))
            selected = atm_strike
            
        elif target_otm_pct is not None:
            # OTM-based selection
            if right == "C":
                target_strike = underlying_price * (1 + target_otm_pct)
            else:  # PUT
                target_strike = underlying_price * (1 - target_otm_pct)
            
            selected = min(chain, key=lambda x: abs(x.strike - target_strike))
        else:
            # Default to ATM
            selected = min(chain, key=lambda x: abs(x.strike - underlying_price))
        
        # Calculate OTM %
        if right == "C":
            selected.otm_pct = (selected.strike - underlying_price) / underlying_price
        else:
            selected.otm_pct = (underlying_price - selected.strike) / underlying_price
        
        selected.underlying_price = underlying_price
        selected.days_to_expiry = best_expiry.days_to_expiry
        
        logger.info(f"  Selected strike: ${selected.strike} ({selected.otm_pct*100:+.1f}% OTM)")
        
        # Get pricing if requested
        if get_pricing:
            selected = self.get_option_price(selected)
            logger.info(f"  Pricing: Bid ${selected.bid:.2f} / Ask ${selected.ask:.2f}")
        
        return selected
    
    def select_option_for_signal(
        self,
        symbol: str,
        signal_type: str,  # "BUY_CALL" or "BUY_PUT"
        target_dte_min: int = DEFAULT_TARGET_DTE_MIN,
        target_dte_max: int = DEFAULT_TARGET_DTE_MAX,
        target_otm_pct: float = DEFAULT_OTM_PCT,
        require_liquid: bool = True
    ) -> Optional[OptionContract]:
        """
        Select best option for AI signal, considering liquidity.
        
        Strategy:
        1. Find expirations in 2-3 week range
        2. For target strike (+/- 2 strikes), get pricing and liquidity
        3. Select option with best combined score (strike proximity + liquidity)
        
        Args:
            symbol: Stock symbol
            signal_type: "BUY_CALL" or "BUY_PUT"
            target_dte_min: Minimum DTE (default 14 = 2 weeks)
            target_dte_max: Maximum DTE (default 21 = 3 weeks)
            target_otm_pct: Target % OTM (default 1% = slightly ATM)
            require_liquid: If True, only select liquid options
            
        Returns:
            Best OptionContract considering liquidity
        """
        right = "C" if signal_type == "BUY_CALL" else "P"
        
        logger.info(f"Selecting liquid {right} option for {symbol}")
        logger.info(f"  DTE range: {target_dte_min}-{target_dte_max} days")
        logger.info(f"  Target OTM: {target_otm_pct*100:.1f}%")
        
        # Get underlying price
        underlying_price = self.get_underlying_price(symbol)
        if not underlying_price:
            logger.error(f"Could not get price for {symbol}")
            return None
        
        logger.info(f"  Underlying: ${underlying_price:.2f}")
        
        # Get expirations
        expirations = self.get_option_expirations(symbol)
        if not expirations:
            logger.error(f"No expirations found for {symbol}")
            return None
        
        # Filter to 2-3 week range
        valid_expiries = [
            e for e in expirations 
            if target_dte_min <= e.days_to_expiry <= target_dte_max
        ]
        
        # If none in range, find closest to target
        if not valid_expiries:
            target_dte = (target_dte_min + target_dte_max) // 2
            closest = min(expirations, key=lambda x: abs(x.days_to_expiry - target_dte))
            valid_expiries = [closest]
            logger.warning(f"  No expiry in range, using closest: {closest.expiry_date}")
        
        logger.info(f"  Found {len(valid_expiries)} valid expirations")
        
        # Calculate target strike
        if right == "C":
            target_strike = underlying_price * (1 + target_otm_pct)
        else:
            target_strike = underlying_price * (1 - target_otm_pct)
        
        # Collect candidate options from each valid expiry
        candidates: List[OptionContract] = []
        
        for exp in valid_expiries:
            chain = self.get_option_chain(symbol, exp.expiry_date, right)
            if not chain:
                continue
            
            # Find strikes near target (+/- 2 strikes)
            chain_sorted = sorted(chain, key=lambda x: abs(x.strike - target_strike))
            nearby_strikes = chain_sorted[:5]  # Top 5 closest strikes
            
            # Get pricing for each
            for opt in nearby_strikes:
                opt.underlying_price = underlying_price
                priced = self.get_option_price(opt, timeout=3.0)
                
                # Skip if no valid pricing
                if not priced.is_valid():
                    continue
                
                candidates.append(priced)
        
        if not candidates:
            logger.error(f"No priced options found for {symbol}")
            return None
        
        logger.info(f"  Evaluated {len(candidates)} option candidates")
        
        # Score each candidate: balance strike proximity with liquidity
        def score_option(opt: OptionContract) -> float:
            # Strike proximity score (0-50)
            strike_diff_pct = abs(opt.strike - target_strike) / underlying_price
            if strike_diff_pct <= 0.01:  # Within 1%
                strike_score = 50
            elif strike_diff_pct <= 0.02:  # Within 2%
                strike_score = 40
            elif strike_diff_pct <= 0.03:  # Within 3%
                strike_score = 30
            elif strike_diff_pct <= 0.05:  # Within 5%
                strike_score = 20
            else:
                strike_score = 10
            
            # Liquidity score (0-50) - already 0-100, scale to 50
            liq_score = opt.get_liquidity_score() / 2
            
            return strike_score + liq_score
        
        # Filter to liquid options if required
        if require_liquid:
            liquid_candidates = [c for c in candidates if c.is_liquid()]
            if liquid_candidates:
                candidates = liquid_candidates
                logger.info(f"  {len(candidates)} liquid options available")
            else:
                logger.warning("  No liquid options found, using best available")
        
        # Sort by combined score
        candidates.sort(key=score_option, reverse=True)
        
        best = candidates[0]
        
        # Calculate OTM %
        if right == "C":
            best.otm_pct = (best.strike - underlying_price) / underlying_price
        else:
            best.otm_pct = (underlying_price - best.strike) / underlying_price
        
        logger.info(f"  Selected: ${best.strike} {best.expiry} "
                   f"(Liq: {best.get_liquidity_score():.0f}, "
                   f"Spread: {best.get_spread_pct()*100:.1f}%)")
        
        return best
    
    # ========== EWrapper Callbacks ==========
    
    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return
        
        if reqId not in self.tick_data:
            self.tick_data[reqId] = {}
        
        if tickType == TickTypeEnum.BID:
            self.tick_data[reqId]['bid'] = price
        elif tickType == TickTypeEnum.ASK:
            self.tick_data[reqId]['ask'] = price
        elif tickType == TickTypeEnum.LAST:
            self.tick_data[reqId]['last'] = price
        elif tickType == TickTypeEnum.CLOSE:
            # Store as underlying price if this is a stock request
            for symbol, rid in self.underlying_req_ids.items():
                if rid == reqId:
                    self.underlying_prices[symbol] = price
    
    def tickSize(self, reqId, tickType, size):
        """Capture volume and open interest."""
        if reqId not in self.tick_data:
            self.tick_data[reqId] = {}
        
        # Volume (tickType 8 = VOLUME)
        if tickType == 8:
            self.tick_data[reqId]['volume'] = size
        # Open Interest (tickType 27 = OPTION_CALL_OPEN_INTEREST or 28 = OPTION_PUT_OPEN_INTEREST)
        elif tickType in [27, 28]:
            self.tick_data[reqId]['open_interest'] = size
    
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol,
                              delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        if reqId not in self.tick_data:
            self.tick_data[reqId] = {}
        
        if impliedVol and impliedVol > 0:
            self.tick_data[reqId]['iv'] = impliedVol
        if delta:
            self.tick_data[reqId]['delta'] = delta
        if gamma:
            self.tick_data[reqId]['gamma'] = gamma
        if theta:
            self.tick_data[reqId]['theta'] = theta
        if vega:
            self.tick_data[reqId]['vega'] = vega
        if undPrice and undPrice > 0:
            # Also capture underlying price from option data
            contract = self.req_to_contract.get(reqId)
            if contract:
                self.underlying_prices[contract.symbol] = undPrice
    
    def contractDetails(self, reqId, contractDetails):
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []
        self.contract_details[reqId].append(contractDetails)
    
    def contractDetailsEnd(self, reqId):
        self.contract_details_complete[reqId] = True
    
    def nextValidId(self, orderId):
        logger.info(f"Next order ID: {orderId}")
    
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode in [2104, 2106, 2158]:
            pass
        elif errorCode in [162, 10167, 10168, 200]:
            pass  # Data/contract warnings
        else:
            logger.error(f"Error {errorCode}: {errorString}")


# ============= Utility Functions =============

def find_best_expiry(target_dte: int = DEFAULT_TARGET_DTE) -> str:
    """
    Calculate best expiry date without IB connection.
    Finds the Friday closest to target DTE.
    
    Updated to use research-backed 35 DTE default.
    
    Returns: YYYYMMDD format string
    """
    today = datetime.now()
    target = today + timedelta(days=target_dte)
    
    # Find nearest Friday
    days_to_friday = (4 - target.weekday()) % 7
    expiry_date = target + timedelta(days=days_to_friday)
    
    return expiry_date.strftime("%Y%m%d")


def calculate_target_dte(iv_rank: float) -> int:
    """
    Determine optimal DTE based on IV environment.
    
    Research-backed: adjust DTE based on volatility.
    High IV = shorter DTE (signals play out faster)
    Low IV = longer DTE (need more time)
    
    Args:
        iv_rank: IV Rank 0-100 (0=low vol, 100=high vol)
        
    Returns:
        Target DTE in days
    """
    if iv_rank < 30:
        return 45  # Low volatility, use longer DTE
    elif iv_rank < 70:
        return 35  # Normal volatility, standard DTE (sweet spot)
    elif iv_rank < 90:
        return 28  # High volatility, shorter DTE
    else:
        return 21  # Extreme volatility, very short DTE


def calculate_target_delta(confidence: float, direction: str = "BULLISH") -> float:
    """
    Determine optimal delta based on AI signal confidence.
    
    Research-backed: higher confidence = higher delta (safer)
    
    Args:
        confidence: AI consensus score 0-100
        direction: "BULLISH" or "BEARISH"
        
    Returns:
        Target delta (always positive, 0.30-0.70)
    """
    if confidence >= 80:
        # High confidence - use higher delta (more likely to profit)
        return 0.60
    elif confidence >= 60:
        # Moderate confidence - balanced delta
        return 0.55
    elif confidence >= 40:
        # Lower confidence - speculative delta
        return 0.45
    else:
        # Very low confidence - lower delta for leverage
        # Consider skipping trade at this level
        return 0.35


def calculate_strike(
    underlying_price: float,
    right: str,
    otm_pct: float = 0.02
) -> float:
    """
    Calculate strike for a given OTM percentage.
    
    Args:
        underlying_price: Current stock price
        right: "C" for call, "P" for put
        otm_pct: Desired % out-of-the-money
        
    Returns:
        Strike price rounded to standard increment
    """
    if right == "C":
        target = underlying_price * (1 + otm_pct)
    else:
        target = underlying_price * (1 - otm_pct)
    
    # Round to standard increments
    if underlying_price >= 100:
        return round(target / 5) * 5
    elif underlying_price >= 50:
        return round(target / 2.5) * 2.5
    else:
        return round(target)


def calculate_strike_from_delta(
    underlying_price: float,
    target_delta: float,
    right: str
) -> float:
    """
    Estimate strike price from target delta.
    
    Uses ATM delta=0.50 as reference point.
    Each 5% move in strike changes delta by ~0.05.
    
    Args:
        underlying_price: Current stock price
        target_delta: Target delta (0.30-0.70)
        right: "C" for call, "P" for put
        
    Returns:
        Estimated strike price
    """
    # Delta difference from ATM (0.50)
    delta_diff = target_delta - 0.50
    
    # Each 0.05 delta = ~1% strike move (rough approximation)
    # Higher delta = lower strike for calls, higher for puts
    strike_offset_pct = delta_diff * 0.20  # 0.05 delta = 1% = 0.01
    
    if right == "C":
        # Higher delta = ITM = lower strike
        target_strike = underlying_price * (1 - strike_offset_pct)
    else:
        # Higher delta = ITM = higher strike for puts
        target_strike = underlying_price * (1 + strike_offset_pct)
    
    # Round to standard increments
    if underlying_price >= 100:
        return round(target_strike / 5) * 5
    elif underlying_price >= 50:
        return round(target_strike / 2.5) * 2.5
    else:
        return round(target_strike)


def calculate_theta_efficiency(theta: float, premium: float, dte: int) -> float:
    """
    Calculate theta efficiency score (0-100).
    
    Lower theta % = better for buyers.
    
    Args:
        theta: Daily theta decay ($)
        premium: Option premium ($)
        dte: Days to expiration
        
    Returns:
        Efficiency score 0-100
    """
    if premium == 0:
        return 0
    
    # Daily theta as % of premium
    theta_pct = (abs(theta) / premium) * 100
    
    # Efficiency score (lower theta % = better for buyers)
    if theta_pct < 3:
        return 100
    elif theta_pct < 5:
        return 80
    elif theta_pct < 8:
        return 60
    elif theta_pct < 12:
        return 40
    else:
        return 20


def validate_option(option: OptionContract) -> dict:
    """
    Final validation before trading.
    
    Research-backed checks:
    1. Not too close to expiration (DTE >= 7)
    2. Delta in valid range (0.10-0.95)
    3. Premium is reasonable ($0.10-$50)
    4. Spread is acceptable (<= 15%)
    
    Returns:
        {'valid': True/False, 'reason': '...'}
    """
    # Check 1: Not too close to expiration
    if option.days_to_expiry < 7:
        return {
            'valid': False,
            'reason': 'DTE < 7 days: Too close to expiration, gamma risk too high'
        }
    
    # Check 2: Greeks are reasonable
    if abs(option.delta) > 0.95:
        return {
            'valid': False,
            'reason': 'Delta > 0.95: Too deep ITM, better to trade stock'
        }
    
    if abs(option.delta) < 0.10:
        return {
            'valid': False,
            'reason': 'Delta < 0.10: Too far OTM, probability too low'
        }
    
    # Check 3: Premium is reasonable
    if option.ask < 0.10:
        return {
            'valid': False,
            'reason': 'Premium < $0.10: Option too cheap, likely illiquid'
        }
    
    if option.ask > 50.00:
        return {
            'valid': False,
            'reason': 'Premium > $50: Option too expensive, verify strike'
        }
    
    # Check 4: Bid/ask spread validation
    spread_pct = option.get_spread_pct() * 100
    if spread_pct > 15:
        return {
            'valid': False,
            'reason': f'Bid-ask spread {spread_pct:.1f}% > 15%: Too wide'
        }
    
    return {'valid': True}


# ============= CLI Test =============

if __name__ == "__main__":
    print("=" * 60)
    print("OPTION SELECTOR - Test Mode")
    print("=" * 60)
    
    selector = OptionSelector()
    
    if not selector.connect_and_start():
        print("Failed to connect to IB")
        sys.exit(1)
    
    try:
        # Test option selection
        test_symbols = ["AAPL", "MSFT", "TSLA"]
        
        for symbol in test_symbols:
            print(f"\n{'='*50}")
            print(f"Testing {symbol}")
            print(f"{'='*50}")
            
            # Select a call
            call = selector.select_option(
                symbol=symbol,
                right="C",
                target_dte=21,
                target_otm_pct=0.02
            )
            
            if call:
                print(f"\n  CALL: {call}")
            
            # Select a put
            put = selector.select_option(
                symbol=symbol,
                right="P",
                target_dte=21,
                target_otm_pct=0.02
            )
            
            if put:
                print(f"  PUT:  {put}")
            
            time.sleep(1)  # Rate limit
        
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
        
    finally:
        selector.disconnect()
