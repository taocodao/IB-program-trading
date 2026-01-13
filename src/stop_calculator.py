"""
Stop Calculator for Volatility-Aware Options Stop-Loss System
==============================================================

Core formula: stop_distance = k × β × σ_index

Where:
- k = aggression factor (0.7-1.5, tunable)
- β = underlying beta vs S&P 500
- σ_index = index daily volatility (from VIX)

Additional adjustments:
- DTE < 7 days: 2x wider stops (high gamma)
- DTE 7-30 days: 1.5x wider stops
- Clamped between min (4%) and max (40%)
"""

from dataclasses import dataclass
import math
from typing import Optional
from scipy.stats import norm


@dataclass
class StopCalculator:
    """
    Computes risk-sized stop distances using Beta × Index Volatility.
    
    Attributes:
        k_aggression: Scaling factor (0.7=conservative, 1.0=default, 1.5=aggressive)
        min_trail_pct: Minimum trail distance (default 4%)
        max_trail_pct: Maximum trail distance (default 40%)
    """
    
    k_aggression: float = 1.0
    min_trail_pct: float = 0.04   # 4% minimum
    max_trail_pct: float = 0.40   # 40% maximum
    
    # DTE multipliers (4-level, gamma-aware)
    dte_30_plus_mult: float = 1.0     # > 30 days: Normal
    dte_14_30_mult: float = 1.2       # 14-30 days: 20% wider (NEW)
    dte_7_14_mult: float = 1.5        # 7-14 days: 50% wider
    dte_under_7_mult: float = 2.0     # < 7 days: 100% wider
    
    def compute_underlying_stop(
        self,
        entry_price: float,
        beta: float,
        index_vol_pct: float,
        days_to_expiry: int,
        direction: str = "long"  # "long" or "short"
    ) -> float:
        """
        Compute underlying stop level using beta × index volatility.
        
        Args:
            entry_price: Underlying price at option entry
            beta: Stock beta vs S&P 500 (e.g., 1.5)
            index_vol_pct: Daily index volatility as decimal (e.g., 0.012 = 1.2%)
            days_to_expiry: Days until option expires
            direction: "long" (stop below) or "short" (stop above)
        
        Returns:
            Stop price level in dollars
        
        Examples:
            >>> calc = StopCalculator(k_aggression=1.0)
            >>> calc.compute_underlying_stop(100, 1.0, 0.01, 30)
            99.0  # 1% below entry (1.0 × 1.0 × 0.01)
            
            >>> calc.compute_underlying_stop(100, 2.0, 0.02, 30)
            96.0  # 4% below entry (1.0 × 2.0 × 0.02)
        """
        # Base trail: k × β × σ
        base_trail = self.k_aggression * beta * index_vol_pct
        
        # Adjust for short DTE (high gamma options need wider stops)
        # 4-level adjustment for gamma-aware sizing
        if days_to_expiry < 7:
            base_trail *= self.dte_under_7_mult
        elif days_to_expiry < 14:
            base_trail *= self.dte_7_14_mult
        elif days_to_expiry <= 30:
            base_trail *= self.dte_14_30_mult
        # else: > 30 days, use base (dte_30_plus_mult = 1.0)
        
        # Clamp to min/max
        trail_pct = max(self.min_trail_pct, min(base_trail, self.max_trail_pct))
        
        # Convert to dollar move
        stop_distance = entry_price * trail_pct
        
        # Compute stop level
        if direction == "long":
            stop_level = entry_price - stop_distance
        else:  # short
            stop_level = entry_price + stop_distance
        
        return max(stop_level, 0.01)  # Never negative
    
    def compute_trail_from_high(
        self,
        underlying_high: float,
        beta: float,
        index_vol_pct: float,
        days_to_expiry: int
    ) -> float:
        """
        Compute trailing stop level based on highest underlying price.
        
        For trailing stops, we track the high and compute stop from there.
        """
        return self.compute_underlying_stop(
            entry_price=underlying_high,
            beta=beta,
            index_vol_pct=index_vol_pct,
            days_to_expiry=days_to_expiry,
            direction="long"
        )
    
    def get_trail_percentage(
        self,
        beta: float,
        index_vol_pct: float,
        days_to_expiry: int
    ) -> float:
        """
        Get the trail percentage without computing price levels.
        Useful for display and logging.
        """
        base_trail = self.k_aggression * beta * index_vol_pct
        
        # 4-level DTE adjustment
        if days_to_expiry < 7:
            base_trail *= self.dte_under_7_mult
        elif days_to_expiry < 14:
            base_trail *= self.dte_7_14_mult
        elif days_to_expiry <= 30:
            base_trail *= self.dte_14_30_mult
        
        return max(self.min_trail_pct, min(base_trail, self.max_trail_pct))


# ============= Black-Scholes Computation =============

def black_scholes_call(
    S: float,      # Underlying price
    K: float,      # Strike price
    T: float,      # Time to expiry (years)
    r: float,      # Risk-free rate
    sigma: float   # Implied volatility (annualized)
) -> float:
    """
    Compute Black-Scholes call option price.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years (e.g., 30 days = 30/365)
        r: Risk-free interest rate (e.g., 0.05 for 5%)
        sigma: Annualized implied volatility (e.g., 0.25 for 25%)
    
    Returns:
        Theoretical call option price
    """
    if T <= 0 or sigma <= 0:
        # At/past expiry or zero vol
        return max(S - K, 0)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(call, 0)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Compute Black-Scholes put option price using put-call parity.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    call = black_scholes_call(S, K, T, r, sigma)
    put = call - S + K * math.exp(-r * T)
    return max(put, 0)


def compute_theoretical_price(
    underlying_price: float,
    strike: float,
    days_to_expiry: int,
    implied_vol: float,
    right: str = "C",
    rate: float = 0.05
) -> float:
    """
    Compute theoretical option price at a given underlying level.
    
    Args:
        underlying_price: Underlying spot price
        strike: Option strike price
        days_to_expiry: Days until expiration
        implied_vol: IV as decimal (0.25 = 25%)
        right: "C" for call, "P" for put
        rate: Risk-free rate (default 5%)
    
    Returns:
        Theoretical option price
    """
    T = days_to_expiry / 365.0
    
    if T <= 0:
        # Expired
        if right == "C":
            return max(underlying_price - strike, 0)
        else:
            return max(strike - underlying_price, 0)
    
    if right == "C":
        return black_scholes_call(underlying_price, strike, T, rate, implied_vol)
    else:
        return black_scholes_put(underlying_price, strike, T, rate, implied_vol)


def compute_smart_limit_price(
    current_bid: float,
    current_ask: float,
    theoretical_price: float,
    spread_participation: float = 0.5,
    allowed_slippage_pct: float = 0.03
) -> float:
    """
    Compute smart limit price for a SELL order.
    
    Strategy:
    - Never below bid (worst case)
    - Never above theoretical (overly optimistic)
    - Target: somewhere in the bid-ask spread
    
    Args:
        current_bid: Current bid price
        current_ask: Current ask price
        theoretical_price: Black-Scholes theoretical value
        spread_participation: How deep into spread (0=bid, 1=ask)
        allowed_slippage_pct: Max slippage below theoretical
    
    Returns:
        Limit price for sell order
    """
    spread = current_ask - current_bid
    
    # Spread-based target
    spread_target = current_bid + spread * spread_participation
    
    # Theoretical with slippage floor
    theo_min = theoretical_price * (1 - allowed_slippage_pct)
    
    # Take the better of spread-based and theo-based
    limit = max(spread_target, theo_min)
    
    # Cap at theoretical (don't expect more)
    limit = min(limit, theoretical_price)
    
    # Floor at bid (never below)
    limit = max(limit, current_bid)
    
    return round(limit, 2)
