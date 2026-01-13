"""
Unit Tests for Stop Calculator
==============================

Tests the core volatility-aware stop calculation logic.
Run with: python -m pytest tests/test_stop_calculator.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stop_calculator import (
    StopCalculator, 
    compute_theoretical_price,
    compute_smart_limit_price,
    black_scholes_call,
    black_scholes_put
)


class TestStopCalculator:
    """Tests for StopCalculator class."""
    
    def test_basic_stop_computation(self):
        """
        Basic test: entry=100, beta=1.0, vol=1% → 1% trail = 99 stop
        """
        calc = StopCalculator(k_aggression=1.0)
        
        stop = calc.compute_underlying_stop(
            entry_price=100.0,
            beta=1.0,
            index_vol_pct=0.01,  # 1%
            days_to_expiry=30
        )
        
        # 1.0 × 1.0 × 0.01 = 0.01 (1%)
        # But clamped to min 4%
        assert stop == 96.0  # 4% min trail
    
    def test_high_beta_high_vol(self):
        """
        High risk: beta=2.0, vol=3% → 6% trail
        """
        calc = StopCalculator(k_aggression=1.0, min_trail_pct=0.01)
        
        stop = calc.compute_underlying_stop(
            entry_price=100.0,
            beta=2.0,
            index_vol_pct=0.03,  # 3%
            days_to_expiry=30
        )
        
        # 1.0 × 2.0 × 0.03 = 0.06 (6%)
        assert stop == 94.0
    
    def test_short_dte_multiplier(self):
        """
        Short DTE (<7 days) doubles the trail.
        """
        calc = StopCalculator(k_aggression=1.0, min_trail_pct=0.01)
        
        # Normal DTE (30 days)
        stop_normal = calc.compute_underlying_stop(100.0, 1.0, 0.02, 30)
        
        # Short DTE (5 days) - should be 2x wider
        stop_short = calc.compute_underlying_stop(100.0, 1.0, 0.02, 5)
        
        # Normal: 2% trail
        # Short: 4% trail (2x)
        assert stop_normal > stop_short
        assert stop_normal == 98.0  # 2%
        assert stop_short == 96.0   # 4%
    
    def test_medium_dte_multiplier(self):
        """
        Medium DTE (7-30 days) uses 1.5x multiplier.
        """
        calc = StopCalculator(k_aggression=1.0, min_trail_pct=0.01)
        
        stop = calc.compute_underlying_stop(
            entry_price=100.0,
            beta=1.0,
            index_vol_pct=0.02,  # 2%
            days_to_expiry=15
        )
        
        # 1.0 × 1.0 × 0.02 × 1.5 = 0.03 (3%)
        assert stop == 97.0
    
    def test_clamping_min(self):
        """
        Very low beta/vol should clamp to minimum trail.
        """
        calc = StopCalculator(min_trail_pct=0.04, max_trail_pct=0.40)
        
        stop = calc.compute_underlying_stop(
            entry_price=100.0,
            beta=0.1,  # Very low beta
            index_vol_pct=0.005,  # Very low vol
            days_to_expiry=60
        )
        
        # 1.0 × 0.1 × 0.005 = 0.0005 (0.05%)
        # Should clamp to min 4%
        assert stop == 96.0
    
    def test_clamping_max(self):
        """
        Extreme beta/vol should clamp to maximum trail.
        """
        calc = StopCalculator(min_trail_pct=0.04, max_trail_pct=0.40)
        
        stop = calc.compute_underlying_stop(
            entry_price=100.0,
            beta=5.0,  # Extreme beta
            index_vol_pct=0.10,  # Extreme vol (VIX ~160!)
            days_to_expiry=3     # Short DTE (2x)
        )
        
        # 1.0 × 5.0 × 0.10 × 2.0 = 1.0 (100%)
        # Should clamp to max 40%
        assert stop == 60.0
    
    def test_aggression_factor(self):
        """
        Aggression factor scales the trail.
        """
        calc_conservative = StopCalculator(k_aggression=0.7, min_trail_pct=0.01)
        calc_aggressive = StopCalculator(k_aggression=1.5, min_trail_pct=0.01)
        
        stop_cons = calc_conservative.compute_underlying_stop(100.0, 1.0, 0.05, 30)
        stop_agg = calc_aggressive.compute_underlying_stop(100.0, 1.0, 0.05, 30)
        
        # Conservative: 0.7 × 1.0 × 0.05 = 3.5%
        # Aggressive: 1.5 × 1.0 × 0.05 = 7.5%
        assert stop_cons == 96.5
        assert stop_agg == 92.5
        assert stop_cons > stop_agg


class TestBlackScholes:
    """Tests for Black-Scholes pricing."""
    
    def test_atm_call(self):
        """ATM call should be roughly half the one-sigma move."""
        price = black_scholes_call(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20
        )
        # Roughly $10 for ATM 1yr 20% vol
        assert 8 < price < 15
    
    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic."""
        price = black_scholes_call(
            S=150, K=100, T=0.1, r=0.05, sigma=0.20
        )
        # Should be close to $50 intrinsic
        assert 49 < price < 52
    
    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        price = black_scholes_call(
            S=50, K=100, T=0.1, r=0.05, sigma=0.20
        )
        assert price < 0.01
    
    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*e^(-rT)."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        
        import math
        parity_diff = call - put - S + K * math.exp(-r * T)
        
        assert abs(parity_diff) < 0.01


class TestSmartLimitPrice:
    """Tests for smart limit price computation."""
    
    def test_normal_spread(self):
        """Normal spread: limit between bid and theoretical."""
        limit = compute_smart_limit_price(
            current_bid=10.00,
            current_ask=10.50,
            theoretical_price=10.20,
            spread_participation=0.5
        )
        
        # Spread-based: 10.00 + 0.50 * 0.5 = 10.25
        # But capped at theoretical 10.20
        assert limit == 10.20
    
    def test_wide_spread(self):
        """Wide spread: limit respects bid floor."""
        limit = compute_smart_limit_price(
            current_bid=5.00,
            current_ask=6.00,
            theoretical_price=5.30,
            spread_participation=0.3
        )
        
        # Spread-based: 5.00 + 1.00 * 0.3 = 5.30
        # Theo with slippage: 5.30 * 0.97 = 5.14
        # Max of those: 5.30
        # Min with theo: 5.30
        assert limit == 5.30
    
    def test_never_below_bid(self):
        """Limit should never go below bid."""
        limit = compute_smart_limit_price(
            current_bid=10.00,
            current_ask=10.50,
            theoretical_price=9.00,  # Theo below bid
            spread_participation=0.5
        )
        
        # Floor at bid
        assert limit >= 10.00


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
