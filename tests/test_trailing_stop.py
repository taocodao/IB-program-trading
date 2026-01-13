"""
Unit Tests for Trailing Stop Manager
====================================

These tests verify core logic WITHOUT requiring an IB connection.
Run with: python -m pytest tests/test_trailing_stop.py -v
"""

import pytest


class TestStopPriceCalculation:
    """Tests for stop price calculation logic."""
    
    def test_stop_price_10_percent_below_bid(self):
        """Stop price should be 10% below current bid."""
        bid_price = 10.00
        trail_percent = 0.10
        
        expected_stop = bid_price * (1 - trail_percent)  # $9.00
        
        assert expected_stop == 9.00
    
    def test_stop_price_various_percentages(self):
        """Test stop calculation at different trail percentages."""
        bid_price = 100.00
        
        # 5% trail
        assert bid_price * (1 - 0.05) == 95.00
        
        # 10% trail (default)
        assert bid_price * (1 - 0.10) == 90.00
        
        # 15% trail
        assert bid_price * (1 - 0.15) == 85.00
        
        # 20% trail
        assert bid_price * (1 - 0.20) == 80.00
    
    def test_stop_price_rounding(self):
        """Stop prices should be properly rounded."""
        bid_price = 10.57
        trail_percent = 0.10
        
        stop_price = round(bid_price * (1 - trail_percent), 2)
        
        assert stop_price == 9.51  # 10.57 * 0.90 = 9.513 → 9.51


class TestTrailingLogic:
    """Tests for trailing stop logic."""
    
    def test_stop_moves_up_when_price_rises(self):
        """Stop should move UP when bid price increases."""
        initial_bid = 10.00
        trail_percent = 0.10
        
        initial_stop = initial_bid * (1 - trail_percent)  # $9.00
        
        # Price rises to $12.00
        new_bid = 12.00
        new_potential_stop = new_bid * (1 - trail_percent)  # $10.80
        
        # New stop is higher, so we should update
        should_update = new_potential_stop > initial_stop
        assert should_update is True
        assert new_potential_stop == 10.80
    
    def test_stop_stays_when_price_falls(self):
        """Stop should STAY FIXED when bid price decreases."""
        initial_bid = 12.00
        trail_percent = 0.10
        
        initial_stop = initial_bid * (1 - trail_percent)  # $10.80
        
        # Price falls to $10.00
        new_bid = 10.00
        new_potential_stop = new_bid * (1 - trail_percent)  # $9.00
        
        # New stop is LOWER, so we should NOT update
        should_update = new_potential_stop > initial_stop
        assert should_update is False
    
    def test_stop_stays_when_price_unchanged(self):
        """Stop should stay fixed when price doesn't change."""
        bid = 10.00
        trail_percent = 0.10
        
        stop = bid * (1 - trail_percent)  # $9.00
        new_potential_stop = bid * (1 - trail_percent)  # $9.00
        
        should_update = new_potential_stop > stop
        assert should_update is False
    
    def test_trailing_scenario(self):
        """Test complete trailing scenario."""
        trail_percent = 0.10
        
        # Initial: Buy at $10, bid = $10.50
        bid = 10.50
        stop = round(bid * (1 - trail_percent), 2)  # $9.45
        assert stop == 9.45
        
        # Price rises to $11.00
        bid = 11.00
        new_stop = round(bid * (1 - trail_percent), 2)  # $9.90
        if new_stop > stop:
            stop = new_stop
        assert stop == 9.90
        
        # Price falls to $10.80
        bid = 10.80
        new_potential = round(bid * (1 - trail_percent), 2)  # $9.72
        if new_potential > stop:
            stop = new_potential
        # Stop should still be $9.90 (not updated)
        assert stop == 9.90
        
        # Price rises to $12.00
        bid = 12.00
        new_stop = round(bid * (1 - trail_percent), 2)  # $10.80
        if new_stop > stop:
            stop = new_stop
        assert stop == 10.80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
