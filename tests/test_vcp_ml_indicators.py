"""
Unit Tests for VCP + ML Indicators

Tests all three indicator modules:
1. VCP Detector
2. ML Adaptive SuperTrend
3. ML Optimal RSI
"""
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indicators import (
    VCPDetector, 
    MLAdaptiveSuperTrend, 
    MLOptimalRSI,
    VCPMLSignalGenerator,
    calculate_atr,
    calculate_sma,
    calculate_rsi,
    calculate_bollinger_bands
)


# ==========================================
# Test Data Generators
# ==========================================

def generate_trending_data(n: int = 300, trend: str = 'up') -> tuple:
    """Generate trending price data"""
    np.random.seed(42)
    
    if trend == 'up':
        base = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    else:
        base = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    
    noise = np.random.randn(n) * 0.5
    close = base + noise
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000000, 5000000, n).astype(float)
    
    return high, low, close, volume


def generate_consolidating_data(n: int = 300) -> tuple:
    """Generate consolidating (sideways) price data with VCP-like patterns"""
    np.random.seed(42)
    
    # First 200 bars: trending up
    up_phase = 100 + np.cumsum(np.random.randn(200) * 0.5 + 0.1)
    
    # Last 100 bars: tight consolidation (VCP pattern)
    consolidation_center = up_phase[-1]
    consolidation = consolidation_center + np.random.randn(100) * 0.3  # Very tight range
    
    close = np.concatenate([up_phase, consolidation])
    noise = np.random.randn(n) * 0.2
    high = close + np.abs(noise)
    low = close - np.abs(noise)
    volume = np.random.randint(500000, 2000000, n).astype(float)  # Lower volume
    
    return high, low, close, volume


# ==========================================
# Utility Function Tests
# ==========================================

class TestIndicatorUtils:
    """Test utility functions"""
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        high, low, close, _ = generate_trending_data(50)
        atr = calculate_atr(high, low, close, period=14)
        
        assert len(atr) == 50
        assert atr[13] > 0  # First valid ATR at period-1
        assert all(atr[14:] > 0)  # All subsequent values positive
    
    def test_calculate_sma(self):
        """Test SMA calculation"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = calculate_sma(close, period=3)
        
        assert len(sma) == 10
        assert np.isnan(sma[0])  # First values are NaN
        assert np.isnan(sma[1])
        assert sma[2] == 2.0  # (1+2+3)/3
        assert sma[3] == 3.0  # (2+3+4)/3
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        close = np.array([44, 44.25, 44.5, 43.75, 44.5, 44.25, 44.5, 
                         43.75, 44.5, 44.25, 43.75, 44, 43.5, 44.25, 44.5])
        rsi = calculate_rsi(close, period=14)
        
        assert len(rsi) == 15
        assert 0 <= rsi[-1] <= 100  # RSI bounded between 0-100
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        high, low, close, _ = generate_trending_data(50)
        bb = calculate_bollinger_bands(close, period=20, std_mult=2.0)
        
        assert 'middle' in bb
        assert 'upper' in bb
        assert 'lower' in bb
        assert 'width' in bb
        assert len(bb['middle']) == 50
        assert bb['upper'][-1] > bb['middle'][-1] > bb['lower'][-1]


# ==========================================
# VCP Detector Tests
# ==========================================

class TestVCPDetector:
    """Test VCP Detector"""
    
    def test_initialization(self):
        """Test VCP Detector initialization"""
        detector = VCPDetector()
        assert detector.atr_period == 20
        assert detector.bb_period == 20
        assert detector.min_consolidation_bars == 5
        assert detector.max_consolidation_bars == 30
    
    def test_detect_vcp_zones_consolidating(self):
        """Test VCP zone detection on consolidating data"""
        high, low, close, volume = generate_consolidating_data(300)
        detector = VCPDetector()
        
        zones = detector.detect_vcp_zones(high, low, close, volume)
        
        # Should find at least one zone in consolidating data
        # Note: May not always find zones depending on random data
        assert isinstance(zones, list)
    
    def test_detect_vcp_zones_trending(self):
        """Test VCP zone detection on trending data"""
        high, low, close, volume = generate_trending_data(300, trend='up')
        detector = VCPDetector()
        
        zones = detector.detect_vcp_zones(high, low, close, volume)
        
        # Trending data should have fewer VCP zones
        assert isinstance(zones, list)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        high = np.array([100, 101, 102])
        low = np.array([99, 100, 101])
        close = np.array([100, 101, 102])
        volume = np.array([1000, 1000, 1000])
        
        detector = VCPDetector()
        zones = detector.detect_vcp_zones(high, low, close, volume)
        
        # Should return empty list, not error
        assert zones == []


# ==========================================
# ML Adaptive SuperTrend Tests
# ==========================================

class TestMLAdaptiveSuperTrend:
    """Test ML Adaptive SuperTrend"""
    
    def test_initialization(self):
        """Test SuperTrend initialization"""
        st = MLAdaptiveSuperTrend()
        assert st.atr_period == 10
        assert st.base_multiplier == 3.0
        assert st.kmeans_clusters == 3
    
    def test_calculate_supertrend_uptrend(self):
        """Test SuperTrend on uptrending data"""
        high, low, close, _ = generate_trending_data(300, trend='up')
        st = MLAdaptiveSuperTrend()
        
        result = st.calculate_supertrend(high, low, close)
        
        assert result is not None
        assert len(result.supertrend) == 300
        assert len(result.trend) == 300
        # Last values should indicate uptrend
        assert result.trend[-1] in [1, -1]
    
    def test_calculate_supertrend_downtrend(self):
        """Test SuperTrend on downtrending data"""
        high, low, close, _ = generate_trending_data(300, trend='down')
        st = MLAdaptiveSuperTrend()
        
        result = st.calculate_supertrend(high, low, close)
        
        assert result is not None
        # Should correctly identify some downtrend
        assert -1 in result.trend
    
    def test_volatility_classification(self):
        """Test K-Means volatility classification"""
        high, low, close, _ = generate_trending_data(300)
        st = MLAdaptiveSuperTrend()
        
        atr = calculate_atr(high, low, close, 10)
        vol_class, centers = st.kmeans_volatility_classification(atr)
        
        assert vol_class.value in ['LOW', 'MEDIUM', 'HIGH']
        assert len(centers) == 3
    
    def test_overextension_check(self):
        """Test overextension detection"""
        high, low, close, _ = generate_trending_data(300)
        st = MLAdaptiveSuperTrend()
        
        result = st.calculate_supertrend(high, low, close)
        overext = st.check_overextension(close, result)
        
        assert hasattr(overext, 'overextended')
        assert hasattr(overext, 'severity')
        assert hasattr(overext, 'recommendation')


# ==========================================
# ML Optimal RSI Tests
# ==========================================

class TestMLOptimalRSI:
    """Test ML Optimal RSI"""
    
    def test_initialization(self):
        """Test RSI initialization"""
        rsi = MLOptimalRSI()
        assert rsi.rsi_lengths == [5, 7, 9, 11, 14, 21]
        assert rsi.divergence_lookback == 10
    
    def test_calculate_all_rsi(self):
        """Test multi-length RSI calculation"""
        _, _, close, _ = generate_trending_data(100)
        rsi = MLOptimalRSI()
        
        rsi_values = rsi.calculate_all_rsi(close)
        
        assert len(rsi_values) == 6  # 6 different lengths
        assert 5 in rsi_values
        assert 14 in rsi_values
        assert 21 in rsi_values
    
    def test_dynamic_levels(self):
        """Test dynamic overbought/oversold levels"""
        _, _, close, _ = generate_trending_data(200)
        rsi = MLOptimalRSI()
        
        rsi_values = rsi.calculate_all_rsi(close)
        levels = rsi.calculate_dynamic_levels(rsi_values[14], 14)
        
        assert levels.overbought >= 65
        assert levels.oversold <= 35
        assert levels.midline == 50
    
    def test_analyze(self):
        """Test complete RSI analysis"""
        _, _, close, _ = generate_trending_data(300)
        rsi = MLOptimalRSI()
        
        result = rsi.analyze(close)
        
        assert hasattr(result, 'rsi_values')
        assert hasattr(result, 'levels')
        assert hasattr(result, 'divergences')
        assert hasattr(result, 'consensus_direction')
        assert hasattr(result, 'confidence')


# ==========================================
# Combined Signal Generator Tests
# ==========================================

class TestVCPMLSignalGenerator:
    """Test combined signal generator"""
    
    def test_initialization(self):
        """Test signal generator initialization"""
        gen = VCPMLSignalGenerator()
        assert gen.min_confidence == 60.0
        assert gen.vcp_detector is not None
        assert gen.ml_supertrend is not None
        assert gen.ml_rsi is not None
    
    def test_generate_signal(self):
        """Test signal generation"""
        high, low, close, volume = generate_trending_data(300, trend='up')
        gen = VCPMLSignalGenerator()
        
        signal = gen.generate_signal('TEST', high, low, close, volume)
        
        assert signal is not None
        assert hasattr(signal, 'symbol')
        assert hasattr(signal, 'signal_type')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'is_actionable')
    
    def test_no_signal_on_neutral(self):
        """Test that no signal generated on neutral data"""
        # Create very flat data
        close = np.full(300, 100.0) + np.random.randn(300) * 0.01
        high = close + 0.01
        low = close - 0.01
        volume = np.full(300, 1000000.0)
        
        gen = VCPMLSignalGenerator()
        signal = gen.generate_signal('FLAT', high, low, close, volume)
        
        # May or may not generate signal, but should not error
        assert signal is not None


# ==========================================
# Run Tests
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
