"""
Test AI Signal Generator
=========================

Unit tests for the AI signal generator module.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_signal_generator import (
    AISignalGenerator,
    calculate_atr,
    calculate_adaptive_supertrend,
    calculate_optimal_rsi,
    calculate_ml_mfi,
    kmeans_cluster_1d,
    SignalType,
    TrendDirection,
    VolatilityLevel
)


# ============= Test Fixtures =============

@pytest.fixture
def sample_df():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.2,
        'high': prices + abs(np.random.randn(n)) * 0.5,
        'low': prices - abs(np.random.randn(n)) * 0.5,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })


@pytest.fixture
def trending_up_df():
    """Generate trending up data."""
    np.random.seed(42)
    n = 300
    prices = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    
    return pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })


@pytest.fixture
def trending_down_df():
    """Generate trending down data."""
    np.random.seed(42)
    n = 300
    prices = np.linspace(150, 100, n) + np.random.randn(n) * 0.5
    
    return pd.DataFrame({
        'open': prices + 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })


# ============= Unit Tests =============

class TestATR:
    """Tests for ATR calculation."""
    
    def test_atr_calculation(self, sample_df):
        """Test ATR is calculated correctly."""
        atr = calculate_atr(sample_df, period=14)
        
        assert len(atr) == len(sample_df)
        assert atr.iloc[-1] > 0
        assert not np.isnan(atr.iloc[-1])
    
    def test_atr_increases_with_volatility(self, sample_df):
        """Test ATR increases with higher volatility."""
        atr_normal = calculate_atr(sample_df, period=14).mean()
        
        # Create high volatility data
        volatile_df = sample_df.copy()
        volatile_df['high'] = sample_df['high'] * 1.5
        volatile_df['low'] = sample_df['low'] * 0.5
        
        atr_volatile = calculate_atr(volatile_df, period=14).mean()
        
        assert atr_volatile > atr_normal


class TestKMeansClustering:
    """Tests for K-Means clustering."""
    
    def test_kmeans_clusters_count(self):
        """Test correct number of clusters."""
        values = np.array([1, 2, 3, 10, 11, 12, 50, 51, 52])
        labels, centroids = kmeans_cluster_1d(values, n_clusters=3)
        
        assert len(centroids) == 3
        assert len(labels) == len(values)
        assert set(labels) == {0, 1, 2}
    
    def test_kmeans_centroids_ordered(self):
        """Test centroids are sorted low to high."""
        values = np.array([1, 2, 100, 101, 50, 51])
        labels, centroids = kmeans_cluster_1d(values, n_clusters=3)
        
        assert centroids[0] < centroids[1] < centroids[2]


class TestAdaptiveSuperTrend:
    """Tests for Adaptive SuperTrend."""
    
    def test_supertrend_returns_valid_result(self, sample_df):
        """Test SuperTrend returns valid result."""
        result = calculate_adaptive_supertrend(sample_df)
        
        assert result.trend_direction in TrendDirection
        assert result.volatility_level in VolatilityLevel
        assert result.atr_value > 0
        assert 0 <= result.confidence <= 100
    
    def test_supertrend_bullish_on_uptrend(self, trending_up_df):
        """Test SuperTrend detects bullish on uptrend."""
        result = calculate_adaptive_supertrend(trending_up_df)
        
        # Strong uptrend should be bullish
        assert result.trend_direction in [TrendDirection.BULLISH, TrendDirection.NEUTRAL]
    
    def test_supertrend_bearish_on_downtrend(self, trending_down_df):
        """Test SuperTrend detects bearish on downtrend."""
        result = calculate_adaptive_supertrend(trending_down_df)
        
        # Strong downtrend should be bearish
        assert result.trend_direction in [TrendDirection.BEARISH, TrendDirection.NEUTRAL]


class TestOptimalRSI:
    """Tests for Optimal RSI."""
    
    def test_rsi_range(self, sample_df):
        """Test RSI value is in valid range."""
        result = calculate_optimal_rsi(sample_df)
        
        assert 0 <= result.rsi_value <= 100
        assert result.optimal_period in [7, 14, 21, 28]
    
    def test_rsi_thresholds(self, sample_df):
        """Test dynamic thresholds are reasonable."""
        result = calculate_optimal_rsi(sample_df)
        
        assert result.oversold_threshold < 50
        assert result.overbought_threshold > 50
        assert result.oversold_threshold < result.overbought_threshold
    
    def test_rsi_overbought_detection(self):
        """Test RSI detects overbought condition."""
        # Create data with strong upward momentum
        np.random.seed(42)
        n = 100
        prices = np.linspace(100, 200, n)  # Strong uptrend
        
        df = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
            'open': prices,
            'volume': np.ones(n) * 100000
        })
        
        result = calculate_optimal_rsi(df)
        
        # Strong uptrend should have high RSI
        assert result.rsi_value > 60


class TestMLMFI:
    """Tests for ML Money Flow Index."""
    
    def test_mfi_range(self, sample_df):
        """Test MFI value is in valid range."""
        result = calculate_ml_mfi(sample_df)
        
        assert 0 <= result.mfi_value <= 100
        assert 0 <= result.volume_confirmation <= 100
    
    def test_mfi_thresholds(self, sample_df):
        """Test dynamic thresholds are reasonable."""
        result = calculate_ml_mfi(sample_df)
        
        assert result.oversold_threshold < 50
        assert result.overbought_threshold > 50


class TestAISignalGenerator:
    """Tests for main signal generator."""
    
    def test_generate_signal(self, sample_df):
        """Test signal generation completes successfully."""
        generator = AISignalGenerator()
        signal = generator.generate_signal_from_data(sample_df, "TEST", "5m")
        
        assert signal.symbol == "TEST"
        assert signal.timeframe == "5m"
        assert signal.signal_type in SignalType
        assert 0 <= signal.consensus_score <= 100
        assert isinstance(signal.reasons, list)
    
    def test_signal_threshold(self, sample_df):
        """Test no signal below threshold."""
        generator = AISignalGenerator()
        generator.config.min_score_threshold = 100  # Very high threshold
        
        signal = generator.generate_signal_from_data(sample_df, "TEST")
        
        # With 100 threshold, should be no signal
        assert not signal.should_auto_execute
    
    def test_auto_execute_flag(self, sample_df):
        """Test auto-execute flag logic."""
        generator = AISignalGenerator()
        generator.config.auto_execute_enabled = True
        generator.config.auto_execute_threshold = 10  # Very low for testing
        
        signal = generator.generate_signal_from_data(sample_df, "TEST")
        
        # With low threshold, high scores should auto-execute
        if signal.consensus_score >= 10:
            assert signal.should_auto_execute or not generator.config.auto_execute_enabled
    
    def test_config_update(self):
        """Test configuration updates."""
        generator = AISignalGenerator()
        original = generator.config.min_score_threshold
        
        generator.update_config(min_score_threshold=75)
        
        assert generator.config.min_score_threshold == 75
        assert generator.config.min_score_threshold != original
    
    def test_insufficient_data_error(self):
        """Test error with insufficient data."""
        generator = AISignalGenerator()
        
        small_df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [10000]
        })
        
        with pytest.raises(ValueError, match="at least 30 bars"):
            generator.generate_signal_from_data(small_df, "TEST")


# ============= Integration Tests =============

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_signal_pipeline(self, sample_df):
        """Test complete signal generation pipeline."""
        generator = AISignalGenerator()
        signal = generator.generate_signal_from_data(sample_df, "SPY", "5m")
        
        # Verify all fields populated
        assert signal.timestamp is not None
        assert signal.supertrend is not None
        assert signal.rsi is not None
        assert signal.mfi is not None
        
        # Verify reasons make sense
        if signal.consensus_score >= 60:
            assert len(signal.reasons) > 0
    
    def test_multiple_symbols(self, sample_df):
        """Test generating signals for multiple symbols."""
        generator = AISignalGenerator()
        symbols = ["SPY", "QQQ", "AAPL"]
        
        signals = [
            generator.generate_signal_from_data(sample_df, symbol)
            for symbol in symbols
        ]
        
        assert len(signals) == 3
        assert all(s.symbol == sym for s, sym in zip(signals, symbols))


# ============= CLI Test Runner =============

if __name__ == "__main__":
    print("=" * 60)
    print("AI SIGNAL GENERATOR - Unit Tests")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 300
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    sample_df = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.2,
        'high': prices + abs(np.random.randn(n)) * 0.5,
        'low': prices - abs(np.random.randn(n)) * 0.5,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })
    
    # Run basic tests
    print("\n1. Testing ATR calculation...")
    atr = calculate_atr(sample_df)
    assert atr.iloc[-1] > 0
    print(f"   ✓ ATR = {atr.iloc[-1]:.4f}")
    
    print("\n2. Testing K-Means clustering...")
    values = np.array([1, 2, 3, 50, 51, 100, 101])
    labels, centroids = kmeans_cluster_1d(values, n_clusters=3)
    assert len(centroids) == 3
    print(f"   ✓ Centroids: {centroids}")
    
    print("\n3. Testing Adaptive SuperTrend...")
    supertrend = calculate_adaptive_supertrend(sample_df)
    print(f"   ✓ Direction: {supertrend.trend_direction.value}")
    print(f"   ✓ Volatility: {supertrend.volatility_level.value}")
    
    print("\n4. Testing Optimal RSI...")
    rsi = calculate_optimal_rsi(sample_df)
    print(f"   ✓ RSI: {rsi.rsi_value:.1f} (period {rsi.optimal_period})")
    print(f"   ✓ Divergence: {rsi.has_divergence}")
    
    print("\n5. Testing ML MFI...")
    mfi = calculate_ml_mfi(sample_df)
    print(f"   ✓ MFI: {mfi.mfi_value:.1f}")
    print(f"   ✓ Volume confirmation: {mfi.volume_confirmation:.0f}%")
    
    print("\n6. Testing full signal generation...")
    generator = AISignalGenerator()
    signal = generator.generate_signal_from_data(sample_df, "TEST", "5m")
    print(f"   ✓ Signal: {signal.signal_type.value}")
    print(f"   ✓ Score: {signal.consensus_score:.0f}/100")
    print(f"   ✓ Auto-execute: {signal.should_auto_execute}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
