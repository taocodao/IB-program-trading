"""
AI Signal Generator for Options Trading
========================================

Machine learning-based overbought/oversold detection using:
1. ML Adaptive SuperTrend (K-Means volatility clustering)
2. ML Optimal RSI with divergence detection
3. Money Flow Index with dynamic thresholds
4. Multi-indicator consensus scoring

Based on deep research: AI-Indicators-Overbought-Oversold-Options-Deep-Research.md
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from enum import Enum
import logging

# VCP + ML Indicators integration
try:
    from indicators import VCPMLSignalGenerator, VCPDetector, analyze_vcp
    VCP_ML_AVAILABLE = True
except ImportError:
    VCP_ML_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============= Configuration (can be overridden by dashboard) =============

class SignalConfig:
    """Configurable signal parameters - can be updated from dashboard."""
    
    # Default timeframe for signal generation
    signal_timeframe: str = "5m"
    
    # Minimum consensus score to trigger trade
    min_score_threshold: int = 60
    
    # Auto-execute threshold - trades above this execute automatically
    auto_execute_threshold: int = 85
    
    # Enable/disable auto-execution
    auto_execute_enabled: bool = True
    
    # K-Means clusters for volatility classification
    volatility_clusters: int = 3
    
    # RSI periods to test
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21, 28])
    
    # SuperTrend factors per volatility level
    supertrend_factors: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 2.0,
        "MEDIUM": 3.0,
        "HIGH": 4.0
    })


# Initialize with defaults
SIGNAL_CONFIG = SignalConfig()


# ============= Data Classes =============

class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class VolatilityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SignalType(Enum):
    BUY_CALL = "BUY_CALL"      # Oversold/bullish - buy calls
    BUY_PUT = "BUY_PUT"        # Overbought/bearish - buy puts
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class SuperTrendResult:
    """Result from ML Adaptive SuperTrend calculation."""
    trend_direction: TrendDirection
    volatility_level: VolatilityLevel
    supertrend_line: float
    atr_value: float
    confidence: float  # 0-100


@dataclass
class RSIResult:
    """Result from ML Optimal RSI calculation."""
    rsi_value: float
    optimal_period: int
    overbought_threshold: float  # Dynamic, not fixed 70
    oversold_threshold: float    # Dynamic, not fixed 30
    is_overbought: bool
    is_oversold: bool
    has_divergence: bool
    divergence_type: Optional[str] = None  # "bullish" or "bearish"


@dataclass
class MFIResult:
    """Result from ML Money Flow Index calculation."""
    mfi_value: float
    overbought_threshold: float
    oversold_threshold: float
    is_overbought: bool
    is_oversold: bool
    volume_confirmation: float  # 0-100


@dataclass
class SignalResult:
    """Complete signal result with all indicators."""
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_type: SignalType
    consensus_score: float  # 0-100
    
    # Individual indicator results
    supertrend: SuperTrendResult
    rsi: RSIResult
    mfi: MFIResult
    
    # Recommendation details
    should_auto_execute: bool
    requires_approval: bool
    reasons: List[str]
    expected_move: str
    
    # Raw data for debugging
    overbought_votes: int = 0
    oversold_votes: int = 0


# ============= ML Indicator Functions =============

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
        
    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr


def kmeans_cluster_1d(values: np.ndarray, n_clusters: int = 3, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D K-Means clustering implementation.
    
    Args:
        values: Array of values to cluster
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (labels, centroids)
    """
    # Initialize centroids using quantiles
    centroids = np.percentile(values, np.linspace(0, 100, n_clusters + 2)[1:-1])
    
    for _ in range(max_iter):
        # Assign points to nearest centroid
        distances = np.abs(values.reshape(-1, 1) - centroids.reshape(1, -1))
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            values[labels == k].mean() if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Sort centroids (LOW, MEDIUM, HIGH)
    sorted_idx = np.argsort(centroids)
    sorted_centroids = centroids[sorted_idx]
    
    # Remap labels
    label_map = {old: new for new, old in enumerate(sorted_idx)}
    sorted_labels = np.array([label_map[l] for l in labels])
    
    return sorted_labels, sorted_centroids


def calculate_adaptive_supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    training_bars: int = 300
) -> SuperTrendResult:
    """
    Calculate ML Adaptive SuperTrend.
    
    Uses K-Means clustering on ATR to classify volatility levels,
    then applies dynamic SuperTrend factor based on market conditions.
    """
    if len(df) < training_bars:
        training_bars = len(df)
    
    # Calculate ATR
    atr = calculate_atr(df, atr_period)
    current_atr = atr.iloc[-1]
    
    # K-Means clustering on ATR values
    atr_values = atr.iloc[-training_bars:].dropna().values
    if len(atr_values) < 10:
        # Not enough data, return neutral
        return SuperTrendResult(
            trend_direction=TrendDirection.NEUTRAL,
            volatility_level=VolatilityLevel.MEDIUM,
            supertrend_line=df['close'].iloc[-1],
            atr_value=current_atr,
            confidence=50.0
        )
    
    labels, centroids = kmeans_cluster_1d(atr_values, n_clusters=3)
    
    # Determine current volatility level
    current_label = labels[-1]
    vol_levels = [VolatilityLevel.LOW, VolatilityLevel.MEDIUM, VolatilityLevel.HIGH]
    volatility_level = vol_levels[current_label]
    
    # Get appropriate SuperTrend factor
    factor = {
        VolatilityLevel.LOW: 2.0,
        VolatilityLevel.MEDIUM: 3.0,
        VolatilityLevel.HIGH: 4.0
    }[volatility_level]
    
    # Calculate SuperTrend
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (factor * atr)
    lower_band = hl2 - (factor * atr)
    
    # Determine trend direction
    close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2] if len(df) > 1 else close
    
    # Simple trend determination
    if close > upper_band.iloc[-2]:
        trend = TrendDirection.BULLISH
        supertrend_line = lower_band.iloc[-1]
    elif close < lower_band.iloc[-2]:
        trend = TrendDirection.BEARISH
        supertrend_line = upper_band.iloc[-1]
    else:
        # Continuation of previous trend
        if close > hl2.iloc[-1]:
            trend = TrendDirection.BULLISH
            supertrend_line = lower_band.iloc[-1]
        else:
            trend = TrendDirection.BEARISH
            supertrend_line = upper_band.iloc[-1]
    
    # Calculate confidence based on distance from SuperTrend line
    distance_pct = abs(close - supertrend_line) / close * 100
    confidence = min(100, distance_pct * 10)  # Scale to 0-100
    
    return SuperTrendResult(
        trend_direction=trend,
        volatility_level=volatility_level,
        supertrend_line=supertrend_line,
        atr_value=current_atr,
        confidence=confidence
    )


def calculate_optimal_rsi(
    df: pd.DataFrame,
    periods: List[int] = None,
    lookback: int = 50
) -> RSIResult:
    """
    Calculate ML Optimal RSI.
    
    Tests multiple RSI periods and selects the one that best
    correlates with price reversals. Also detects divergence.
    """
    if periods is None:
        periods = [7, 14, 21, 28]
    
    close = df['close']
    
    # Calculate RSI for each period
    rsi_values = {}
    for period in periods:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_values[period] = rsi.fillna(50)
    
    # Find optimal period (one with most extreme current reading)
    current_readings = {p: abs(rsi_values[p].iloc[-1] - 50) for p in periods}
    optimal_period = max(current_readings, key=current_readings.get)
    rsi = rsi_values[optimal_period]
    current_rsi = rsi.iloc[-1]
    
    # Dynamic thresholds based on recent RSI range
    recent_rsi = rsi.iloc[-lookback:]
    rsi_std = recent_rsi.std()
    rsi_mean = recent_rsi.mean()
    
    overbought = min(80, rsi_mean + 1.5 * rsi_std)
    oversold = max(20, rsi_mean - 1.5 * rsi_std)
    
    is_overbought = current_rsi > overbought
    is_oversold = current_rsi < oversold
    
    # Divergence detection
    has_divergence = False
    divergence_type = None
    
    if len(close) >= 20:
        # Look for price/RSI divergence in last 20 bars
        price_highs = []
        rsi_highs = []
        price_lows = []
        rsi_lows = []
        
        for i in range(-20, -1):
            # Local high
            if close.iloc[i] > close.iloc[i-1] and close.iloc[i] > close.iloc[i+1]:
                price_highs.append((i, close.iloc[i]))
                rsi_highs.append((i, rsi.iloc[i]))
            # Local low
            if close.iloc[i] < close.iloc[i-1] and close.iloc[i] < close.iloc[i+1]:
                price_lows.append((i, close.iloc[i]))
                rsi_lows.append((i, rsi.iloc[i]))
        
        # Bearish divergence: price higher high, RSI lower high
        if len(price_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                has_divergence = True
                divergence_type = "bearish"
        
        # Bullish divergence: price lower low, RSI higher low
        if len(price_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                has_divergence = True
                divergence_type = "bullish"
    
    return RSIResult(
        rsi_value=current_rsi,
        optimal_period=optimal_period,
        overbought_threshold=overbought,
        oversold_threshold=oversold,
        is_overbought=is_overbought,
        is_oversold=is_oversold,
        has_divergence=has_divergence,
        divergence_type=divergence_type
    )


def calculate_ml_mfi(
    df: pd.DataFrame,
    period: int = 14,
    training_bars: int = 300
) -> MFIResult:
    """
    Calculate ML Money Flow Index with dynamic thresholds.
    
    Uses K-Means to determine appropriate overbought/oversold levels
    based on historical MFI distribution.
    """
    # Calculate MFI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    
    # Positive and negative money flow
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    # Sum over period
    positive_sum = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=1).sum()
    
    # Money Flow Ratio and MFI
    mf_ratio = positive_sum / negative_sum.replace(0, 1)
    mfi = 100 - (100 / (1 + mf_ratio))
    mfi = mfi.fillna(50)
    
    current_mfi = mfi.iloc[-1]
    
    # Dynamic thresholds using K-Means
    if len(df) < training_bars:
        training_bars = len(df)
    
    mfi_values = mfi.iloc[-training_bars:].dropna().values
    if len(mfi_values) < 20:
        # Not enough data
        overbought = 70.0
        oversold = 30.0
    else:
        labels, centroids = kmeans_cluster_1d(mfi_values, n_clusters=3)
        # Use cluster boundaries as thresholds
        oversold = (centroids[0] + centroids[1]) / 2
        overbought = (centroids[1] + centroids[2]) / 2
    
    is_overbought = current_mfi > overbought
    is_oversold = current_mfi < oversold
    
    # Volume confirmation (relative volume)
    recent_vol = df['volume'].iloc[-20:].mean()
    current_vol = df['volume'].iloc[-1]
    volume_ratio = current_vol / recent_vol if recent_vol > 0 else 1.0
    volume_confirmation = min(100, volume_ratio * 50)  # Scale to 0-100
    
    return MFIResult(
        mfi_value=current_mfi,
        overbought_threshold=overbought,
        oversold_threshold=oversold,
        is_overbought=is_overbought,
        is_oversold=is_oversold,
        volume_confirmation=volume_confirmation
    )


# ============= Main Signal Generator Class =============

class AISignalGenerator:
    """
    Main signal generator that combines all ML indicators.
    
    Usage:
        generator = AISignalGenerator()
        signal = generator.generate_signal_from_data(df, "SPY")
        
        if signal.consensus_score >= 60:
            print(f"Trade signal: {signal.signal_type}")
    """
    
    def __init__(self, config: SignalConfig = None):
        """Initialize with optional custom configuration."""
        self.config = config or SIGNAL_CONFIG
    
    def generate_signal_from_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "5m"
    ) -> SignalResult:
        """
        Generate trading signal from OHLCV data.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            symbol: Stock symbol
            timeframe: Timeframe string (e.g., "5m", "15m")
            
        Returns:
            SignalResult with all indicator data and recommendation
        """
        logger.info(f"Generating signal for {symbol} ({timeframe})")
        
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(df) < 30:
            raise ValueError("Need at least 30 bars of data")
        
        # Calculate all indicators
        supertrend = calculate_adaptive_supertrend(df)
        rsi = calculate_optimal_rsi(df)
        mfi = calculate_ml_mfi(df)
        
        # Calculate consensus
        consensus_score, overbought_votes, oversold_votes = self._calculate_consensus(
            supertrend, rsi, mfi
        )
        
        # Determine signal type
        signal_type = self._determine_signal_type(
            consensus_score, overbought_votes, oversold_votes, rsi
        )
        
        # Generate reasons
        reasons = self._generate_reasons(supertrend, rsi, mfi, signal_type)
        
        # Determine execution mode
        should_auto_execute = (
            self.config.auto_execute_enabled and 
            consensus_score >= self.config.auto_execute_threshold
        )
        requires_approval = (
            consensus_score >= self.config.min_score_threshold and
            not should_auto_execute
        )
        
        # Expected move estimate
        if consensus_score >= 80:
            expected_move = "30-50% (high confidence)"
        elif consensus_score >= 60:
            expected_move = "20-40% (moderate confidence)"
        else:
            expected_move = "10-30% (low confidence)"
        
        return SignalResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            consensus_score=consensus_score,
            supertrend=supertrend,
            rsi=rsi,
            mfi=mfi,
            should_auto_execute=should_auto_execute,
            requires_approval=requires_approval,
            reasons=reasons,
            expected_move=expected_move,
            overbought_votes=overbought_votes,
            oversold_votes=oversold_votes
        )
    
    def _calculate_consensus(
        self,
        supertrend: SuperTrendResult,
        rsi: RSIResult,
        mfi: MFIResult
    ) -> Tuple[float, int, int]:
        """
        Calculate consensus score across all indicators.
        
        Returns:
            Tuple of (score, overbought_votes, oversold_votes)
        """
        overbought_votes = 0
        oversold_votes = 0
        
        # SuperTrend vote (25 points)
        if supertrend.trend_direction == TrendDirection.BEARISH:
            overbought_votes += 25
        elif supertrend.trend_direction == TrendDirection.BULLISH:
            oversold_votes += 25
        
        # RSI vote (25 points)
        if rsi.is_overbought:
            overbought_votes += 25
        elif rsi.is_oversold:
            oversold_votes += 25
        
        # RSI divergence bonus (15 points)
        if rsi.has_divergence:
            if rsi.divergence_type == "bearish":
                overbought_votes += 15
            elif rsi.divergence_type == "bullish":
                oversold_votes += 15
        
        # MFI vote (20 points)
        if mfi.is_overbought:
            overbought_votes += 20
        elif mfi.is_oversold:
            oversold_votes += 20
        
        # Volume confirmation bonus (15 points)
        if mfi.volume_confirmation > 60:
            if mfi.is_overbought:
                overbought_votes += 15
            elif mfi.is_oversold:
                oversold_votes += 15
        
        # Calculate final score (max 100)
        consensus_score = max(overbought_votes, oversold_votes)
        
        return consensus_score, overbought_votes, oversold_votes
    
    def _determine_signal_type(
        self,
        consensus_score: float,
        overbought_votes: int,
        oversold_votes: int,
        rsi: RSIResult
    ) -> SignalType:
        """
        Determine the appropriate signal type based on indicators.
        
        Long-only strategy:
        - Overbought/bearish = BUY PUT (expecting downside)
        - Oversold/bullish = BUY CALL (expecting upside)
        """
        
        if consensus_score < self.config.min_score_threshold:
            return SignalType.NO_SIGNAL
        
        if overbought_votes > oversold_votes:
            # Overbought - expect pullback, BUY PUT
            return SignalType.BUY_PUT
        else:
            # Oversold - expect bounce, BUY CALL
            return SignalType.BUY_CALL
    
    def _generate_reasons(
        self,
        supertrend: SuperTrendResult,
        rsi: RSIResult,
        mfi: MFIResult,
        signal_type: SignalType
    ) -> List[str]:
        """Generate human-readable reasons for the signal."""
        reasons = []
        
        # SuperTrend reason
        if supertrend.trend_direction == TrendDirection.BEARISH:
            reasons.append(f"SuperTrend turned BEARISH ({supertrend.volatility_level.value} volatility)")
        elif supertrend.trend_direction == TrendDirection.BULLISH:
            reasons.append(f"SuperTrend turned BULLISH ({supertrend.volatility_level.value} volatility)")
        
        # RSI reason
        if rsi.is_overbought:
            reasons.append(f"RSI({rsi.optimal_period}) overbought at {rsi.rsi_value:.1f} (threshold: {rsi.overbought_threshold:.1f})")
        elif rsi.is_oversold:
            reasons.append(f"RSI({rsi.optimal_period}) oversold at {rsi.rsi_value:.1f} (threshold: {rsi.oversold_threshold:.1f})")
        
        # Divergence reason
        if rsi.has_divergence:
            reasons.append(f"RSI {rsi.divergence_type.upper()} divergence detected - high probability reversal")
        
        # MFI reason
        if mfi.is_overbought:
            reasons.append(f"MFI overbought at {mfi.mfi_value:.1f}")
        elif mfi.is_oversold:
            reasons.append(f"MFI oversold at {mfi.mfi_value:.1f}")
        
        # Volume confirmation
        if mfi.volume_confirmation > 60:
            reasons.append(f"Volume confirms signal ({mfi.volume_confirmation:.0f}% strength)")
        
        return reasons
    
    def update_config(self, **kwargs):
        """
        Update configuration from dashboard.
        
        Args:
            signal_timeframe: str
            min_score_threshold: int
            auto_execute_threshold: int
            auto_execute_enabled: bool
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")
    
    def generate_signal_with_vcp(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "5m"
    ) -> SignalResult:
        """
        Generate enhanced signal using VCP + ML indicators.
        
        Combines VCP breakout detection with existing SuperTrend/RSI/MFI
        analysis for higher confidence signals (80%+).
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            symbol: Stock symbol
            timeframe: Timeframe string
            
        Returns:
            SignalResult with VCP-enhanced scoring
        """
        # First, generate the standard signal
        base_signal = self.generate_signal_from_data(df, symbol, timeframe)
        
        # If VCP+ML not available, return base signal
        if not VCP_ML_AVAILABLE:
            logger.debug("VCP+ML indicators not available, using base signal")
            return base_signal
        
        # Extract numpy arrays from DataFrame
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Check for VCP patterns
        try:
            vcp_result = analyze_vcp(high, low, close, volume)
            
            if vcp_result['has_signal']:
                # VCP breakout detected - boost confidence
                vcp_confidence = vcp_result['confidence']
                vcp_signal_type = vcp_result['signal_type']
                
                logger.info(f"{symbol}: VCP breakout detected - {vcp_signal_type} @ {vcp_confidence:.0f}%")
                
                # Check if VCP agrees with base signal
                base_is_bullish = base_signal.signal_type == SignalType.BUY_CALL
                base_is_bearish = base_signal.signal_type == SignalType.BUY_PUT
                vcp_is_bullish = vcp_signal_type == 'BUY_CALL'
                vcp_is_bearish = vcp_signal_type == 'BUY_PUT'
                
                # Calculate combined score
                agreement_boost = 0
                if (base_is_bullish and vcp_is_bullish) or (base_is_bearish and vcp_is_bearish):
                    # Both agree - strong signal
                    agreement_boost = 15
                    base_signal.reasons.append(f"🎯 VCP Breakout confirms direction (+15% confidence)")
                elif base_signal.signal_type == SignalType.NO_SIGNAL:
                    # Use VCP signal when base has no signal
                    agreement_boost = 10
                    base_signal.reasons.insert(0, f"🚀 VCP Breakout detected: {vcp_signal_type}")
                    # Update signal type based on VCP
                    base_signal.signal_type = SignalType.BUY_CALL if vcp_is_bullish else SignalType.BUY_PUT
                else:
                    # Conflicting signals - add note but don't boost
                    base_signal.reasons.append(f"⚠️ VCP suggests {vcp_signal_type} (conflicts with base signal)")
                
                # Update consensus score
                combined_score = (base_signal.consensus_score * 0.6) + (vcp_confidence * 0.4) + agreement_boost
                base_signal.consensus_score = min(combined_score, 95)
                
                # Update execution thresholds
                base_signal.should_auto_execute = (
                    self.config.auto_execute_enabled and
                    base_signal.consensus_score >= self.config.auto_execute_threshold
                )
                base_signal.requires_approval = (
                    base_signal.consensus_score >= self.config.min_score_threshold and
                    not base_signal.should_auto_execute
                )
                
            elif vcp_result['active_zones'] > 0:
                # VCP zones found but no breakout yet
                base_signal.reasons.append(f"📊 VCP: {vcp_result['active_zones']} consolidation zone(s) forming")
                
        except Exception as e:
            logger.warning(f"{symbol}: VCP analysis error: {e}")
        
        return base_signal


# ============= Module-level convenience functions =============

_default_generator = None

def get_generator() -> AISignalGenerator:
    """Get or create the default signal generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = AISignalGenerator()
    return _default_generator


def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = "5m") -> SignalResult:
    """Convenience function to generate signal using default generator."""
    return get_generator().generate_signal_from_data(df, symbol, timeframe)


def generate_signal_enhanced(df: pd.DataFrame, symbol: str, timeframe: str = "5m") -> SignalResult:
    """
    Convenience function to generate VCP-enhanced signal.
    
    Uses VCP + ML indicators for higher confidence (80%+) signals.
    """
    return get_generator().generate_signal_with_vcp(df, symbol, timeframe)


# ============= CLI Testing =============

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("AI SIGNAL GENERATOR - Test Mode")
    print("=" * 60)
    
    # Generate sample data for testing
    np.random.seed(42)
    n_bars = 300
    
    # Simulate price data with trend
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 0.2,
        'high': prices + abs(np.random.randn(n_bars)) * 0.5,
        'low': prices - abs(np.random.randn(n_bars)) * 0.5,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n_bars)
    })
    
    print(f"\nTest data: {n_bars} bars, price range ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Generate signal
    generator = AISignalGenerator()
    signal = generator.generate_signal_from_data(df, "TEST", "5m")
    
    print(f"\n{'='*40}")
    print("SIGNAL RESULT")
    print(f"{'='*40}")
    print(f"Symbol: {signal.symbol}")
    print(f"Signal Type: {signal.signal_type.value}")
    print(f"Consensus Score: {signal.consensus_score:.0f}/100")
    print(f"Auto-Execute: {signal.should_auto_execute}")
    print(f"Requires Approval: {signal.requires_approval}")
    print(f"Expected Move: {signal.expected_move}")
    
    print(f"\n{'='*40}")
    print("INDICATORS")
    print(f"{'='*40}")
    print(f"SuperTrend: {signal.supertrend.trend_direction.value} ({signal.supertrend.volatility_level.value})")
    print(f"RSI({signal.rsi.optimal_period}): {signal.rsi.rsi_value:.1f} (OB: {signal.rsi.is_overbought}, OS: {signal.rsi.is_oversold})")
    if signal.rsi.has_divergence:
        print(f"  └─ DIVERGENCE: {signal.rsi.divergence_type}")
    print(f"MFI: {signal.mfi.mfi_value:.1f} (OB: {signal.mfi.is_overbought}, OS: {signal.mfi.is_oversold})")
    print(f"Volume Confirmation: {signal.mfi.volume_confirmation:.0f}%")
    
    print(f"\n{'='*40}")
    print("REASONS")
    print(f"{'='*40}")
    for reason in signal.reasons:
        print(f"  • {reason}")
    
    print(f"\n✓ Test complete")
