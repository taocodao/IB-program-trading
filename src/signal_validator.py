"""
Signal Validator - 3-Condition Framework
=========================================

Validates trade setups using the 3-condition framework from research:
1. Price Exhaustion: SuperTrend reversal + RSI divergence
2. Volatility Fading: ATR contracting + volume declining  
3. Sentiment Extreme: IV skew extreme

Trade signals are strongest when 3/3 conditions are met.
2/3 conditions = good setup with tighter risk management.
1/3 conditions = wait for better setup.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum
import pandas as pd
import numpy as np

# Import from our modules
try:
    from ai_signal_generator import (
        SignalResult, TrendDirection, calculate_atr
    )
    from iv_skew_analyzer import SkewResult, SkewInterpretation
except ImportError:
    # Handle relative imports
    from .ai_signal_generator import (
        SignalResult, TrendDirection, calculate_atr
    )
    from .iv_skew_analyzer import SkewResult, SkewInterpretation

logger = logging.getLogger(__name__)


# ============= Data Classes =============

class SetupQuality(Enum):
    IDEAL = "IDEAL"           # 3/3 conditions - 95% confidence
    GOOD = "GOOD"             # 2/3 conditions - 70% confidence
    NOT_IDEAL = "NOT_IDEAL"   # 1/3 conditions - 40% confidence
    NO_SETUP = "NO_SETUP"     # 0/3 conditions - do not trade


@dataclass
class ValidationResult:
    """Result from signal validation."""
    timestamp: datetime
    symbol: str
    
    # Individual conditions
    price_exhaustion_met: bool
    volatility_fading_met: bool
    sentiment_extreme_met: bool
    
    # Overall assessment
    conditions_met: int
    setup_quality: SetupQuality
    confidence: float  # 0-100
    
    # Recommendations
    should_trade: bool
    action: str
    risk_notes: List[str]
    
    # Details for debugging
    price_exhaustion_details: str
    volatility_fading_details: str
    sentiment_extreme_details: str


# ============= Signal Validator =============

class SignalValidator:
    """
    Validates trade setups using the 3-condition framework.
    
    Usage:
        validator = SignalValidator()
        
        # Get AI signal and IV skew first
        signal = ai_generator.generate_signal_from_data(df, symbol)
        skew = iv_analyzer.analyze_skew(symbol)
        
        # Validate the setup
        validation = validator.validate_setup(symbol, signal, skew, df)
        
        if validation.should_trade:
            print(f"Execute trade with {validation.confidence}% confidence")
    """
    
    def __init__(
        self,
        min_conditions_for_trade: int = 2,
        atr_contraction_threshold: float = 0.8,
        volume_decline_threshold: float = 0.7
    ):
        """
        Initialize validator with thresholds.
        
        Args:
            min_conditions_for_trade: Minimum conditions needed (default 2)
            atr_contraction_threshold: Ratio for ATR contraction (default 0.8)
            volume_decline_threshold: Ratio for volume decline (default 0.7)
        """
        self.min_conditions = min_conditions_for_trade
        self.atr_threshold = atr_contraction_threshold
        self.volume_threshold = volume_decline_threshold
    
    def validate_setup(
        self,
        symbol: str,
        signal: SignalResult,
        skew: Optional[SkewResult],
        df: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate a trade setup against 3-condition framework.
        
        Args:
            symbol: Stock symbol
            signal: Result from AISignalGenerator
            skew: Result from IVSkewAnalyzer (optional)
            df: OHLCV DataFrame for additional calculations
            
        Returns:
            ValidationResult with assessment
        """
        logger.info(f"Validating setup for {symbol}")
        
        # Check each condition
        price_exhaustion, price_details = self._check_price_exhaustion(signal)
        volatility_fading, vol_details = self._check_volatility_fading(df)
        sentiment_extreme, sent_details = self._check_sentiment_extreme(skew)
        
        # Count conditions met
        conditions_met = sum([
            price_exhaustion,
            volatility_fading,
            sentiment_extreme
        ])
        
        # Determine setup quality
        if conditions_met >= 3:
            quality = SetupQuality.IDEAL
            confidence = 95.0
        elif conditions_met >= 2:
            quality = SetupQuality.GOOD
            confidence = 70.0
        elif conditions_met >= 1:
            quality = SetupQuality.NOT_IDEAL
            confidence = 40.0
        else:
            quality = SetupQuality.NO_SETUP
            confidence = 20.0
        
        # Should we trade?
        should_trade = conditions_met >= self.min_conditions
        
        # Generate action and risk notes
        action, risk_notes = self._generate_recommendations(
            quality, signal, conditions_met
        )
        
        return ValidationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            price_exhaustion_met=price_exhaustion,
            volatility_fading_met=volatility_fading,
            sentiment_extreme_met=sentiment_extreme,
            conditions_met=conditions_met,
            setup_quality=quality,
            confidence=confidence,
            should_trade=should_trade,
            action=action,
            risk_notes=risk_notes,
            price_exhaustion_details=price_details,
            volatility_fading_details=vol_details,
            sentiment_extreme_details=sent_details
        )
    
    def _check_price_exhaustion(self, signal: SignalResult) -> tuple:
        """
        Check Condition 1: Price Exhaustion.
        
        Criteria:
        - SuperTrend shows reversal
        - RSI shows divergence
        - Or RSI at extreme with high confidence
        """
        details_parts = []
        score = 0
        
        # SuperTrend reversal check
        if signal.supertrend.trend_direction in [TrendDirection.BEARISH, TrendDirection.BULLISH]:
            if signal.supertrend.confidence >= 50:
                score += 1
                details_parts.append(f"SuperTrend {signal.supertrend.trend_direction.value}")
        
        # RSI divergence is a strong signal
        if signal.rsi.has_divergence:
            score += 2
            details_parts.append(f"RSI {signal.rsi.divergence_type} divergence")
        
        # RSI at extremes
        if signal.rsi.is_overbought or signal.rsi.is_oversold:
            score += 1
            status = "overbought" if signal.rsi.is_overbought else "oversold"
            details_parts.append(f"RSI {signal.rsi.rsi_value:.1f} ({status})")
        
        # MFI confirmation
        if signal.mfi.is_overbought or signal.mfi.is_oversold:
            score += 0.5
            status = "overbought" if signal.mfi.is_overbought else "oversold"
            details_parts.append(f"MFI {status}")
        
        # Need at least 2 points for exhaustion
        is_met = score >= 2
        details = ", ".join(details_parts) if details_parts else "No exhaustion signals"
        
        return is_met, details
    
    def _check_volatility_fading(self, df: pd.DataFrame) -> tuple:
        """
        Check Condition 2: Volatility Fading.
        
        Criteria:
        - Short-term ATR < Long-term ATR (contracting)
        - Recent volume declining vs average
        """
        if len(df) < 50:
            return False, "Insufficient data for volatility analysis"
        
        details_parts = []
        score = 0
        
        # ATR contraction check
        atr = calculate_atr(df, period=14)
        atr_short = atr.iloc[-14:].mean()
        atr_long = atr.iloc[-50:].mean()
        
        if atr_long > 0:
            atr_ratio = atr_short / atr_long
            if atr_ratio < self.atr_threshold:
                score += 1
                details_parts.append(f"ATR contracting ({atr_ratio:.2f}x long-term)")
        
        # Volume decline check
        vol = df['volume']
        vol_recent = vol.iloc[-5:].mean()
        vol_avg = vol.iloc[-20:].mean()
        
        if vol_avg > 0:
            vol_ratio = vol_recent / vol_avg
            if vol_ratio < self.volume_threshold:
                score += 1
                details_parts.append(f"Volume declining ({vol_ratio:.2f}x average)")
        
        # Need at least 1 point
        is_met = score >= 1
        details = ", ".join(details_parts) if details_parts else "Volatility not fading"
        
        return is_met, details
    
    def _check_sentiment_extreme(self, skew: Optional[SkewResult]) -> tuple:
        """
        Check Condition 3: Sentiment Extreme.
        
        Criteria:
        - IV skew at extreme (puts/calls ratio > 1.3 or < 0.8)
        """
        if skew is None:
            return False, "No IV skew data available"
        
        if skew.is_extreme:
            details = f"IV skew extreme: {skew.skew_ratio:.2f} ({skew.interpretation.value})"
            return True, details
        else:
            details = f"IV skew normal: {skew.skew_ratio:.2f}"
            return False, details
    
    def _generate_recommendations(
        self,
        quality: SetupQuality,
        signal: SignalResult,
        conditions_met: int
    ) -> tuple:
        """Generate action recommendation and risk notes."""
        risk_notes = []
        
        if quality == SetupQuality.IDEAL:
            action = f"Execute {signal.signal_type.value} with full position size"
            risk_notes.append("Ideal setup - all 3 conditions met")
            risk_notes.append("Use standard stop distance")
        elif quality == SetupQuality.GOOD:
            action = f"Execute {signal.signal_type.value} with reduced position"
            risk_notes.append("Good setup - 2/3 conditions met")
            risk_notes.append("Consider tighter stops")
        elif quality == SetupQuality.NOT_IDEAL:
            action = "Wait for better setup"
            risk_notes.append(f"Only {conditions_met}/3 conditions met")
            risk_notes.append("High risk of false signal")
        else:
            action = "No trade - conditions not met"
            risk_notes.append("Setup does not meet minimum criteria")
        
        return action, risk_notes


# ============= Convenience functions =============

_default_validator = None

def get_validator() -> SignalValidator:
    """Get or create the default signal validator."""
    global _default_validator
    if _default_validator is None:
        _default_validator = SignalValidator()
    return _default_validator


def validate_setup(
    symbol: str,
    signal: SignalResult,
    skew: Optional[SkewResult],
    df: pd.DataFrame
) -> ValidationResult:
    """Convenience function to validate using default validator."""
    return get_validator().validate_setup(symbol, signal, skew, df)


# ============= CLI Testing =============

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from ai_signal_generator import AISignalGenerator
    from iv_skew_analyzer import IVSkewAnalyzer
    
    print("=" * 60)
    print("SIGNAL VALIDATOR - Test Mode")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_bars = 300
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 0.2,
        'high': prices + abs(np.random.randn(n_bars)) * 0.5,
        'low': prices - abs(np.random.randn(n_bars)) * 0.5,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n_bars)
    })
    
    # Get signal
    sig_gen = AISignalGenerator()
    signal = sig_gen.generate_signal_from_data(df, "TEST", "5m")
    
    # Get IV skew
    iv_analyzer = IVSkewAnalyzer(simulation_mode=True)
    skew = iv_analyzer.analyze_skew("TEST", expiry_days=30, underlying_price=prices[-1])
    
    # Validate
    validator = SignalValidator()
    result = validator.validate_setup("TEST", signal, skew, df)
    
    print(f"\n{'='*40}")
    print("VALIDATION RESULT")
    print(f"{'='*40}")
    print(f"Symbol: {result.symbol}")
    print(f"Setup Quality: {result.setup_quality.value}")
    print(f"Conditions Met: {result.conditions_met}/3")
    print(f"Confidence: {result.confidence:.0f}%")
    print(f"Should Trade: {result.should_trade}")
    
    print(f"\n{'='*40}")
    print("CONDITIONS")
    print(f"{'='*40}")
    print(f"[{'✓' if result.price_exhaustion_met else '✗'}] Price Exhaustion: {result.price_exhaustion_details}")
    print(f"[{'✓' if result.volatility_fading_met else '✗'}] Volatility Fading: {result.volatility_fading_details}")
    print(f"[{'✓' if result.sentiment_extreme_met else '✗'}] Sentiment Extreme: {result.sentiment_extreme_details}")
    
    print(f"\n{'='*40}")
    print("RECOMMENDATION")
    print(f"{'='*40}")
    print(f"Action: {result.action}")
    for note in result.risk_notes:
        print(f"  • {note}")
    
    print("\n✓ Test complete")
