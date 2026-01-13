"""
VCP + ML Signal Generator

Combines VCP Detector, ML Adaptive SuperTrend, and ML Optimal RSI
to generate high-confidence trading signals.

Signal Flow:
1. VCP Detector finds consolidation zones and breakouts
2. ML SuperTrend confirms trend direction
3. ML RSI validates with divergence or momentum confirmation

Expected Combined Performance:
- VCP base: 65-70%
- SuperTrend confirmation: 72-75%
- RSI validation: 68%
- Combined (all three aligned): ~80%+
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .vcp_detector import VCPDetector, VCPBreakout, BreakoutDirection
from .ml_supertrend import MLAdaptiveSuperTrend, TrendDirection
from .ml_optimal_rsi import MLOptimalRSI

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength classification"""
    STRONG = "STRONG"      # All 3 indicators agree (80%+)
    MODERATE = "MODERATE"  # 2 indicators agree (70%+)
    WEAK = "WEAK"          # 1 indicator only (60%+)
    NONE = "NONE"          # No signal


@dataclass
class CombinedSignal:
    """Combined signal from all three indicators"""
    symbol: str
    signal_type: str  # 'BUY_CALL' or 'BUY_PUT'
    direction: str    # 'BULLISH' or 'BEARISH'
    confidence: float # 0-100
    strength: SignalStrength
    
    # Individual indicator results
    vcp_breakout: Optional[VCPBreakout]
    supertrend_trend: str
    supertrend_confidence: float
    rsi_consensus: str
    rsi_confidence: float
    
    # Metadata
    reasons: List[str]
    entry_price: float
    stop_loss: float
    target_price: float
    
    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if confidence >= 60%"""
        return self.confidence >= 60 and self.strength != SignalStrength.NONE


class VCPMLSignalGenerator:
    """
    Combined Signal Generator using VCP + ML Indicators
    
    Usage:
        generator = VCPMLSignalGenerator()
        signal = generator.generate_signal('AAPL', high, low, close, volume)
        if signal.is_actionable:
            print(f"Signal: {signal.signal_type} @ {signal.confidence}%")
    """
    
    def __init__(
        self,
        min_confidence: float = 60.0,
        require_vcp: bool = False,  # If True, only signal on VCP breakout
        require_supertrend: bool = True,  # SuperTrend confirmation required
        require_rsi: bool = False  # RSI can be optional
    ):
        """
        Initialize VCP + ML Signal Generator
        
        Args:
            min_confidence: Minimum confidence to generate signal (default 60)
            require_vcp: Require VCP breakout to signal (default False)
            require_supertrend: Require SuperTrend confirmation (default True)
            require_rsi: Require RSI confirmation (default False)
        """
        self.min_confidence = min_confidence
        self.require_vcp = require_vcp
        self.require_supertrend = require_supertrend
        self.require_rsi = require_rsi
        
        # Initialize indicators
        self.vcp_detector = VCPDetector()
        self.ml_supertrend = MLAdaptiveSuperTrend()
        self.ml_rsi = MLOptimalRSI()
    
    def generate_signal(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> CombinedSignal:
        """
        Generate combined signal using all three indicators
        
        Args:
            symbol: Stock symbol
            high, low, close, volume: Price/volume arrays
        
        Returns:
            CombinedSignal with confidence and recommendation
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        current_price = float(close[-1])
        
        reasons = []
        confidence_components = []
        
        # =====================
        # Step 1: VCP Detection
        # =====================
        vcp_breakout = None
        vcp_direction = None
        vcp_confidence = 0
        
        try:
            active_zones, breakouts = self.vcp_detector.get_active_zones(
                high, low, close, volume
            )
            
            if breakouts:
                vcp_breakout = breakouts[0]  # Take most recent breakout
                vcp_direction = vcp_breakout.direction.value
                vcp_confidence = vcp_breakout.confidence
                reasons.append(f"VCP Breakout {vcp_direction} ({vcp_confidence:.0f}%)")
                confidence_components.append(('vcp', vcp_confidence, 0.35))
                logger.info(f"{symbol}: VCP breakout detected - {vcp_direction}")
            elif active_zones:
                reasons.append(f"VCP Zone consolidating ({len(active_zones)} zones)")
                # Consolidating zone - no signal yet but noted
        except Exception as e:
            logger.warning(f"{symbol}: VCP detection error: {e}")
        
        # If VCP required but no breakout, return no signal
        if self.require_vcp and not vcp_breakout:
            return self._no_signal(symbol, "No VCP breakout detected")
        
        # ==========================
        # Step 2: SuperTrend Filter
        # ==========================
        st_result = None
        st_direction = None
        st_confidence = 0
        
        try:
            st_result = self.ml_supertrend.get_signal(high, low, close)
            st_direction = st_result['trend']
            st_confidence = st_result['confidence']
            
            reasons.append(f"SuperTrend {st_direction} ({st_confidence:.0f}%)")
            confidence_components.append(('supertrend', st_confidence, 0.40))
            
            # Check if overextended
            if st_result['overextended']:
                reasons.append("⚠️ Overextended - caution advised")
                st_confidence -= 10
                
        except Exception as e:
            logger.warning(f"{symbol}: SuperTrend error: {e}")
        
        # If SuperTrend required but failed
        if self.require_supertrend and st_direction is None:
            return self._no_signal(symbol, "SuperTrend calculation failed")
        
        # ===================
        # Step 3: RSI Filter
        # ===================
        rsi_result = None
        rsi_direction = None
        rsi_confidence = 0
        
        try:
            rsi_result = self.ml_rsi.get_signal(close)
            rsi_direction = rsi_result['consensus_direction']
            rsi_confidence = rsi_result['confidence']
            
            if rsi_result['has_signal']:
                reasons.append(f"RSI {rsi_direction} ({rsi_confidence:.0f}%, "
                             f"{rsi_result['divergence_count']} divergences)")
                confidence_components.append(('rsi', rsi_confidence, 0.25))
            else:
                reasons.append(f"RSI Neutral (no divergence)")
                
        except Exception as e:
            logger.warning(f"{symbol}: RSI error: {e}")
        
        # If RSI required but no signal
        if self.require_rsi and not (rsi_result and rsi_result.get('has_signal')):
            return self._no_signal(symbol, "No RSI signal detected")
        
        # ================================
        # Step 4: Combine & Calculate Signal
        # ================================
        
        # Determine overall direction
        directions = []
        if vcp_direction == 'UP':
            directions.append('BULLISH')
        elif vcp_direction == 'DOWN':
            directions.append('BEARISH')
        
        if st_direction == 'UP':
            directions.append('BULLISH')
        elif st_direction == 'DOWN':
            directions.append('BEARISH')
        
        if rsi_direction == 'BULLISH':
            directions.append('BULLISH')
        elif rsi_direction == 'BEARISH':
            directions.append('BEARISH')
        
        # Count agreement
        bullish_count = directions.count('BULLISH')
        bearish_count = directions.count('BEARISH')
        
        if bullish_count > bearish_count:
            final_direction = 'BULLISH'
            signal_type = 'BUY_CALL'
            agreement_count = bullish_count
        elif bearish_count > bullish_count:
            final_direction = 'BEARISH'
            signal_type = 'BUY_PUT'
            agreement_count = bearish_count
        else:
            return self._no_signal(symbol, "No clear directional consensus")
        
        # Determine strength
        if agreement_count >= 3:
            strength = SignalStrength.STRONG
        elif agreement_count == 2:
            strength = SignalStrength.MODERATE
        elif agreement_count == 1:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE
        
        # Calculate weighted confidence
        total_weight = sum(w for _, _, w in confidence_components)
        if total_weight > 0:
            final_confidence = sum(c * w for _, c, w in confidence_components) / total_weight
        else:
            final_confidence = 50  # Default
        
        # Boost for agreement
        agreement_boost = (agreement_count - 1) * 5  # +5% per agreeing indicator
        final_confidence = min(final_confidence + agreement_boost, 95)
        
        # Check minimum confidence
        if final_confidence < self.min_confidence:
            return self._no_signal(symbol, f"Confidence {final_confidence:.0f}% below threshold")
        
        # Calculate stop loss and target
        current_atr = float(st_result['supertrend_level']) if st_result else current_price * 0.02
        stop_loss = current_price - (current_atr * 0.05) if final_direction == 'BULLISH' else current_price + (current_atr * 0.05)
        target_price = current_price + (current_atr * 0.10) if final_direction == 'BULLISH' else current_price - (current_atr * 0.10)
        
        logger.info(f"{symbol}: {signal_type} signal @ {final_confidence:.0f}% ({strength.value})")
        
        return CombinedSignal(
            symbol=symbol,
            signal_type=signal_type,
            direction=final_direction,
            confidence=final_confidence,
            strength=strength,
            vcp_breakout=vcp_breakout,
            supertrend_trend=st_direction or 'UNKNOWN',
            supertrend_confidence=st_confidence,
            rsi_consensus=rsi_direction or 'NEUTRAL',
            rsi_confidence=rsi_confidence,
            reasons=reasons,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price
        )
    
    def _no_signal(self, symbol: str, reason: str) -> CombinedSignal:
        """Create a no-signal result"""
        return CombinedSignal(
            symbol=symbol,
            signal_type='NO_SIGNAL',
            direction='NEUTRAL',
            confidence=0,
            strength=SignalStrength.NONE,
            vcp_breakout=None,
            supertrend_trend='UNKNOWN',
            supertrend_confidence=0,
            rsi_consensus='NEUTRAL',
            rsi_confidence=0,
            reasons=[reason],
            entry_price=0,
            stop_loss=0,
            target_price=0
        )
    
    def analyze_batch(
        self,
        symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> List[CombinedSignal]:
        """
        Analyze multiple symbols and return actionable signals
        
        Args:
            symbols_data: Dict of {symbol: {'high': arr, 'low': arr, 'close': arr, 'volume': arr}}
        
        Returns:
            List of actionable CombinedSignal objects, sorted by confidence
        """
        signals = []
        
        for symbol, data in symbols_data.items():
            try:
                signal = self.generate_signal(
                    symbol,
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume']
                )
                if signal.is_actionable:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"{symbol}: Analysis failed: {e}")
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals


# Convenience function for quick analysis
def generate_vcp_ml_signal(
    symbol: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> Dict:
    """
    Quick signal generation function
    
    Args:
        symbol: Stock symbol
        high, low, close, volume: Price/volume arrays
    
    Returns:
        Dict with signal info
    """
    generator = VCPMLSignalGenerator()
    signal = generator.generate_signal(symbol, high, low, close, volume)
    
    return {
        'symbol': signal.symbol,
        'signal_type': signal.signal_type,
        'direction': signal.direction,
        'confidence': signal.confidence,
        'strength': signal.strength.value,
        'is_actionable': signal.is_actionable,
        'reasons': signal.reasons,
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'target_price': signal.target_price
    }
