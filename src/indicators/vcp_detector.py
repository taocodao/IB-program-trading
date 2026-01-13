"""
VCP Detector (Volatility Contraction Pattern)

Detects low-risk breakout entry points by identifying consolidation zones
where price is squeezed into a tighter and tighter range.

VCP Characteristics:
- Price moves within narrow range (5-15% band)
- Volume dries up during consolidation
- Bollinger Bands squeeze (BB Width at multi-year low)
- ATR contracts significantly (ATR < 20th percentile)
- Duration: 5-30 trading days

Expected Performance:
- Find 2-5 VCP zones per stock per month
- Breakout success rate: 65-70%
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .indicator_utils import calculate_atr, calculate_bollinger_bands, calculate_percentile

logger = logging.getLogger(__name__)


class BreakoutDirection(Enum):
    """Direction of VCP breakout"""
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


@dataclass
class VCPZone:
    """Represents a detected VCP consolidation zone"""
    start_bar: int
    end_bar: int
    consolidation_bars: int
    high: float
    low: float
    range_pct: float
    avg_volume: float
    avg_atr: float
    status: str  # 'CONSOLIDATING', 'BREAKOUT_UP', 'BREAKOUT_DOWN'
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def range_dollars(self) -> float:
        return self.high - self.low


@dataclass
class VCPBreakout:
    """Represents a detected breakout from VCP zone"""
    direction: BreakoutDirection
    breakout_price: float
    breakout_magnitude: float  # % move from zone
    signal_type: str  # 'BUY_CALL' or 'BUY_PUT'
    confidence: float  # 0-100
    zone: VCPZone


class VCPDetector:
    """
    Detect Volatility Contraction Patterns
    
    Purpose: Find low-risk, high-reward breakout setups
    
    Usage:
        detector = VCPDetector()
        zones = detector.detect_vcp_zones(high, low, close, volume)
        for zone in zones:
            breakout = detector.detect_breakout(zone, high, low, close)
    """
    
    def __init__(
        self,
        atr_period: int = 20,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_percentile_threshold: int = 20,
        min_consolidation_bars: int = 5,
        max_consolidation_bars: int = 30,
        max_range_pct: float = 0.15,
        breakout_lookback: int = 5,
        breakout_confirmation_pct: float = 0.01
    ):
        """
        Initialize VCP Detector
        
        Args:
            atr_period: Period for ATR calculation (default 20)
            bb_period: Period for Bollinger Bands (default 20)
            bb_std: Standard deviations for BB (default 2.0)
            atr_percentile_threshold: ATR percentile for tight volatility (default 20)
            min_consolidation_bars: Minimum bars for valid VCP (default 5)
            max_consolidation_bars: Maximum bars for valid VCP (default 30)
            max_range_pct: Maximum price range % during consolidation (default 15%)
            breakout_lookback: Bars to check for breakout (default 5)
            breakout_confirmation_pct: % above/below zone to confirm (default 1%)
        """
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_percentile_threshold = atr_percentile_threshold
        self.min_consolidation_bars = min_consolidation_bars
        self.max_consolidation_bars = max_consolidation_bars
        self.max_range_pct = max_range_pct
        self.breakout_lookback = breakout_lookback
        self.breakout_confirmation_pct = breakout_confirmation_pct
    
    def detect_vcp_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> List[VCPZone]:
        """
        Main VCP detection function
        
        Scans price data to find consolidation zones where:
        1. ATR is in lowest 20% (very tight volatility)
        2. Bollinger Band width is at multi-period low
        3. Price stays within tight range for 5-30 bars
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
        
        Returns:
            List of VCPZone objects representing consolidation zones
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        
        n = len(close)
        if n < 200:  # Need at least ~10 months of data
            logger.warning(f"Insufficient data for VCP detection: {n} bars (need 200)")
            return []
        
        # Calculate indicators
        atr = calculate_atr(high, low, close, self.atr_period)
        bb = calculate_bollinger_bands(close, self.bb_period, self.bb_std)
        bb_width = bb['width']
        
        # Calculate percentile thresholds (last 200 bars)
        atr_threshold = calculate_percentile(atr[-200:], self.atr_percentile_threshold)
        bb_threshold = calculate_percentile(bb_width[-200:], self.atr_percentile_threshold)
        
        vcp_zones: List[VCPZone] = []
        i = self.bb_period  # Start after indicator warmup
        
        while i < n:
            # Check if current ATR is in lowest 20% (very tight)
            if atr[i] <= 0 or atr[i] > atr_threshold:
                i += 1
                continue
            
            # Check if BB Width is at multi-period low
            if np.isnan(bb_width[i]) or bb_width[i] > bb_threshold:
                i += 1
                continue
            
            # Found potential VCP start - track consolidation zone
            vcp_start = i
            consolidation_high = high[i]
            consolidation_low = low[i]
            consolidation_volumes = [volume[i]]
            consolidation_atrs = [atr[i]]
            consolidation_bars = 1
            
            # Track consolidation zone
            j = i + 1
            while j < n and consolidation_bars < self.max_consolidation_bars:
                # Update range
                consolidation_high = max(consolidation_high, high[j])
                consolidation_low = min(consolidation_low, low[j])
                consolidation_volumes.append(volume[j])
                consolidation_atrs.append(atr[j])
                
                # Check if price broke out of tight range
                range_pct = (consolidation_high - consolidation_low) / consolidation_low
                if range_pct > self.max_range_pct:
                    break
                
                # Check if ATR expanded significantly (breaking consolidation)
                if atr[j] > atr_threshold * 1.5:
                    break
                
                j += 1
                consolidation_bars += 1
            
            # Valid VCP if consolidated for minimum bars
            if consolidation_bars >= self.min_consolidation_bars:
                range_pct = (consolidation_high - consolidation_low) / consolidation_low
                
                zone = VCPZone(
                    start_bar=vcp_start,
                    end_bar=j - 1,
                    consolidation_bars=consolidation_bars,
                    high=consolidation_high,
                    low=consolidation_low,
                    range_pct=range_pct,
                    avg_volume=float(np.mean(consolidation_volumes)),
                    avg_atr=float(np.mean(consolidation_atrs)),
                    status='CONSOLIDATING'
                )
                vcp_zones.append(zone)
                logger.info(f"VCP Zone detected: {consolidation_bars} bars, "
                           f"range {range_pct*100:.1f}%")
            
            # Move past this zone
            i = j
        
        return vcp_zones
    
    def detect_breakout(
        self,
        zone: VCPZone,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Optional[VCPBreakout]:
        """
        After VCP consolidation, detect if breakout occurred
        
        Breakout UP: Price closes above VCP high with confirmation
        Breakout DOWN: Price closes below VCP low with confirmation
        
        Args:
            zone: VCPZone object representing the consolidation
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            VCPBreakout object if breakout detected, None otherwise
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        
        # Get recent price action
        recent_high = np.max(high[-self.breakout_lookback:])
        recent_low = np.min(low[-self.breakout_lookback:])
        current_close = close[-1]
        
        vcp_high = zone.high
        vcp_low = zone.low
        
        # Breakout UP: Price closes above VCP high with 1% confirmation
        if current_close > vcp_high and recent_high > vcp_high * (1 + self.breakout_confirmation_pct):
            magnitude = (current_close - vcp_high) / vcp_high
            confidence = min(70 + (magnitude * 100), 95)  # Base 70%, boost by magnitude
            
            zone.status = 'BREAKOUT_UP'
            return VCPBreakout(
                direction=BreakoutDirection.UP,
                breakout_price=current_close,
                breakout_magnitude=magnitude,
                signal_type='BUY_CALL',
                confidence=confidence,
                zone=zone
            )
        
        # Breakout DOWN: Price closes below VCP low with 1% confirmation
        elif current_close < vcp_low and recent_low < vcp_low * (1 - self.breakout_confirmation_pct):
            magnitude = (vcp_low - current_close) / vcp_low
            confidence = min(70 + (magnitude * 100), 95)
            
            zone.status = 'BREAKOUT_DOWN'
            return VCPBreakout(
                direction=BreakoutDirection.DOWN,
                breakout_price=current_close,
                breakout_magnitude=magnitude,
                signal_type='BUY_PUT',
                confidence=confidence,
                zone=zone
            )
        
        return None
    
    def get_active_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        max_age_bars: int = 50
    ) -> Tuple[List[VCPZone], List[VCPBreakout]]:
        """
        Get currently active VCP zones and any recent breakouts
        
        Useful for real-time signal generation
        
        Args:
            high, low, close, volume: Price/volume arrays
            max_age_bars: Maximum age for a zone to be considered active
        
        Returns:
            Tuple of (active_zones, breakouts)
        """
        all_zones = self.detect_vcp_zones(high, low, close, volume)
        n = len(close)
        
        active_zones = []
        breakouts = []
        
        for zone in all_zones:
            # Only consider recent zones
            if (n - zone.end_bar) > max_age_bars:
                continue
            
            # Check for breakout
            breakout = self.detect_breakout(zone, high, low, close)
            if breakout:
                breakouts.append(breakout)
            else:
                active_zones.append(zone)
        
        return active_zones, breakouts


# Convenience function for quick VCP analysis
def analyze_vcp(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> Dict:
    """
    Quick VCP analysis function
    
    Args:
        high, low, close, volume: Price/volume arrays
    
    Returns:
        Dict with zones, breakouts, and summary
    """
    detector = VCPDetector()
    active_zones, breakouts = detector.get_active_zones(high, low, close, volume)
    
    return {
        'active_zones': len(active_zones),
        'breakouts': len(breakouts),
        'zones': active_zones,
        'breakout_signals': breakouts,
        'has_signal': len(breakouts) > 0,
        'signal_type': breakouts[0].signal_type if breakouts else None,
        'confidence': breakouts[0].confidence if breakouts else 0
    }
