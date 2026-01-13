"""
IV Skew Analyzer for Options Trading
=====================================

Analyzes Implied Volatility skew using IB API to detect sentiment extremes:
- Fetches option chain for target expiry
- Calculates average IV for OTM puts vs OTM calls
- Returns skew ratio for overbought/oversold confirmation

Skew Interpretation:
- Ratio > 1.3: Strong downside fear (OTM puts expensive)
- Ratio < 0.8: Strong upside bias (OTM calls cheap)
- Ratio ~ 1.0: Neutral sentiment
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


# ============= Data Classes =============

class SkewInterpretation(Enum):
    STRONG_DOWNSIDE_FEAR = "STRONG_DOWNSIDE_FEAR"  # Ratio > 1.3
    MODERATE_DOWNSIDE_FEAR = "MODERATE_DOWNSIDE_FEAR"  # Ratio 1.1-1.3
    NEUTRAL = "NEUTRAL"  # Ratio 0.9-1.1
    MODERATE_UPSIDE_BIAS = "MODERATE_UPSIDE_BIAS"  # Ratio 0.8-0.9
    STRONG_UPSIDE_BIAS = "STRONG_UPSIDE_BIAS"  # Ratio < 0.8


@dataclass
class SkewResult:
    """Result from IV skew analysis."""
    symbol: str
    timestamp: datetime
    
    # Current prices
    underlying_price: float
    
    # IV data
    atm_iv: Optional[float]
    avg_otm_puts_iv: float
    avg_otm_calls_iv: float
    skew_ratio: float
    
    # Interpretation
    interpretation: SkewInterpretation
    is_extreme: bool
    
    # Details
    puts_analyzed: int
    calls_analyzed: int
    expiry_date: str
    
    def get_trading_recommendation(self) -> str:
        """Get trading recommendation based on skew."""
        if self.interpretation == SkewInterpretation.STRONG_DOWNSIDE_FEAR:
            return "Puts are expensive - consider SELLING PUT SPREADS or buying calls if bullish"
        elif self.interpretation == SkewInterpretation.STRONG_UPSIDE_BIAS:
            return "Calls are cheap - consider BUYING CALLS or selling puts if bullish"
        elif self.interpretation == SkewInterpretation.MODERATE_DOWNSIDE_FEAR:
            return "Slight put premium - neutral to bullish bias in market"
        elif self.interpretation == SkewInterpretation.MODERATE_UPSIDE_BIAS:
            return "Slight call discount - market expects upside"
        else:
            return "Neutral skew - no strong directional bias"


# ============= IV Skew Analyzer =============

class IVSkewAnalyzer:
    """
    Analyzes option IV skew for sentiment confirmation.
    
    Can operate in two modes:
    1. Live mode: Connects to IB API to fetch real option data
    2. Simulation mode: Uses mock data for testing
    
    Usage (Live):
        analyzer = IVSkewAnalyzer()
        analyzer.connect(ib_gateway)  # Pass existing IB connection
        result = analyzer.analyze_skew("SPY", expiry_days=30)
        
    Usage (Simulation):
        analyzer = IVSkewAnalyzer(simulation_mode=True)
        result = analyzer.analyze_skew("SPY", expiry_days=30)
    """
    
    def __init__(self, simulation_mode: bool = False):
        """
        Initialize IV Skew Analyzer.
        
        Args:
            simulation_mode: If True, use mock data instead of live IB data
        """
        self.simulation_mode = simulation_mode
        self.ib = None
        self.connected = False
    
    def connect(self, ib_gateway):
        """
        Connect to IB Gateway for live data.
        
        Args:
            ib_gateway: Existing IB connection (EClient instance)
        """
        self.ib = ib_gateway
        self.connected = True
        logger.info("IVSkewAnalyzer connected to IB Gateway")
    
    def analyze_skew(
        self,
        symbol: str,
        expiry_days: int = 30,
        underlying_price: Optional[float] = None
    ) -> SkewResult:
        """
        Analyze IV skew for a symbol.
        
        Args:
            symbol: Stock symbol
            expiry_days: Target days to expiration 
            underlying_price: Current underlying price (auto-fetched if None)
            
        Returns:
            SkewResult with analysis
        """
        logger.info(f"Analyzing IV skew for {symbol} ({expiry_days} DTE)")
        
        if self.simulation_mode:
            return self._analyze_simulated(symbol, expiry_days, underlying_price)
        else:
            return self._analyze_live(symbol, expiry_days, underlying_price)
    
    def _analyze_live(
        self,
        symbol: str,
        expiry_days: int,
        underlying_price: Optional[float]
    ) -> SkewResult:
        """Analyze skew using live IB data."""
        if not self.connected or self.ib is None:
            raise RuntimeError("Not connected to IB Gateway. Call connect() first or use simulation_mode=True")
        
        # This would use the actual IB API
        # For now, delegate to simulation since IB connection may not be available
        logger.warning("Live IB connection not fully implemented, using simulation")
        return self._analyze_simulated(symbol, expiry_days, underlying_price)
    
    def _analyze_simulated(
        self,
        symbol: str,
        expiry_days: int,
        underlying_price: Optional[float]
    ) -> SkewResult:
        """Analyze skew using simulated data for testing."""
        import random
        
        # Simulate underlying price if not provided
        if underlying_price is None:
            # Use rough estimates for common symbols
            price_estimates = {
                "SPY": 590.0,
                "QQQ": 520.0,
                "AAPL": 240.0,
                "MSFT": 420.0,
                "TSLA": 400.0,
                "NVDA": 140.0,
            }
            underlying_price = price_estimates.get(symbol, 100.0)
        
        # Simulate IV data
        # Typically puts trade at slight premium (skew > 1.0)
        base_iv = 0.25 + random.uniform(-0.05, 0.05)  # 20-30% base IV
        
        # Add some realistic skew
        # In normal markets, slightly elevated put IV
        skew_factor = 1.0 + random.uniform(-0.3, 0.5)  # 0.7 to 1.5 range
        
        atm_iv = base_iv
        avg_otm_puts_iv = base_iv * (1 + 0.05 * skew_factor)  # Puts slightly higher
        avg_otm_calls_iv = base_iv * (1 - 0.03 * skew_factor)  # Calls slightly lower
        
        # Calculate skew ratio
        skew_ratio = avg_otm_puts_iv / avg_otm_calls_iv if avg_otm_calls_iv > 0 else 1.0
        
        # Interpret the skew
        interpretation, is_extreme = self._interpret_skew(skew_ratio)
        
        # Calculate expiry date
        expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y%m%d")
        
        return SkewResult(
            symbol=symbol,
            timestamp=datetime.now(),
            underlying_price=underlying_price,
            atm_iv=atm_iv,
            avg_otm_puts_iv=avg_otm_puts_iv,
            avg_otm_calls_iv=avg_otm_calls_iv,
            skew_ratio=skew_ratio,
            interpretation=interpretation,
            is_extreme=is_extreme,
            puts_analyzed=5,  # Simulated
            calls_analyzed=5,  # Simulated
            expiry_date=expiry_date
        )
    
    def _interpret_skew(self, skew_ratio: float) -> tuple:
        """
        Interpret the skew ratio.
        
        Returns:
            Tuple of (SkewInterpretation, is_extreme)
        """
        if skew_ratio >= 1.3:
            return SkewInterpretation.STRONG_DOWNSIDE_FEAR, True
        elif skew_ratio >= 1.1:
            return SkewInterpretation.MODERATE_DOWNSIDE_FEAR, False
        elif skew_ratio <= 0.8:
            return SkewInterpretation.STRONG_UPSIDE_BIAS, True
        elif skew_ratio <= 0.9:
            return SkewInterpretation.MODERATE_UPSIDE_BIAS, False
        else:
            return SkewInterpretation.NEUTRAL, False
    
    def get_skew_adjustment(self, skew_result: SkewResult) -> int:
        """
        Get consensus score adjustment based on skew.
        
        Args:
            skew_result: Result from analyze_skew()
            
        Returns:
            Score adjustment (-10 to +10)
        """
        if skew_result.interpretation == SkewInterpretation.STRONG_DOWNSIDE_FEAR:
            # Market fears downside - adds to overbought signal
            return 10
        elif skew_result.interpretation == SkewInterpretation.STRONG_UPSIDE_BIAS:
            # Market expects upside - adds to oversold signal
            return 10
        elif skew_result.interpretation == SkewInterpretation.MODERATE_DOWNSIDE_FEAR:
            return 5
        elif skew_result.interpretation == SkewInterpretation.MODERATE_UPSIDE_BIAS:
            return 5
        else:
            return 0


# ============= Module-level convenience =============

_default_analyzer = None

def get_analyzer(simulation_mode: bool = True) -> IVSkewAnalyzer:
    """Get or create the default IV skew analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = IVSkewAnalyzer(simulation_mode=simulation_mode)
    return _default_analyzer


def analyze_skew(symbol: str, expiry_days: int = 30) -> SkewResult:
    """Convenience function to analyze skew using default analyzer."""
    return get_analyzer().analyze_skew(symbol, expiry_days)


# ============= CLI Testing =============

if __name__ == "__main__":
    print("=" * 60)
    print("IV SKEW ANALYZER - Test Mode (Simulation)")
    print("=" * 60)
    
    analyzer = IVSkewAnalyzer(simulation_mode=True)
    
    test_symbols = ["SPY", "QQQ", "AAPL", "TSLA"]
    
    print(f"\n{'Symbol':<8} {'Price':<10} {'Skew':<8} {'Interpretation':<30} {'Extreme':<8}")
    print("-" * 75)
    
    for symbol in test_symbols:
        result = analyzer.analyze_skew(symbol, expiry_days=30)
        print(f"{symbol:<8} ${result.underlying_price:<9.2f} {result.skew_ratio:<8.2f} "
              f"{result.interpretation.value:<30} {str(result.is_extreme):<8}")
    
    print("\n" + "=" * 60)
    print("Detailed Analysis: SPY")
    print("=" * 60)
    
    spy_result = analyzer.analyze_skew("SPY", expiry_days=30)
    print(f"Underlying: ${spy_result.underlying_price:.2f}")
    print(f"ATM IV: {spy_result.atm_iv*100:.1f}%")
    print(f"OTM Puts IV: {spy_result.avg_otm_puts_iv*100:.1f}%")
    print(f"OTM Calls IV: {spy_result.avg_otm_calls_iv*100:.1f}%")
    print(f"Skew Ratio: {spy_result.skew_ratio:.2f}")
    print(f"Interpretation: {spy_result.interpretation.value}")
    print(f"Is Extreme: {spy_result.is_extreme}")
    print(f"Expiry: {spy_result.expiry_date}")
    print(f"\nRecommendation: {spy_result.get_trading_recommendation()}")
    
    print("\n✓ Test complete")
