"""
VCP + ML Indicators - Real Data Test
=====================================

Tests the VCP + ML indicators with real market data from Yahoo Finance
(since IB Gateway requires market hours).

Usage:
    python tests/test_vcp_real_data.py
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Try to import yfinance for real data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Run: pip install yfinance")

from indicators import (
    VCPDetector,
    MLAdaptiveSuperTrend,
    MLOptimalRSI,
    VCPMLSignalGenerator,
    analyze_vcp,
    analyze_supertrend,
    analyze_rsi
)


def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed")
    
    print(f"Fetching {symbol} data ({period})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    # Rename columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    print(f"   Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def test_single_symbol(symbol: str):
    """Test VCP+ML indicators on a single symbol"""
    print(f"\n{'='*60}")
    print(f"Testing {symbol}")
    print('='*60)
    
    # Fetch data
    df = fetch_stock_data(symbol, "1y")
    
    # Extract arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    current_price = close[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    
    # Test 1: VCP Detection
    print(f"\n[1] VCP DETECTOR")
    print("-" * 40)
    vcp = VCPDetector()
    zones = vcp.detect_vcp_zones(high, low, close, volume)
    active_zones, breakouts = vcp.get_active_zones(high, low, close, volume)
    
    print(f"   Total VCP zones found: {len(zones)}")
    print(f"   Active zones (last 50 bars): {len(active_zones)}")
    print(f"   Recent breakouts: {len(breakouts)}")
    
    if active_zones:
        zone = active_zones[-1]
        print(f"   Latest zone: ${zone.low:.2f} - ${zone.high:.2f} ({zone.consolidation_bars} bars)")
    
    if breakouts:
        b = breakouts[0]
        print(f"   🚀 BREAKOUT: {b.direction.value} @ ${b.breakout_price:.2f} ({b.confidence:.0f}%)")
    
    # Test 2: ML SuperTrend
    print(f"\n[2] ML ADAPTIVE SUPERTREND")
    print("-" * 40)
    st = MLAdaptiveSuperTrend()
    st_result = st.get_signal(high, low, close)
    
    print(f"   Trend: {st_result['trend']}")
    print(f"   Volatility: {st_result['volatility_class']}")
    print(f"   SuperTrend Level: ${st_result['supertrend_level']:.2f}")
    print(f"   Confidence: {st_result['confidence']:.1f}%")
    print(f"   Overextended: {st_result['overextended']}")
    
    # Test 3: ML Optimal RSI
    print(f"\n[3] ML OPTIMAL RSI")
    print("-" * 40)
    rsi = MLOptimalRSI()
    rsi_result = rsi.get_signal(close)
    
    print(f"   Consensus: {rsi_result['consensus_direction']}")
    print(f"   Divergences: {rsi_result['divergence_count']}")
    print(f"   Confidence: {rsi_result['confidence']:.1f}%")
    print(f"   RSI(14): {rsi_result['current_rsi'].get(14, 0):.1f}")
    
    # Test 4: Combined Signal
    print(f"\n[4] COMBINED SIGNAL (VCP + ML)")
    print("-" * 40)
    generator = VCPMLSignalGenerator()
    signal = generator.generate_signal(symbol, high, low, close, volume)
    
    print(f"   Signal: {signal.signal_type}")
    print(f"   Direction: {signal.direction}")
    print(f"   Confidence: {signal.confidence:.1f}%")
    print(f"   Strength: {signal.strength.value}")
    print(f"   Actionable: {signal.is_actionable}")
    
    if signal.reasons:
        print(f"   Reasons:")
        for reason in signal.reasons:
            print(f"      • {reason}")
    
    return signal


def run_batch_scan(symbols: list):
    """Run VCP+ML scan on multiple symbols"""
    print("\n" + "="*60)
    print("BATCH SCAN - VCP + ML INDICATORS")
    print("="*60)
    
    generator = VCPMLSignalGenerator()
    actionable_signals = []
    
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, "1y")
            signal = generator.generate_signal(
                symbol,
                df['high'].values,
                df['low'].values,
                df['close'].values,
                df['volume'].values
            )
            
            status = "✅" if signal.is_actionable else "❌"
            print(f"   {status} {symbol}: {signal.signal_type} @ {signal.confidence:.0f}% ({signal.strength.value})")
            
            if signal.is_actionable:
                actionable_signals.append(signal)
                
        except Exception as e:
            print(f"   ⚠️ {symbol}: Error - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(actionable_signals)}/{len(symbols)} actionable signals")
    print("="*60)
    
    if actionable_signals:
        print("\nActionable Signals:")
        for sig in sorted(actionable_signals, key=lambda x: x.confidence, reverse=True):
            print(f"   🎯 {sig.symbol}: {sig.signal_type} @ {sig.confidence:.0f}%")
            for reason in sig.reasons[:2]:
                print(f"      └ {reason}")
    
    return actionable_signals


if __name__ == "__main__":
    # Check yfinance
    if not YFINANCE_AVAILABLE:
        print("\n❌ Please install yfinance: pip install yfinance")
        sys.exit(1)
    
    print("="*60)
    print("VCP + ML INDICATORS - REAL DATA TEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test individual symbols
    test_symbols = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA"]
    
    for symbol in test_symbols:
        try:
            test_single_symbol(symbol)
        except Exception as e:
            print(f"\n❌ {symbol} failed: {e}")
    
    # Batch scan (optional)
    print("\n\n")
    run_batch_scan(["AMD", "META", "GOOGL", "AMZN", "NFLX"])
    
    print("\n✅ TEST COMPLETE")
