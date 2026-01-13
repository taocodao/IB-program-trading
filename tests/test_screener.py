"""
Test Screener Components
========================

Verify formulas and indicators work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from screener.formulas import (
    expected_move, abnormality_score, opportunity_rating,
    enhanced_score, classify_signal, get_direction
)
from screener.indicators import (
    calculate_macd, calculate_rsi, bollinger_bands,
    volume_ratio, macd_status, rsi_status, bb_position
)
import pandas as pd
import numpy as np


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_expected_move():
    print_header("TEST 1: Expected Move Formula")
    
    test_cases = [
        ("AAPL", 1.25, 20, 260),
        ("TSLA", 2.00, 24, 445),
        ("SPY", 1.00, 18, 627),
        ("NVDA", 1.80, 30, 185),
    ]
    
    print(f"\n{'Symbol':<8} {'Beta':<6} {'VIX':<6} {'Price':<10} {'Exp %':<10} {'Exp $':<10}")
    print("-" * 60)
    
    for symbol, beta, vix, price in test_cases:
        exp_pct, exp_dollars = expected_move(beta, vix, price)
        print(f"{symbol:<8} {beta:<6.2f} {vix:<6} ${price:<9.2f} {exp_pct:<10.3f}% ${exp_dollars:<9.2f}")


def test_abnormality():
    print_header("TEST 2: Abnormality Score")
    
    test_cases = [
        (0.5, 0.25, "Normal volatility"),
        (1.0, 0.5, "2x expected"),
        (2.5, 0.5, "5x expected - EXCEPTIONAL"),
        (-3.0, 0.4, "7.5x expected (down move)"),
    ]
    
    print(f"\n{'Actual':<10} {'Expected':<10} {'Score':<10} {'Note':<25}")
    print("-" * 60)
    
    for actual, expected, note in test_cases:
        score = abnormality_score(actual, expected)
        print(f"{actual:<10.2f}% {expected:<10.2f}% {score:<10.2f}x {note}")


def test_enhanced_scoring():
    print_header("TEST 3: Enhanced Scoring with Technicals")
    
    # Base case
    base_score = enhanced_score(
        actual_pct=2.5, expected_pct=0.5,
        vol_ratio=1.5, macd_state="neutral",
        rsi_value=55, bb_pos="NORMAL", direction="UP"
    )
    
    # With bullish divergence (bearish MACD on UP move = reversion signal)
    reversion_score = enhanced_score(
        actual_pct=2.5, expected_pct=0.5,
        vol_ratio=2.0, macd_state="bearish",
        rsi_value=78, bb_pos="OVERBOUGHT", direction="UP"
    )
    
    # With confirming technicals (less edge)
    confirming_score = enhanced_score(
        actual_pct=2.5, expected_pct=0.5,
        vol_ratio=1.2, macd_state="bullish",
        rsi_value=65, bb_pos="NORMAL", direction="UP"
    )
    
    print(f"\nScenario                           Score   Signal")
    print("-" * 50)
    print(f"Base (neutral technicals):         {base_score:.1f}     {classify_signal(base_score)}")
    print(f"Reversion (bearish on UP):         {reversion_score:.1f}    {classify_signal(reversion_score)}")
    print(f"Confirming (bullish on UP):        {confirming_score:.1f}     {classify_signal(confirming_score)}")


def test_indicators():
    print_header("TEST 4: Technical Indicators")
    
    # Generate sample price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    volume = np.random.randint(100000, 500000, 100)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volume
    })
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    status = macd_status(macd.iloc[-1], signal.iloc[-1], hist.iloc[-1])
    print(f"\nMACD: {macd.iloc[-1]:.4f}")
    print(f"Signal: {signal.iloc[-1]:.4f}")
    print(f"Histogram: {hist.iloc[-1]:.4f}")
    print(f"Status: {status}")
    
    # RSI
    rsi = calculate_rsi(df['close'])
    print(f"\nRSI: {rsi.iloc[-1]:.2f}")
    print(f"Status: {rsi_status(rsi.iloc[-1])}")
    
    # Bollinger Bands
    upper, mid, lower = bollinger_bands(df['close'])
    bb_pos = bb_position(df['close'].iloc[-1], upper.iloc[-1], lower.iloc[-1])
    print(f"\nBB Upper: {upper.iloc[-1]:.2f}")
    print(f"BB Middle: {mid.iloc[-1]:.2f}")
    print(f"BB Lower: {lower.iloc[-1]:.2f}")
    print(f"Current Price: {df['close'].iloc[-1]:.2f}")
    print(f"Position: {bb_pos}")
    
    # Volume
    vol_r = volume_ratio(df['volume'].iloc[-1], df['volume'])
    print(f"\nVolume Ratio: {vol_r:.2f}x")


def test_signal_classification():
    print_header("TEST 5: Signal Classification")
    
    scores = [15, 25, 45, 65, 85, 100]
    
    print(f"\n{'Score':<10} {'Signal':<15} {'Expected Win Rate':<20}")
    print("-" * 50)
    
    win_rates = {
        "EXCEPTIONAL": "~70%",
        "EXCELLENT": "~62%",
        "GOOD": "~55%",
        "FAIR": "~50%",
        "WEAK": "< 50%"
    }
    
    for score in scores:
        signal = classify_signal(score)
        wr = win_rates.get(signal, "")
        print(f"{score:<10} {signal:<15} {wr:<20}")


def test_full_scenario():
    print_header("TEST 6: Full Screening Scenario")
    
    print("\nSimulating TSLA alert:")
    print("-" * 40)
    
    # Data
    beta = 2.0
    vix = 24.5
    price = 445.0
    prev_close = 430.0
    
    actual_pct = (price - prev_close) / prev_close * 100
    direction = get_direction(actual_pct)
    exp_pct, exp_dollars = expected_move(beta, vix, price)
    abn = abnormality_score(actual_pct, exp_pct)
    
    score = enhanced_score(
        actual_pct, exp_pct,
        vol_ratio=2.1,
        macd_state="bearish",
        rsi_value=76,
        bb_pos="OVERBOUGHT",
        direction=direction
    )
    signal = classify_signal(score)
    
    print(f"  Symbol: TSLA")
    print(f"  Price: ${price:.2f} (prev: ${prev_close:.2f})")
    print(f"  Move: {actual_pct:+.2f}% ({direction})")
    print(f"  Expected: {exp_pct:.2f}%")
    print(f"  Abnormality: {abn:.2f}x")
    print(f"  Volume: 2.1x average")
    print(f"  MACD: bearish (divergence!)")
    print(f"  RSI: 76 (overbought)")
    print(f"  BB: OVERBOUGHT")
    print(f"  ---")
    print(f"  Score: {score:.1f}")
    print(f"  Signal: {signal}")
    
    if abn >= 1.5 and score >= 60:
        print(f"\n  *** ALERT TRIGGERED ***")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCREENER COMPONENT TESTS")
    print("=" * 60)
    
    test_expected_move()
    test_abnormality()
    test_enhanced_scoring()
    test_indicators()
    test_signal_classification()
    test_full_scenario()
    
    print("\n" + "=" * 60)
    print("✓ All tests complete")
    print("=" * 60 + "\n")
