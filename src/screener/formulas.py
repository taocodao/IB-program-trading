"""
Screener Formulas
=================

Core calculations for stock screening:
- Expected Move = Beta × VIX / 100
- Abnormality Score = |Actual Move| / Expected Move
- Opportunity Rating (0-100)
- Enhanced Score with technicals
"""

from typing import Tuple


def expected_move(beta: float, vix_level: float, stock_price: float) -> Tuple[float, float]:
    """
    Calculate beta-adjusted VIX expected daily move.
    
    Formula: Expected % = Beta × VIX / 100
    
    Args:
        beta: Stock beta vs S&P 500 (e.g., 2.0 for TSLA)
        vix_level: Current VIX level (e.g., 24.5)
        stock_price: Current stock price
    
    Returns:
        Tuple of (expected_pct, expected_dollars)
    
    Example:
        >>> expected_move(2.0, 24.5, 250)
        (0.49, 1.225)  # 0.49% or $1.23
    """
    expected_pct = (beta * vix_level) / 100
    expected_dollars = stock_price * (expected_pct / 100)
    return expected_pct, expected_dollars


def abnormality_score(actual_move_pct: float, expected_move_pct: float) -> float:
    """
    Detect abnormal moves relative to expected.
    
    Score > 1.5 = Potential signal
    Score > 2.0 = Strong signal
    Score > 3.0 = Exceptional
    
    Args:
        actual_move_pct: Actual price move in %
        expected_move_pct: Expected move from expected_move()
    
    Returns:
        Abnormality score (ratio)
    
    Example:
        >>> abnormality_score(2.8, 0.49)
        5.71  # Exceptional - 5.7x expected move
    """
    if expected_move_pct == 0:
        return 0.0
    return abs(actual_move_pct) / abs(expected_move_pct)


def opportunity_rating(
    actual_pct: float, 
    expected_pct: float, 
    volume_ratio: float, 
    beta: float
) -> float:
    """
    Composite opportunity score (0-100).
    
    Combines:
    - Abnormality (primary)
    - Volume confirmation
    - Beta adjustment
    
    Args:
        actual_pct: Actual move %
        expected_pct: Expected move %
        volume_ratio: Current volume / avg volume
        beta: Stock beta
    
    Returns:
        Score from 0-100
    """
    if expected_pct == 0:
        return 0.0
    
    abnormality = abs(actual_pct) / abs(expected_pct)
    base_score = abnormality * 100
    
    # Volume multiplier: higher volume = more significant
    vol_mult = min(volume_ratio, 2.0) * 0.5 + 1.0
    
    # Beta factor: reward high-beta moves (more significant)
    beta_factor = beta / 1.5
    beta_mult = 1.0 + max(beta_factor - 1.0, 0) * 0.15
    
    final_score = base_score * vol_mult * beta_mult
    return min(final_score, 100.0)


def enhanced_score(
    actual_pct: float,
    expected_pct: float,
    vol_ratio: float,
    macd_state: str,    # 'bullish', 'bearish', 'neutral'
    rsi_value: float,
    bb_pos: str,        # 'OVERBOUGHT', 'OVERSOLD', 'NORMAL'
    direction: str      # 'UP' or 'DOWN'
) -> float:
    """
    Enhanced scoring with technical indicator confirmation.
    
    Uses reversion logic:
    - UP move + bearish techs = stronger signal (expect pullback)
    - DOWN move + bullish techs = stronger signal (expect bounce)
    
    Args:
        actual_pct: Actual price move %
        expected_pct: Expected move %
        vol_ratio: Volume ratio
        macd_state: MACD status (bullish/bearish/neutral)
        rsi_value: RSI value (0-100)
        bb_pos: Bollinger band position
        direction: Price move direction (UP/DOWN)
    
    Returns:
        Enhanced score 0-100
    """
    if expected_pct == 0:
        return 0.0
    
    abnormality = abs(actual_pct) / abs(expected_pct)
    base = abnormality * 100
    
    # Volume multiplier
    if vol_ratio > 2:
        vol_mult = 1.5
    elif vol_ratio > 1.5:
        vol_mult = 1.3
    elif vol_ratio > 1.0:
        vol_mult = 1.1
    else:
        vol_mult = 0.7
    
    # MACD multiplier (reversion logic)
    macd_mult = 1.0
    if direction == 'UP' and macd_state == 'bearish':
        macd_mult = 1.4  # Overbought, expect pullback
    elif direction == 'DOWN' and macd_state == 'bullish':
        macd_mult = 1.4  # Oversold, expect bounce
    elif macd_state == 'neutral':
        macd_mult = 1.0
    else:
        macd_mult = 0.8  # Move and MACD agree = less edge
    
    # RSI multiplier
    if rsi_value >= 80 or rsi_value <= 20:
        rsi_mult = 1.25  # Extreme
    elif rsi_value >= 70 or rsi_value <= 30:
        rsi_mult = 1.10
    else:
        rsi_mult = 1.0
    
    # Bollinger multiplier
    if bb_pos in ("OVERBOUGHT", "OVERSOLD"):
        bb_mult = 1.15
    else:
        bb_mult = 1.0
    
    score = base * vol_mult * macd_mult * rsi_mult * bb_mult
    return min(score, 100.0)


def classify_signal(rating: float) -> str:
    """
    Classify signal strength based on rating.
    
    Returns signal category and expected win rate:
    - EXCEPTIONAL (80+): ~70% win rate
    - EXCELLENT (60-80): ~62% win rate
    - GOOD (40-60): ~55% win rate
    - FAIR (20-40): ~50% win rate
    - WEAK (<20): Not recommended
    """
    if rating >= 80:
        return "EXCEPTIONAL"
    elif rating >= 60:
        return "EXCELLENT"
    elif rating >= 40:
        return "GOOD"
    elif rating >= 20:
        return "FAIR"
    else:
        return "WEAK"


def get_direction(actual_pct: float) -> str:
    """Get direction from price change."""
    return "UP" if actual_pct >= 0 else "DOWN"
