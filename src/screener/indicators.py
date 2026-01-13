"""
Technical Indicators
====================

Calculate MACD, RSI, Bollinger Bands, and Volume Ratio
for stock screening confirmation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of close prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_status(macd: float, signal: float, histogram: float) -> str:
    """
    Determine MACD status for trend confirmation.
    
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if macd > signal and histogram > 0:
        return "bullish"
    elif macd < signal and histogram < 0:
        return "bearish"
    else:
        return "neutral"


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        prices: Series of close prices
        period: RSI period (default 14)
    
    Returns:
        RSI series (0-100)
    
    Interpretation:
        RSI > 70: Overbought (pullback likely)
        RSI < 30: Oversold (bounce likely)
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def rsi_status(rsi_value: float) -> str:
    """
    Classify RSI status.
    
    Returns: 'overbought', 'oversold', or 'neutral'
    """
    if rsi_value >= 70:
        return "overbought"
    elif rsi_value <= 30:
        return "oversold"
    else:
        return "neutral"


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of close prices
        period: Moving average period (default 20)
        std_mult: Standard deviation multiplier (default 2)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def bb_position(price: float, upper: float, lower: float) -> str:
    """
    Determine Bollinger Band position.
    
    Returns: 'OVERBOUGHT', 'OVERSOLD', or 'NORMAL'
    """
    if price >= upper:
        return "OVERBOUGHT"
    elif price <= lower:
        return "OVERSOLD"
    else:
        return "NORMAL"


def volume_ratio(
    current_volume: float,
    volume_series: pd.Series,
    lookback: int = 20
) -> float:
    """
    Calculate volume ratio vs average.
    
    Args:
        current_volume: Current bar volume
        volume_series: Historical volume series
        lookback: Number of bars for average (default 20)
    
    Returns:
        Ratio (1.0 = average, 2.0 = 2x average)
    """
    avg_vol = volume_series.tail(lookback).mean()
    if avg_vol == 0 or pd.isna(avg_vol):
        return 1.0
    return current_volume / avg_vol


def get_all_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate all indicators from OHLCV DataFrame.
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
    
    Returns:
        Dictionary with all indicator values
    """
    if len(df) < 26:
        # Not enough data for MACD
        return {
            "macd": 0, "signal": 0, "histogram": 0, "macd_state": "neutral",
            "rsi": 50, "rsi_state": "neutral",
            "bb_upper": df['close'].iloc[-1] * 1.02,
            "bb_lower": df['close'].iloc[-1] * 0.98,
            "bb_pos": "NORMAL",
            "volume_ratio": 1.0
        }
    
    close = df['close']
    volume = df['volume']
    
    # MACD
    macd, sig, hist = calculate_macd(close)
    macd_st = macd_status(macd.iloc[-1], sig.iloc[-1], hist.iloc[-1])
    
    # RSI
    rsi = calculate_rsi(close)
    rsi_st = rsi_status(rsi.iloc[-1])
    
    # Bollinger Bands
    upper, mid, lower = bollinger_bands(close)
    bb_pos_str = bb_position(close.iloc[-1], upper.iloc[-1], lower.iloc[-1])
    
    # Volume
    vol_ratio = volume_ratio(volume.iloc[-1], volume)
    
    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(sig.iloc[-1]),
        "histogram": float(hist.iloc[-1]),
        "macd_state": macd_st,
        "rsi": float(rsi.iloc[-1]),
        "rsi_state": rsi_st,
        "bb_upper": float(upper.iloc[-1]),
        "bb_lower": float(lower.iloc[-1]),
        "bb_pos": bb_pos_str,
        "volume_ratio": vol_ratio
    }
