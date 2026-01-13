"""
Indicator Utility Functions
Shared calculations for ATR, SMA, EMA, percentiles
"""
import numpy as np
from typing import List, Dict, Optional, Union


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 20) -> np.ndarray:
    """
    Calculate Average True Range (ATR)
    
    ATR measures volatility in absolute dollar terms.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    ATR = EMA of True Ranges
    
    Args:
        high: Array of high prices
        low: Array of low prices  
        close: Array of close prices
        period: ATR period (default 20)
    
    Returns:
        Array of ATR values
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    
    n = len(close)
    true_ranges = np.zeros(n)
    
    # First bar: simple high-low range
    true_ranges[0] = high[0] - low[0]
    
    # Subsequent bars: max of (H-L, |H-Cprev|, |L-Cprev|)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        true_ranges[i] = max(tr1, tr2, tr3)
    
    # Calculate EMA of true ranges
    atr = np.zeros(n)
    if n >= period:
        # First ATR is simple average
        atr[period-1] = np.mean(true_ranges[:period])
        
        # Subsequent values use EMA
        multiplier = 2 / (period + 1)
        for i in range(period, n):
            atr[i] = (true_ranges[i] - atr[i-1]) * multiplier + atr[i-1]
    
    return atr


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        data: Array of prices
        period: SMA period
    
    Returns:
        Array of SMA values (NaN for insufficient data)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: Array of prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    ema = np.zeros(n)
    
    if n < period:
        return ema
    
    # First EMA is SMA
    ema[period-1] = np.mean(data[:period])
    
    # Subsequent EMAs
    multiplier = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_percentile(data: np.ndarray, percentile: float) -> float:
    """
    Calculate percentile value from data
    
    Args:
        data: Array of values
        percentile: Percentile (0-100)
    
    Returns:
        Value at given percentile
    """
    data = np.asarray(data, dtype=float)
    # Remove NaN and zero values
    clean_data = data[~np.isnan(data) & (data > 0)]
    if len(clean_data) == 0:
        return 0.0
    return float(np.percentile(clean_data, percentile))


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, 
                               std_mult: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Calculate Bollinger Bands
    
    Args:
        close: Array of close prices
        period: SMA period (default 20)
        std_mult: Standard deviation multiplier (default 2.0)
    
    Returns:
        Dict with 'middle', 'upper', 'lower', 'width' arrays
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    
    middle = calculate_sma(close, period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    width = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        std_dev = np.std(window)
        upper[i] = middle[i] + (std_mult * std_dev)
        lower[i] = middle[i] - (std_mult * std_dev)
        width[i] = upper[i] - lower[i]
    
    return {
        'middle': middle,
        'upper': upper,
        'lower': lower,
        'width': width
    }


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    Args:
        prices: Array of close prices
        period: RSI period (default 14)
    
    Returns:
        Array of RSI values (0-100)
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    # First RSI calculation (simple average)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # Subsequent RSI calculations (smoothed)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i + 1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - (100 / (1 + rs))
    
    return rsi
