# Screener package
"""
Real-Time Stock Screener
========================

Components:
- formulas.py: Expected move, abnormality score, scoring
- indicators.py: MACD, RSI, Bollinger, Volume
- ib_gateway.py: IB TWS/Gateway connection
- data_store.py: Watchlist, prev close, alerts
- main_screener.py: Entry point

Usage:
    python -m screener.main_screener
"""

from .formulas import (
    expected_move,
    abnormality_score,
    opportunity_rating,
    enhanced_score,
    classify_signal
)

from .indicators import (
    calculate_macd,
    calculate_rsi,
    bollinger_bands,
    volume_ratio
)

__all__ = [
    "expected_move",
    "abnormality_score",
    "opportunity_rating",
    "enhanced_score",
    "classify_signal",
    "calculate_macd",
    "calculate_rsi",
    "bollinger_bands",
    "volume_ratio"
]
