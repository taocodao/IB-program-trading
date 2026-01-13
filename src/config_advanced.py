"""
Advanced Configuration for Volatility-Aware Stop-Loss System
=============================================================

All settings can be overridden via environment variables.
"""

import os
from datetime import time
from dotenv import load_dotenv

load_dotenv()

# ============= IB CONNECTION =============

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))  # 7497=TWS, 4002=Gateway
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "101"))  # Different from simple system

# ============= STRATEGY PARAMETERS =============

# Aggression factor: scales the beta × vol stop distance
# 0.7 = conservative (tighter stops)
# 1.0 = default (balanced)
# 1.5 = aggressive (wider stops, more room)
K_AGGRESSION = float(os.getenv("K_AGGRESSION", "1.0"))

# Minimum and maximum trail percentages (clamping)
MIN_TRAIL_PCT = float(os.getenv("MIN_TRAIL_PCT", "0.04"))   # 4%
MAX_TRAIL_PCT = float(os.getenv("MAX_TRAIL_PCT", "0.40"))   # 40%

# DTE multipliers (high gamma options need wider stops)
DTE_30_PLUS_MULTIPLIER = 1.0      # Normal
DTE_7_30_MULTIPLIER = 1.5         # 50% wider
DTE_UNDER_7_MULTIPLIER = 2.0      # 100% wider

# ============= VOLATILITY SETTINGS =============

# Use VIX for index volatility (vs ATR)
USE_VIX = True
VIX_UPDATE_INTERVAL = 30  # seconds

# Fallback daily volatility if VIX unavailable (1.2% ~ VIX 19)
DEFAULT_DAILY_VOL = 0.012

# ============= EXIT EXECUTION =============

# How deep into bid-ask spread to place limit (0=bid, 1=ask)
EXIT_SPREAD_PARTICIPATION = float(os.getenv("EXIT_SPREAD_PARTICIPATION", "0.5"))

# Max slippage below theoretical price allowed
EXIT_ALLOWED_SLIPPAGE_PCT = float(os.getenv("EXIT_ALLOWED_SLIPPAGE_PCT", "0.03"))

# Maximum reprice attempts before falling back to bid
EXIT_MAX_REPRICES = int(os.getenv("EXIT_MAX_REPRICES", "5"))

# Seconds between reprice attempts
EXIT_REPRICE_INTERVAL = int(os.getenv("EXIT_REPRICE_INTERVAL", "10"))

# ============= MARKET HOURS =============

MARKET_OPEN_TIME = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET

# ============= LOGGING =============

LOG_FILE = os.getenv("LOG_FILE", "logs/volatility_stops.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============= PAPER TRADING =============

PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")

# ============= FILTERING =============

_allowed = os.getenv("ALLOWED_SYMBOLS", "")
ALLOWED_SYMBOLS = [s.strip() for s in _allowed.split(",") if s.strip()]

_excluded = os.getenv("EXCLUDED_SYMBOLS", "")
EXCLUDED_SYMBOLS = [s.strip() for s in _excluded.split(",") if s.strip()]

MIN_POSITION_QUANTITY = int(os.getenv("MIN_POSITION_QUANTITY", "1"))

# ============= RECONNECTION =============

RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "5"))
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "10"))

# ============= RISK-FREE RATE =============

# Used for Black-Scholes calculations
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

# ============= VIX REQUEST ID =============

VIX_REQ_ID = 9999  # Reserved request ID for VIX subscription

# ============= PORTFOLIO SETTINGS =============

PORTFOLIO_SIZE = float(os.getenv("PORTFOLIO_SIZE", "100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))

# Per-position loss limits (based on option purchase cost)
# e.g., 10% daily = if you bought $500 of options, max loss today is $50
MAX_LOSS_PCT_PER_DAY = float(os.getenv("MAX_LOSS_PCT_PER_DAY", "0.10"))   # 10%
MAX_LOSS_PCT_TOTAL = float(os.getenv("MAX_LOSS_PCT_TOTAL", "0.30"))       # 30%

# Daily portfolio-wide loss limit
MAX_DAILY_PORTFOLIO_LOSS_PCT = float(os.getenv("MAX_DAILY_PORTFOLIO_LOSS_PCT", "0.025"))

# ============= DTE MULTIPLIERS (4-LEVEL) =============
# Refined DTE adjustments for gamma-aware stop sizing

DTE_30_PLUS_MULTIPLIER = 1.0      # > 30 days: Normal
DTE_14_30_MULTIPLIER = 1.2        # 14-30 days: 20% wider (NEW)
DTE_7_14_MULTIPLIER = 1.5         # 7-14 days: 50% wider
DTE_UNDER_7_MULTIPLIER = 2.0      # < 7 days: 100% wider

# ============= STATISTICS =============

STATS_FILE = os.getenv("STATS_FILE", "logs/asymmetry_stats.json")

