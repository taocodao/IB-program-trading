"""
Configuration File for Trailing Stop-Loss Manager
==================================================

All settings can be overridden via environment variables.
Copy .env.example to .env and customize for your setup.
"""

import os
from datetime import time
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()

# ============= IB CONNECTION SETTINGS =============

# TWS connection: 127.0.0.1:7497
# IB Gateway connection: 127.0.0.1:4002
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "100"))

# Account configuration (leave empty to use default account)
ACCOUNT_ID = os.getenv("ACCOUNT_ID", "")

# ============= TRADING PARAMETERS =============

# Trail percentage: how far below bid to place stop
# 0.10 = 10% below current bid price
TRAIL_PERCENT = float(os.getenv("TRAIL_PERCENT", "0.10"))

# Minimum time between order updates (seconds)
# Higher value = fewer API calls, slower adjustments
MIN_UPDATE_INTERVAL = int(os.getenv("MIN_UPDATE_INTERVAL", "2"))

# Market hours (EST) - used for scheduling
MARKET_OPEN_TIME = time(9, 30)   # 9:30 AM EST
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM EST

# ============= LOGGING SETTINGS =============

# Log file path (relative to project root or absolute)
LOG_FILE = os.getenv("LOG_FILE", "logs/trailing_stop.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR

# ============= SAFETY LIMITS =============

# Maximum positions to monitor (safety check)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "100"))

# Only process options with days to expiration >= this value
MIN_DAYS_TO_EXPIRY = int(os.getenv("MIN_DAYS_TO_EXPIRY", "1"))

# Minimum position quantity to manage
MIN_POSITION_QUANTITY = int(os.getenv("MIN_POSITION_QUANTITY", "1"))

# ============= FILTERING =============

# Only process specific symbols (comma-separated, empty = all)
# Example: "SPY,QQQ,AAPL"
_allowed = os.getenv("ALLOWED_SYMBOLS", "")
ALLOWED_SYMBOLS = [s.strip() for s in _allowed.split(",") if s.strip()]

# Exclude these symbols (comma-separated)
_excluded = os.getenv("EXCLUDED_SYMBOLS", "")
EXCLUDED_SYMBOLS = [s.strip() for s in _excluded.split(",") if s.strip()]

# ============= PAPER TRADING MODE =============

# Set to True to use paper trading account
# IMPORTANT: Always test with paper trading first!
PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")

# ============= RECONNECTION SETTINGS =============

RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "5"))
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "10"))  # seconds

# ============= MARKET DATA SETTINGS =============

# Market data type: 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
MARKET_DATA_TYPE = int(os.getenv("MARKET_DATA_TYPE", "1"))

# ============= NOTIFICATION SETTINGS (Optional) =============

# Email alerts when stop is executed
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "False").lower() in ("true", "1", "yes")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))

# Telegram alerts
ENABLE_TELEGRAM_ALERTS = os.getenv("ENABLE_TELEGRAM_ALERTS", "False").lower() in ("true", "1", "yes")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
