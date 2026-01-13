# Configuration File for Trailing Stop-Loss Manager
# Save as: config.py

import os
from datetime import time

# ============= IB CONNECTION SETTINGS =============

# TWS/IB Gateway connection
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))  # 7497 for TWS, 4002 for IB Gateway
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "100"))

# Account configuration
ACCOUNT_ID = os.getenv("ACCOUNT_ID", "")  # Leave empty to use default account

# ============= TRADING PARAMETERS =============

# Trail percentage (how far below bid to place stop)
TRAIL_PERCENT = 0.10  # 10% below current bid price

# Minimum time between order updates (seconds)
# Higher value = fewer updates = lower API calls
MIN_UPDATE_INTERVAL = 2  # seconds

# Market hours (EST)
MARKET_OPEN_TIME = time(9, 30)      # 9:30 AM EST
MARKET_CLOSE_TIME = time(16, 0)     # 4:00 PM EST

# ============= LOGGING SETTINGS =============

LOG_FILE = "trailing_stop.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============= SAFETY LIMITS =============

# Maximum positions to monitor (safety check)
MAX_POSITIONS = 100

# Only process options with days to expiration >= this value
MIN_DAYS_TO_EXPIRY = 1

# ============= NOTIFICATION SETTINGS =============

# Email alerts when stop is executed (optional)
ENABLE_EMAIL_ALERTS = False
EMAIL_RECIPIENT = "your-email@example.com"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_PORT = 587

# Telegram alerts (optional)
ENABLE_TELEGRAM_ALERTS = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# ============= PAPER TRADING MODE =============

# Set to True to use paper trading account
PAPER_TRADING = True

# ============= FILTERING =============

# Only process specific symbols (leave empty to process all)
ALLOWED_SYMBOLS = []  # Example: ["AAPL", "MSFT", "SPY"]

# Exclude these symbols
EXCLUDED_SYMBOLS = []  # Example: ["VIX", "RUT"]

# Only process positions with min quantity
MIN_POSITION_QUANTITY = 1

# ============= ADVANCED =============

# Reconnection settings
RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 10  # seconds

# Market data request settings
REQUEST_REAL_TIME_DATA = True
MARKET_DATA_TYPE = 1  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
