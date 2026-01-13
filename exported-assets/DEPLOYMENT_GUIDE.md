# DEPLOYMENT GUIDE: Options Trailing Stop-Loss Manager

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing & Validation](#testing--validation)
6. [Deployment Options](#deployment-options)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Safety Best Practices](#safety-best-practices)

---

## Prerequisites

### System Requirements
- Python 3.11 or higher
- Interactive Brokers account (live or paper trading)
- TWS (Trader Workstation) or IB Gateway installed and running
- Minimum 500 MB disk space
- Stable internet connection

### IB Account Setup
1. Ensure your account is **fully open and funded** ($500+ minimum balance)
2. Enable **API access** in TWS:
   - Open TWS → Settings (Ctrl+,)
   - Navigate to Trading → General
   - Check "Enable ActiveX and Socket Clients"
   - Click Apply and restart TWS
3. For IB Gateway, API is enabled by default
4. Confirm you have **market data subscriptions** for options
   - Most data subscriptions waived if you generate $30+/month commissions

### Software Dependencies
```bash
# All required packages
ibapi>=10.26          # Interactive Brokers API
python>=3.11          # Python version
```

---

## Installation Steps

### Step 1: Install Python Dependencies

```bash
# Using pip
pip install ibapi>=10.26

# Or upgrade existing installation
pip install --upgrade ibapi
```

**Troubleshooting:**
- If `pip` not found, ensure Python is in PATH
- On Linux/Mac: Use `pip3` instead of `pip`
- On Windows: May need to run PowerShell as Administrator

### Step 2: Download Application Files

Create a project directory and download these files:

```
trailing_stop_project/
├── trailing_stop_mgr.py        # Main application
├── config.py                    # Configuration settings
├── README.md                    # Documentation
└── logs/                        # (created automatically)
    └── trailing_stop.log        # Log output
```

### Step 3: Verify IB API Installation

Test the API connection:

```python
# test_connection.py
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
    
    def nextValidId(self, orderId):
        print(f"✓ Connected! Next Order ID: {orderId}")
        self.disconnect()

app = TestApp()
app.connect("127.0.0.1", 7497, 1)
app.run()
```

Run this test:
```bash
python test_connection.py
```

Expected output:
```
✓ Connected! Next Order ID: 1234
```

---

## Configuration

### Edit config.py

Open `config.py` and configure for your setup:

```python
# CONNECTION SETTINGS
IB_HOST = "127.0.0.1"        # Usually localhost
IB_PORT = 7497               # 7497 for TWS, 4002 for IB Gateway
IB_CLIENT_ID = 100           # Unique client ID

# TRADING PARAMETERS
TRAIL_PERCENT = 0.10         # 10% below bid (adjust as needed)
MIN_UPDATE_INTERVAL = 2      # Seconds between order updates

# PAPER VS LIVE TRADING
PAPER_TRADING = True         # Start with paper trading!

# SAFETY LIMITS
MAX_POSITIONS = 100
MIN_POSITION_QUANTITY = 1

# LOGGING
LOG_LEVEL = "INFO"           # DEBUG for verbose output
```

### Key Configuration Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TRAIL_PERCENT` | 0.10 (10%) | Higher % = wider stop loss, more protection |
| `MIN_UPDATE_INTERVAL` | 2 sec | Higher = fewer API calls, slower adjustments |
| `PAPER_TRADING` | True | Always start here for testing |
| `LOG_LEVEL` | INFO | Use DEBUG for troubleshooting |

---

## Running the Application

### Option 1: Direct Execution (Recommended for Testing)

```bash
# Ensure TWS/IB Gateway is running first!
python trailing_stop_mgr.py
```

Expected console output:
```
2025-01-09 09:30:15 - __main__ - INFO - ======================================================
2025-01-09 09:30:15 - __main__ - INFO - OPTIONS TRAILING STOP-LOSS MANAGER
2025-01-09 09:30:15 - __main__ - INFO - ======================================================
2025-01-09 09:30:15 - __main__ - INFO - Connecting to 127.0.0.1:7497 (client ID: 100)...
2025-01-09 09:30:16 - __main__ - INFO - Connected successfully!
2025-01-09 09:30:16 - __main__ - INFO - Requesting all positions...
2025-01-09 09:30:18 - __main__ - INFO - Position loaded: AAPL 2025-01-17 150.0 CALL x10 @ $3.50
2025-01-09 09:30:18 - __main__ - INFO - Starting position monitoring...
2025-01-09 09:30:18 - __main__ - INFO - Monitoring is now active. Press Ctrl+C to stop.
```

### Option 2: Run in Background (Linux/Mac)

```bash
# Run with output to log file
python trailing_stop_mgr.py > output.log 2>&1 &

# Check if running
ps aux | grep trailing_stop_mgr.py

# Stop the process
pkill -f trailing_stop_mgr.py
```

### Option 3: Use systemd Service (Linux)

Create `/etc/systemd/system/trailing-stop.service`:

```ini
[Unit]
Description=Options Trailing Stop-Loss Manager
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trailing_stop_project
ExecStart=/usr/bin/python3 /home/ubuntu/trailing_stop_project/trailing_stop_mgr.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trailing-stop
sudo systemctl start trailing-stop
sudo systemctl status trailing-stop
```

---

## Testing & Validation

### Step 1: Paper Trading Test (Recommended First)

```python
# In config.py
PAPER_TRADING = True
```

1. Start application during market hours
2. Open your paper trading account in TWS
3. Manually buy a small options contract (e.g., 1 SPY call)
4. Watch logs for stop-loss placement
5. Verify stop order appears in TWS under "Orders"

### Step 2: Verify Stop Placement

Expected log sequence:
```
2025-01-09 09:31:00 - __main__ - INFO - PLACED STOP-LOSS: SPY @ $215.50 (Bid: $239.45)
```

Check in TWS:
- Open "Monitor" tab → "Orders"
- Should show SELL order type "STP" (Stop)
- Verify quantity and strike match your position

### Step 3: Test Trail-Up Functionality

During market hours with live option prices:
1. Watch for bid price movements in logs
2. When bid rises significantly, logs should show:
```
2025-01-09 10:15:30 - __main__ - INFO - SPY: Price moved up $2.15. 
    Updating stop: $215.50 -> $217.65
2025-01-09 10:15:30 - __main__ - INFO - PLACED STOP-LOSS: SPY @ $217.65 (Bid: $242.95)
```
3. Verify old stop order is cancelled and new one placed in TWS

### Step 4: Simulate Stop Execution

On paper trading account:
1. Manually adjust market to hit your stop price
2. Watch for log:
```
2025-01-09 10:30:00 - __main__ - WARNING - *** STOP LOSS EXECUTED *** Order 1001 filled at $215.47
2025-01-09 10:30:00 - __main__ - WARNING - Position closed: SPY 2025-01-17 240.0 CALL x1
```

---

## Deployment Options

### Cloud Deployment (Recommended for 24/5 Operation)

#### Option A: AWS EC2 (Linux)

```bash
# 1. Launch t2.micro instance (Free Tier eligible)
# 2. SSH into instance
# 3. Install Python and dependencies

sudo apt update
sudo apt install python3-pip python3-dev -y
pip3 install ibapi

# 4. Upload your files
scp -i key.pem trailing_stop_mgr.py ec2-user@instance:/home/ec2-user/
scp -i key.pem config.py ec2-user@instance:/home/ec2-user/

# 5. Install and start systemd service (see above)
```

#### Option B: DigitalOcean Droplet

```bash
# Similar to AWS, but simpler setup
# 1. Create $6/month Ubuntu 22.04 droplet
# 2. SSH in and follow AWS setup steps
# 3. Use systemd service for auto-restart
```

#### Option C: Raspberry Pi (Home Server)

```bash
# Low-cost 24/5 operation
# 1. Install Raspberry Pi OS
# 2. Follow Linux installation steps
# 3. Run as systemd service
# 4. Ensure stable power supply + uninterruptible power supply (UPS)
```

### Local Deployment (Development/Testing)

Simply run:
```bash
python trailing_stop_mgr.py
```

Keep terminal open or run in background.

---

## Monitoring & Troubleshooting

### View Real-time Logs

```bash
# Follow logs in real-time
tail -f trailing_stop.log

# See last 50 lines
tail -50 trailing_stop.log

# Search for errors
grep ERROR trailing_stop.log

# Count placed stops
grep "PLACED STOP-LOSS" trailing_stop.log | wc -l
```

### Common Issues & Solutions

#### Issue 1: "Connection refused"
```
Error 1: 10061 Can't connect to TWS/IB Gateway
```

**Solution:**
- Ensure TWS or IB Gateway is running
- Check IB_HOST and IB_PORT in config.py
- Verify API is enabled in TWS settings
- Try restarting TWS

#### Issue 2: "No positions found"
```
INFO - Position loading complete. Found 0 option positions
```

**Solution:**
- Verify you have open option positions in your account
- Check positions are in the paper/live account you're connected to
- Ensure positions are Options (OPT), not stocks
- Check ALLOWED_SYMBOLS filter if set

#### Issue 3: "Stop order not placed"
```
WARNING - No bid price yet for SPY
```

**Solution:**
- Wait for market data to arrive (first 2-3 seconds)
- Verify market data subscription is active
- Check that bid price is receiving ticks in logs
- Enable DEBUG logging: LOG_LEVEL = "DEBUG"

#### Issue 4: "No orders updating"
```
INFO - Monitoring check #100: 1 positions tracked, 0 stops active
```

**Solution:**
- Increase MIN_UPDATE_INTERVAL throttle
- Check logs for bid price updates
- Verify positions have current_bid values
- Review EWrapper callbacks are firing

### Enable Debug Logging

In `config.py`:
```python
LOG_LEVEL = "DEBUG"
```

This shows every API callback and decision.

---

## Safety Best Practices

### Before Going Live

✅ **DO:**
- [ ] Test with paper trading first (minimum 5 trading days)
- [ ] Test with 1-2 contracts before scaling up
- [ ] Monitor the first hour of trading daily
- [ ] Keep TWS/IB Gateway running on stable hardware
- [ ] Have backup connection method (mobile app)
- [ ] Keep API logs backed up
- [ ] Review all position closures weekly

❌ **DON'T:**
- [ ] Don't start with live trading
- [ ] Don't run on unstable internet
- [ ] Don't use a single server without redundancy
- [ ] Don't ignore API errors and warnings
- [ ] Don't modify code while trading
- [ ] Don't set trail percent too tight (risk whipsaw)

### Risk Management

1. **Position Size Limits**
   ```python
   # In config.py
   MAX_POSITIONS = 100      # Don't monitor more than this
   MIN_POSITION_QUANTITY = 1
   ```

2. **Stop-Loss Validation**
   - Always verify stop price = bid * (1 - 0.10)
   - Check logs for "PLACED STOP-LOSS" with correct price
   - Confirm order appears in TWS within 1 second

3. **Manual Override**
   - You can always cancel any order in TWS manually
   - Application respects manual order cancellations
   - Reconnect will restore monitoring after interruption

4. **Daily Checklist**
   ```bash
   # Start of day
   1. Verify TWS/IB Gateway running
   2. Check account has sufficient balance
   3. Verify market data subscriptions active
   4. Start application
   5. Monitor first 30 minutes
   
   # End of day
   1. Review log file for any errors
   2. Confirm all open stops are active
   3. Backup log file
   ```

### Emergency Stop

If anything goes wrong:

```bash
# 1. Kill the application
Ctrl+C  # If running in terminal
pkill -f trailing_stop_mgr.py  # If in background

# 2. Cancel all pending orders in TWS manually
# 3. Review logs to understand what happened
# 4. Fix any issues in config.py or code
# 5. Restart application

# Or emergency order cancellation from Python:
python -c "from ibapi.client import EClient; \
           app = EClient(None); app.connect('127.0.0.1', 7497, 1); \
           [app.cancelOrder(i) for i in range(1000, 2000)]"
```

---

## Performance Monitoring

### Key Metrics to Track

```bash
# Orders placed per day
grep "PLACED STOP-LOSS" trailing_stop.log | wc -l

# Orders updated
grep "Updating stop" trailing_stop.log | wc -l

# Stops executed
grep "STOP LOSS EXECUTED" trailing_stop.log

# Average response time
grep "CHECK" trailing_stop.log | tail -20

# API errors
grep "ERROR" trailing_stop.log
```

### System Resources

Monitor on deployment server:
```bash
# CPU & Memory usage
top

# Disk space
df -h

# Network connections
netstat -an | grep 7497

# Process info
ps aux | grep trailing_stop
```

---

## Support & Troubleshooting Resources

- **IBKR Campus**: https://www.interactivebrokers.com/campus/
- **API Documentation**: https://interactivebrokers.github.io/tws-api/
- **GitHub Examples**: https://github.com/InteractiveBrokers/ibapi
- **Community**: /r/algotrading on Reddit
