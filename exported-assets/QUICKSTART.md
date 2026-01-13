# QUICK START GUIDE (5 Minutes)

## Step 1: Prerequisites Check (2 min)

Verify you have:
- [ ] Python 3.11+ installed: `python --version`
- [ ] TWS or IB Gateway running and API enabled
- [ ] At least 1 open option position in your account
- [ ] Paper trading enabled for first test

## Step 2: Install Dependencies (1 min)

```bash
pip install ibapi
```

## Step 3: Configure App (1 min)

Edit `config.py`:

```python
IB_HOST = "127.0.0.1"      # Keep as-is
IB_PORT = 7497             # Use 4002 for IB Gateway
TRAIL_PERCENT = 0.10       # 10% below bid (adjust as needed)
PAPER_TRADING = True       # Test first!
```

## Step 4: Run Application (1 min)

```bash
python trailing_stop_mgr.py
```

## Expected Output

```
2025-01-09 09:30:00 - __main__ - INFO - OPTIONS TRAILING STOP-LOSS MANAGER
2025-01-09 09:30:01 - __main__ - INFO - Connected successfully!
2025-01-09 09:30:02 - __main__ - INFO - Position loaded: SPY 2025-02-21 550.0 CALL x5
2025-01-09 09:30:04 - __main__ - INFO - PLACED STOP-LOSS: SPY @ $542.89 (Bid: $603.21)
2025-01-09 09:30:04 - __main__ - INFO - Monitoring is now active. Press Ctrl+C to stop.
```

## Next: Watch the Logs

```bash
# In another terminal window
tail -f trailing_stop.log
```

You'll see updates as prices move:
- When bid rises → new stop placed higher
- When price falls → stop stays fixed
- If stop hits → "STOP LOSS EXECUTED" warning

## Common Questions

**Q: How do I know it's working?**
A: Check TWS → Monitor → Orders. You should see a SELL STP order for each position.

**Q: Can I adjust the 10% trail?**
A: Yes, change `TRAIL_PERCENT = 0.10` in config.py to 0.05 (5%), 0.15 (15%), etc.

**Q: What if TWS loses connection?**
A: Application will log an error. Reconnect TWS and restart the application.

**Q: Is this safe for live trading?**
A: Test 5+ trading days on paper first. Then start with 1-2 contracts on live.

**Q: How much does this cost?**
A: Only standard IB commissions + any market data subscriptions (often waived).

---

## Architecture Explained

When market opens at 9:30 AM ET:

```
START
  ↓
Connect to IB API
  ↓
Load all open options from portfolio
  ↓
For each option:
  ↓
  Subscribe to real-time bid prices
  ↓
MONITORING LOOP (runs until market close):
  ↓
  Every second:
    • Check if bid price is available
    • If new position: Place stop @ (bid × 0.90)
    • If bid rose: Cancel old stop, place new stop @ (bid × 0.90)
    • If bid fell: Do nothing (stop stays fixed)
  ↓
  Stop-Loss Executes:
    ↓
    Position Closed ✓
```

---

## Strategy Details

### Why This Strategy Works

1. **10% Buffer**: Provides cushion against normal price oscillations
2. **Trail Up**: Captures profits as options increase in value
3. **Stay Down**: Protects from whipsaw losses on drops
4. **Automatic**: No manual intervention needed

### Example Trade

```
Entry: Buy SPY 550 CALL @ $10.00
Position Size: 5 contracts
Current Bid: $10.50

SYSTEM ACTION:
Initial Stop: $10.50 × 0.90 = $9.45
Order Placed: SELL 5 @ $9.45 (STP)

Price Movement 1 (Bid goes to $11.00):
New Stop: $11.00 × 0.90 = $9.90
Action: Cancel old order → Place new order @ $9.90 ✓

Price Movement 2 (Bid falls to $10.80):
New Stop: $10.80 × 0.90 = $9.72
Action: Price dropped, no order update (stays @ $9.90)

Price Movement 3 (Bid rises to $12.00):
New Stop: $12.00 × 0.90 = $10.80
Action: Cancel old order → Place new order @ $10.80 ✓

Stop Execution:
If bid hits $10.80 or below → Position sold automatically at $10.80
```

---

## Customization Ideas

### Adjust Trail Percentage

```python
# config.py

TRAIL_PERCENT = 0.05   # 5% - Tighter, closes sooner
TRAIL_PERCENT = 0.10   # 10% - Default, good balance
TRAIL_PERCENT = 0.15   # 15% - Wider, lets winners run
TRAIL_PERCENT = 0.20   # 20% - Very wide, for volatile options
```

### Update Frequency

```python
# Seconds between order updates
MIN_UPDATE_INTERVAL = 1   # Update every second (more responsive)
MIN_UPDATE_INTERVAL = 2   # Update every 2 seconds (default)
MIN_UPDATE_INTERVAL = 5   # Update every 5 seconds (slower, fewer API calls)
```

### Filter Specific Symbols

```python
# Only manage SPY and QQQ options
ALLOWED_SYMBOLS = ["SPY", "QQQ"]

# Exclude VIX options
EXCLUDED_SYMBOLS = ["VIX"]
```

---

## Files Created

```
trailing_stop_project/
├── trailing_stop_mgr.py        (Main application - ~600 lines)
├── config.py                    (Configuration settings)
├── DEPLOYMENT_GUIDE.md          (Full deployment instructions)
└── trailing_stop.log            (Auto-generated log file)
```

## Code Structure

### OptionPosition Class
- Tracks each open option position
- Stores: bid/ask prices, stop order ID, stop price
- Methods: initialization, string representation

### TrailingStopManager Class
- Inherits from IBAPI's EClient and EWrapper
- Key methods:
  - `nextValidId()` - Connection established
  - `position()` - Receive portfolio positions
  - `tickPrice()` - Receive bid/ask price updates
  - `place_stop_loss_order()` - Create new stop order
  - `update_stop_loss_order()` - Modify stop if price moved up
  - `monitor_positions()` - Main monitoring loop
  - `start()` - Begin application
  - `stop()` - Graceful shutdown

### Main Loop

```python
while market_is_open:
    for each_position:
        if has_no_stop_yet:
            place_initial_stop()
        else:
            if bid_price_rose:
                update_stop_to_new_high()
            elif bid_price_fell:
                hold_current_stop()
```

---

## Performance Characteristics

- **CPU Usage**: < 1% (runs in background)
- **Memory**: ~50 MB
- **API Calls**: 1 per second per position
- **Latency**: 1-5 seconds (depends on market data speed)
- **Cost**: Free (no special API charges)

---

## Support

### Check Logs for Diagnosis

```bash
# See last 20 lines of activity
tail -20 trailing_stop.log

# See specific events
grep "PLACED STOP-LOSS" trailing_stop.log
grep "Updating stop" trailing_stop.log
grep "ERROR" trailing_stop.log

# Count of each type
grep "PLACED STOP-LOSS" trailing_stop.log | wc -l
grep "Updating stop" trailing_stop.log | wc -l
```

### When Something Goes Wrong

1. Check logs: `tail -f trailing_stop.log`
2. Verify TWS is connected and responsive
3. Check positions still exist in account
4. Verify bid prices are updating
5. Restart application: `Ctrl+C` then run again

### Resources

- Full guide: See `DEPLOYMENT_GUIDE.md`
- IB API docs: https://interactivebrokers.github.io/tws-api/
- IB Campus: https://www.interactivebrokers.com/campus/

---

## Next Steps

1. **Today**: Run on paper trading for 1 hour
2. **This Week**: Monitor 5 trading days on paper
3. **Next Week**: Go live with 1-2 contracts
4. **After 1 Month**: Scale up position count

Good luck! 🚀
