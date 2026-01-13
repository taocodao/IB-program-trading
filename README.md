# IB Trailing Stop-Loss Manager

Automated trailing stop-loss system for Interactive Brokers options trading.

## Strategy

When the market opens, this system:
1. Loads all your open option positions
2. Places a **STOP LOSS** order at **10% below current bid**
3. As prices move:
   - **Price UP** → Stop moves up with it
   - **Price DOWN** → Stop stays fixed (does not move down)
4. When stop is hit → Position automatically sold

```
Example:
─────────────────────────────────────
Buy SPY 550 CALL, Bid = $10.50
→ Stop placed at $10.50 × 0.90 = $9.45

Bid rises to $12.00
→ Stop updated to $12.00 × 0.90 = $10.80

Bid falls to $11.00  
→ Stop STAYS at $10.80 (no change)

Bid hits $10.80
→ Position SOLD automatically ✓
─────────────────────────────────────
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
# Copy and edit environment file
copy .env.example .env
# Edit .env with your settings
```

### 3. Start TWS/IB Gateway
- Open TWS or IB Gateway
- Enable API: File → Global Config → API → Settings
- Check "Enable ActiveX and Socket Clients"

### 4. Test Connection
```bash
python tests/test_connection.py
```

### 5. Run Application
```bash
python src/trailing_stop_mgr.py
# Or use: scripts\run.bat (Windows)
```

## Project Structure

```
IB-program-trading/
├── src/
│   ├── trailing_stop_mgr.py   # Main application
│   └── config.py              # Configuration
├── tests/
│   ├── test_connection.py     # IB connectivity test
│   └── test_trailing_stop.py  # Unit tests
├── scripts/
│   ├── run.bat                # Windows launcher
│   └── run.ps1                # PowerShell launcher
├── logs/                      # Log files (auto-created)
├── docs/
│   ├── QUICKSTART.md          # 5-minute setup
│   └── DEPLOYMENT_GUIDE.md    # Full deployment guide
├── .env.example               # Environment template
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Configuration

Edit `.env` or `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `IB_PORT` | 7497 | TWS=7497, Gateway=4002 |
| `TRAIL_PERCENT` | 0.10 | 10% below bid |
| `PAPER_TRADING` | True | Always test first! |

## Testing

```bash
# Unit tests (no IB connection needed)
python -m pytest tests/test_trailing_stop.py -v

# Connection test (requires TWS/Gateway running)
python tests/test_connection.py
```

## Safety

⚠️ **Always test with paper trading first!**

1. Set `PAPER_TRADING=True` in `.env`
2. Run for 5+ trading days on paper
3. Start live with 1-2 contracts
4. Monitor actively for first week

## Logs

```bash
# View real-time logs
tail -f logs/trailing_stop.log

# Windows PowerShell
Get-Content logs\trailing_stop.log -Wait
```

## License

MIT
