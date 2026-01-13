"""
Real-Time Stock Screener - Main Entry Point
============================================

Implements the screening loop:
1. Connect to IB Gateway
2. Subscribe to watchlist symbols and VIX
3. Every N seconds:
   - Calculate actual vs expected move
   - Compute technical indicators
   - Score opportunities
   - Raise alerts for signals

Usage:
    python main_screener.py
"""

import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from screener.ib_gateway import ScreenerGateway
from screener.formulas import (
    expected_move, abnormality_score, enhanced_score, 
    classify_signal, get_direction
)
from screener.indicators import get_all_indicators
from screener.data_store import DataStore, load_watchlist_with_betas

logger = logging.getLogger(__name__)


# ============= Configuration =============

SCAN_INTERVAL = 5           # Seconds between scans
ABN_THRESHOLD_NORMAL = 1.5  # Abnormality threshold when VIX < 25
ABN_THRESHOLD_HIGH_VIX = 1.2  # Lower threshold when VIX >= 25
VIX_HIGH_LEVEL = 25
MIN_SCORE = 60              # Minimum score to alert
WATCHLIST_PATH = "watchlist.csv"


def screen_symbol(
    gateway: ScreenerGateway,
    data_store: DataStore,
    symbol: str,
    beta: float,
    vix_level: float
) -> dict | None:
    """
    Screen a single symbol for opportunities.
    
    Returns alert dict if signal detected, else None.
    """
    # Get current price
    last_price = gateway.get_latest_price(symbol)
    if not last_price or last_price <= 0:
        return None
    
    # Get previous close
    prev_close = data_store.get_prev_close(symbol)
    if not prev_close or prev_close <= 0:
        # Try to get from gateway
        prev_close = gateway.get_prev_close(symbol)
        if prev_close and prev_close > 0:
            data_store.set_prev_close(symbol, prev_close)
        else:
            return None
    
    # Calculate actual move
    actual_pct = (last_price - prev_close) / prev_close * 100
    direction = get_direction(actual_pct)
    
    # Calculate expected move
    exp_pct, exp_dollars = expected_move(beta, vix_level, last_price)
    if exp_pct == 0:
        return None
    
    # Get technical indicators
    hist_df = gateway.request_historical_bars(symbol, "1 min", "1 D")
    if hist_df is None or len(hist_df) < 26:
        # Not enough data for indicators, use defaults
        indicators = {
            "macd_state": "neutral",
            "rsi": 50.0,
            "bb_pos": "NORMAL",
            "volume_ratio": 1.0
        }
    else:
        indicators = get_all_indicators(hist_df)
    
    # Compute scores
    abn_score = abnormality_score(actual_pct, exp_pct)
    score = enhanced_score(
        actual_pct, exp_pct,
        indicators['volume_ratio'],
        indicators['macd_state'],
        indicators['rsi'],
        indicators['bb_pos'],
        direction
    )
    signal = classify_signal(score)
    
    return {
        "symbol": symbol,
        "price": last_price,
        "prev_close": prev_close,
        "actual_pct": actual_pct,
        "expected_pct": exp_pct,
        "abnormality": abn_score,
        "score": score,
        "signal": signal,
        "direction": direction,
        "volume_ratio": indicators['volume_ratio'],
        "macd_state": indicators['macd_state'],
        "rsi": indicators['rsi'],
        "bb_pos": indicators['bb_pos'],
        "beta": beta,
        "vix": vix_level
    }


def screening_loop(
    gateway: ScreenerGateway,
    data_store: DataStore,
    watchlist: list,
    scan_interval: int = SCAN_INTERVAL
):
    """
    Main screening loop.
    
    Runs continuously, checking each symbol for opportunities.
    """
    logger.info("=" * 60)
    logger.info("SCREENING LOOP STARTED")
    logger.info("=" * 60)
    logger.info(f"Monitoring {len(watchlist)} symbols")
    logger.info(f"Scan interval: {scan_interval} seconds")
    logger.info(f"Min score: {MIN_SCORE}")
    logger.info("=" * 60)
    
    # Track recent alerts for deduplication
    recent_alerts: dict[str, datetime] = {}
    ALERT_COOLDOWN = 300  # 5 minutes
    
    scan_count = 0
    
    while True:
        try:
            scan_count += 1
            vix_level = gateway.get_vix()
            
            # Adjust threshold based on VIX
            threshold = ABN_THRESHOLD_HIGH_VIX if vix_level >= VIX_HIGH_LEVEL else ABN_THRESHOLD_NORMAL
            
            results = []
            
            for item in watchlist:
                symbol = item.symbol
                beta = item.beta
                
                result = screen_symbol(gateway, data_store, symbol, beta, vix_level)
                if not result:
                    continue
                
                # Filter by threshold
                if result['abnormality'] < threshold:
                    continue
                if result['score'] < MIN_SCORE:
                    continue
                
                # Deduplication
                last_alert = recent_alerts.get(symbol)
                if last_alert and (datetime.now() - last_alert).seconds < ALERT_COOLDOWN:
                    continue
                
                results.append(result)
            
            # Sort by score
            results.sort(key=lambda r: r['score'], reverse=True)
            
            # Process alerts
            for result in results:
                # Print alert
                print_alert(result)
                
                # Save to database
                data_store.save_alert(result)
                
                # Track for deduplication
                recent_alerts[result['symbol']] = datetime.now()
            
            # Status update every 60 scans (5 minutes)
            if scan_count % 60 == 0:
                logger.info(
                    f"[Scan #{scan_count}] VIX: {vix_level:.1f} | "
                    f"Threshold: {threshold:.2f} | "
                    f"Alerts today: {len(data_store.get_alerts_today())}"
                )
            
            time.sleep(scan_interval)
            
        except KeyboardInterrupt:
            logger.info("\nStopping screener...")
            break
        except Exception as e:
            logger.error(f"Error in screening loop: {e}")
            time.sleep(scan_interval)


def print_alert(result: dict):
    """Print a formatted alert to console."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Color coding based on signal
    signal = result['signal']
    if signal == "EXCEPTIONAL":
        color = "\033[92m"  # Green
    elif signal == "EXCELLENT":
        color = "\033[93m"  # Yellow
    else:
        color = "\033[0m"   # Default
    
    reset = "\033[0m"
    
    print(f"\n{color}{'='*60}{reset}")
    print(f"{color}*** {signal} SIGNAL ***{reset}")
    print(f"{'='*60}")
    print(f"  Symbol:     {result['symbol']}")
    print(f"  Price:      ${result['price']:.2f}")
    print(f"  Move:       {result['actual_pct']:+.2f}% ({result['direction']})")
    print(f"  Expected:   {result['expected_pct']:.2f}%")
    print(f"  Abnormality: {result['abnormality']:.2f}x")
    print(f"  Score:      {result['score']:.1f}/100")
    print(f"  ---")
    print(f"  Volume:     {result['volume_ratio']:.2f}x avg")
    print(f"  MACD:       {result['macd_state']}")
    print(f"  RSI:        {result['rsi']:.1f}")
    print(f"  BB:         {result['bb_pos']}")
    print(f"  VIX:        {result['vix']:.1f}")
    print(f"  Time:       {timestamp}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('screener.log')
        ]
    )
    
    print("\n" + "=" * 60)
    print("REAL-TIME STOCK SCREENER")
    print("=" * 60)
    print(f"Formula: Expected Move = Beta × VIX / 100")
    print(f"Signal: Abnormality > {ABN_THRESHOLD_NORMAL}x + Technicals")
    print("=" * 60 + "\n")
    
    # Load watchlist
    watchlist = load_watchlist_with_betas(WATCHLIST_PATH)
    if not watchlist:
        logger.error("No symbols in watchlist!")
        return
    
    logger.info(f"Loaded {len(watchlist)} symbols")
    
    # Initialize data store
    data_store = DataStore()
    
    # Connect to IB
    gateway = ScreenerGateway()
    if not gateway.connect_and_start():
        logger.error("Failed to connect to IB Gateway")
        return
    
    # Subscribe to symbols
    logger.info("Subscribing to market data...")
    for item in watchlist:
        gateway.subscribe_stock(item.symbol)
    
    # Subscribe to VIX
    gateway.subscribe_vix()
    
    # Wait for data to flow
    logger.info("Waiting for market data...")
    time.sleep(5)
    
    # Start screening loop
    try:
        screening_loop(gateway, data_store, watchlist)
    finally:
        gateway.disconnect()
        logger.info("Screener stopped.")


if __name__ == "__main__":
    main()
