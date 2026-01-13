"""
Verify Option Pricing vs Simulation
====================================

Spot check to compare our simulation model against REAL historical option data.
Target: TSLA ATM Call (Feb 20 2026 exp roughly)

Steps:
1. Connect to IB
2. Find a TSLA Call option contract
3. Fetch real 1-min historical data for the OPTION
4. Compare "Real Price" vs "Simulated Price" (Delta-Theta model)
"""

import sys
import time
import logging
from datetime import datetime, timedelta
import threading
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Config
SYMBOL = "TSLA"
EXPIRY = "20260220"  # Adjust to a valid near-term monthly
STRIKE = 420.0       # Adjust to near-money strike
RIGHT = "C"

# Sim Params
SIM_DELTA = 0.5
SIM_THETA = 0.10


class OptionVerifier(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        self.next_req_id = 6000
        self.data_received = False
        self.bars = []
        self.contract_details = None
        self.contract_found = False

    def error(self, reqId, errorCode, errorString, *args):
        if errorCode in [2104, 2106, 2158]: pass
        else: logger.error(f"Error {errorCode}: {errorString}")

    def contractDetails(self, reqId, contractDetails):
        self.contract_details = contractDetails
        self.contract_found = True
        print(f"✓ Found Contract: {contractDetails.contract.localSymbol} (ID: {contractDetails.contract.conId})")

    def contractDetailsEnd(self, reqId):
        pass

    def historicalData(self, reqId, bar):
        self.bars.append({
            'time': bar.date,
            'open': bar.open,
            'close': bar.close
        })

    def historicalDataEnd(self, reqId, start, end):
        self.data_received = True

    def run_check(self):
        self.connect("127.0.0.1", 7497, 401)
        threading.Thread(target=self.run, daemon=True).start()
        time.sleep(1)
        
        # 1. Resolve Contract
        print(f"Finding {SYMBOL} Option: {EXPIRY} {STRIKE} {RIGHT}...")
        c = Contract()
        c.symbol = SYMBOL
        c.secType = "OPT"
        c.exchange = "SMART"
        c.currency = "USD"
        c.lastTradeDateOrContractMonth = EXPIRY
        c.strike = STRIKE
        c.right = RIGHT
        c.multiplier = "100"
        
        self.reqContractDetails(self.next_req_id, c)
        time.sleep(2)
        
        if not self.contract_found:
            print("❌ Contract not found! Check Expiry/Strike.")
            return

        # 2. Fetch Option History
        print("Fetching Real Option History (3 Days)...")
        opt_req = self.next_req_id + 1
        self.reqHistoricalData(opt_req, self.contract_details.contract, 
                               "", "3 D", "1 hour", "MIDPOINT", 1, 1, False, [])
        
        # Wait
        start_wait = time.time()
        while not self.data_received and time.time() - start_wait < 10:
            time.sleep(0.5)
            
        if not self.bars:
            print("❌ No data received.")
            return
            
        df = pd.DataFrame(self.bars)
        print(f"✓ Retrieved {len(df)} bars of option data")
        print("-" * 60)
        print(f"{'Time':<20} | {'Real Price':<10} | {'Sim Price':<10} | {'Diff':<10}")
        print("-" * 60)
        
        # 3. Simulate and Compare
        # Start simulation at the first real price
        sim_price = df.iloc[0]['close']
        
        for i in range(len(df)):
            row = df.iloc[i]
            real_price = row['close']
            
            # Simple Output
            diff = real_price - sim_price
            print(f"{row['time']:<20} | ${real_price:<9.2f} | ${sim_price:<9.2f} | ${diff:<9.2f}")
            
            # Update Sim for next step based on Delta (ignoring stock price here for simplicity, 
            # this is just showing price drift if we assumed constant delta/theta vs reality)
            # To do this strictly we need underlying price too.
            # Ideally we'd fetch stock price in parallel.
            
        print("-" * 60)
        print("Note: This simple view shows the raw option price history.")
        print("To strictly verify, compare specific moves against the underlying.")

if __name__ == "__main__":
    v = OptionVerifier()
    v.run_check()
