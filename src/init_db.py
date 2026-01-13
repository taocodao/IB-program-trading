"""
Initialize Database Tables
==========================
Run this once to create the necessary tables in the database.
"""

import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Now import database module which will create tables
from database import DBManager

db_url = os.getenv("DB_URL")
print(f"Connecting to: {db_url[:50]}...")

db = DBManager(db_url)
print("✅ Tables created successfully!")
print("   - market_signals")
print("   - trades")
print("   - portfolio_snapshots")
