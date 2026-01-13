"""
Database Migration: Multi-Tenant Support
=========================================

Adds new tables for multi-user trading platform:
- users (linked to Privy)
- ib_accounts (encrypted credentials)
- user_watchlists
- user_settings

Run: python migrations/001_multi_tenant.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, 
    Boolean, Text, ForeignKey, Enum as SQLEnum, Numeric, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
from datetime import datetime

# Load database URL
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    print("❌ DB_URL not found in environment")
    sys.exit(1)

Base = declarative_base()


# ========== New Tables ==========

class User(Base):
    """Users linked to Privy authentication."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    privy_did = Column(String(255), unique=True, nullable=False)  # Privy user ID
    email = Column(String(255))
    name = Column(String(100))
    wallet_address = Column(String(42))  # Optional embedded wallet
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)


class IBAccount(Base):
    """IB account credentials (encrypted)."""
    __tablename__ = 'ib_accounts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    account_id = Column(String(20))  # IB account number
    credentials_encrypted = Column(Text, nullable=False)  # AES-256 encrypted
    trading_mode = Column(String(10), default='paper')  # paper/live
    gateway_port = Column(Integer)  # Assigned port (4001-4020)
    is_connected = Column(Boolean, default=False)
    last_connected_at = Column(DateTime)


class UserWatchlist(Base):
    """User watchlists."""
    __tablename__ = 'user_watchlists'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    symbol = Column(String(10), nullable=False)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=datetime.now)


class UserSettings(Base):
    """Per-user risk and trading settings."""
    __tablename__ = 'user_settings'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    
    # AI Signal Settings
    min_ai_score = Column(Integer, default=60)  # 60-100
    auto_execute_score = Column(Integer, default=85)  # Auto-trade threshold
    
    # Position Sizing
    max_position_size = Column(Numeric(12, 2), default=10000)
    max_positions = Column(Integer, default=10)
    max_contracts = Column(Integer, default=2)
    
    # Risk Management
    stop_aggression = Column(Numeric(3, 2), default=1.0)  # 0.5-2.0
    risk_tolerance = Column(String(20), default='moderate')  # conservative/moderate/aggressive
    
    # Option Selection (Research-backed defaults)
    target_dte_min = Column(Integer, default=30)
    target_dte_max = Column(Integer, default=45)
    target_delta = Column(Numeric(3, 2), default=0.55)
    
    updated_at = Column(DateTime, default=datetime.now)


def run_migration():
    """Execute the migration."""
    print(f"Connecting to database...")
    engine = create_engine(DB_URL)
    
    print("Creating new tables...")
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Add user_id column to existing tables if they exist
    with engine.connect() as conn:
        # Check if trades table exists and needs user_id
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='trades' AND column_name='user_id'
        """))
        if not result.fetchone():
            print("  Adding user_id to trades table...")
            try:
                conn.execute(text("""
                    ALTER TABLE trades ADD COLUMN user_id UUID REFERENCES users(id)
                """))
                conn.commit()
            except Exception as e:
                print(f"  Note: {e}")
        
        # Check portfolio_snapshots
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='portfolio_snapshots' AND column_name='user_id'
        """))
        if not result.fetchone():
            print("  Adding user_id to portfolio_snapshots table...")
            try:
                conn.execute(text("""
                    ALTER TABLE portfolio_snapshots ADD COLUMN user_id UUID REFERENCES users(id)
                """))
                conn.commit()
            except Exception as e:
                print(f"  Note: {e}")
    
    print("\n✅ Migration complete!")
    print("\nTables created:")
    print("  - users")
    print("  - ib_accounts")
    print("  - user_watchlists")
    print("  - user_settings")
    print("\nModified tables:")
    print("  - trades (added user_id)")
    print("  - portfolio_snapshots (added user_id)")


if __name__ == "__main__":
    run_migration()
