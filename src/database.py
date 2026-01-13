"""
Database Models for Trading System
==================================

Uses SQLAlchemy to handle data persistence.
Supports SQLite (local) and PostgreSQL (AWS RDS).
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import json

Base = declarative_base()

class MarketSignal(Base):
    """Raw Screener Data for ML Training"""
    __tablename__ = 'market_signals'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    symbol = Column(String)
    environment = Column(String, default="PAPER") # PAPER, LIVE, BACKTEST
    price = Column(Float)
    
    # Core Metrics
    beta = Column(Float)
    expected_pct = Column(Float)
    actual_pct = Column(Float)
    abnormality = Column(Float)
    score = Column(Float)
    
    # Indicators (Stored as raw values)
    rsi = Column(Float)
    macd_val = Column(Float)
    volume_ratio = Column(Float)
    bb_position = Column(String)
    
    # Action
    triggered_trade = Column(Boolean, default=False)

class TradeDetails(Base):
    """Executed Trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    environment = Column(String, default="PAPER") # PAPER, LIVE, BACKTEST
    
    # Entry
    entry_time = Column(DateTime)
    entry_price_opt = Column(Float)
    entry_price_stk = Column(Float)
    contract_details = Column(String) # e.g. "20260220 420C"
    quantity = Column(Integer)
    
    # Exit
    exit_time = Column(DateTime, nullable=True)
    exit_price_opt = Column(Float, nullable=True)
    exit_price_stk = Column(Float, nullable=True)
    
    # Result
    pnl = Column(Float, default=0.0)
    exit_reason = Column(String, nullable=True) # "STOP", "PROFIT", "EXPIRY"
    
    # Stop Tracking
    initial_stop = Column(Float)
    final_stop = Column(Float)

class PortfolioSnapshot(Base):
    """Daily/Hourly Portfolio Value"""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    environment = Column(String, default="PAPER") # PAPER, LIVE, BACKTEST
    total_value = Column(Float)
    cash = Column(Float)
    positions_count = Column(Integer)
    unrealized_pnl = Column(Float)

# ==========================================
# Database Manager
# ==========================================

class DBManager:
    def __init__(self, db_url=None):
        # Default to SQLite if no URL provided
        if not db_url:
            db_url = os.getenv("DB_URL", "sqlite:///trading_data.db")
            
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    def log_signal(self, data: dict, env: str = "PAPER") -> int:
        """Save a screener signal."""
        session = self.Session()
        try:
            data['environment'] = env
            sig = MarketSignal(**data)
            session.add(sig)
            session.commit()
            return sig.id
        except Exception as e:
            session.rollback()
            print(f"DB Error (Signal): {e}")
            return -1
        finally:
            session.close()

    def log_trade_entry(self, data: dict, env: str = "PAPER") -> int:
        """Save trade entry."""
        session = self.Session()
        try:
            data['environment'] = env
            trade = TradeDetails(**data)
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            session.rollback()
            print(f"DB Error (Trade): {e}")
            return -1
        finally:
            session.close()

    def update_trade_exit(self, trade_id: int, data: dict):
        """Update trade with exit details."""
        session = self.Session()
        try:
            trade = session.query(TradeDetails).filter_by(id=trade_id).first()
            if trade:
                for k, v in data.items():
                    setattr(trade, k, v)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"DB Error (Exit): {e}")
        finally:
            session.close()
