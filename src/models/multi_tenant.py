"""
Multi-Tenant Database Models
============================

SQLAlchemy models for multi-user trading platform.
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, 
    Text, ForeignKey, Numeric, create_engine
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid
import os

# Load .env file
from dotenv import load_dotenv
load_dotenv()

Base = declarative_base()


class User(Base):
    """Users linked to Privy authentication."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    privy_did = Column(String(255), unique=True, nullable=False)
    email = Column(String(255))
    name = Column(String(100))
    wallet_address = Column(String(42))
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    ib_account = relationship("IBAccount", back_populates="user", uselist=False)
    watchlist = relationship("UserWatchlist", back_populates="user")
    settings = relationship("UserSettings", back_populates="user", uselist=False)
    trades = relationship("Trade", back_populates="user")


class IBAccount(Base):
    """IB account credentials (encrypted)."""
    __tablename__ = 'ib_accounts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), unique=True)
    account_id = Column(String(20))
    credentials_encrypted = Column(Text, nullable=False)
    trading_mode = Column(String(10), default='paper')
    gateway_port = Column(Integer)
    is_connected = Column(Boolean, default=False)
    last_connected_at = Column(DateTime)
    
    user = relationship("User", back_populates="ib_account")


class UserWatchlist(Base):
    """User watchlists."""
    __tablename__ = 'user_watchlists'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    symbol = Column(String(10), nullable=False)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=datetime.now)
    
    user = relationship("User", back_populates="watchlist")


class UserSettings(Base):
    """Per-user risk and trading settings."""
    __tablename__ = 'user_settings'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    
    # AI Signal Settings
    min_ai_score = Column(Integer, default=60)
    auto_execute_score = Column(Integer, default=85)
    
    # Position Sizing
    max_position_size = Column(Numeric(12, 2), default=10000)
    max_positions = Column(Integer, default=10)
    max_contracts = Column(Integer, default=2)
    
    # Risk Management
    stop_aggression = Column(Numeric(3, 2), default=1.0)
    risk_tolerance = Column(String(20), default='moderate')
    
    # Option Selection (Research-backed)
    target_dte_min = Column(Integer, default=30)
    target_dte_max = Column(Integer, default=45)
    target_delta = Column(Numeric(3, 2), default=0.55)
    
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    user = relationship("User", back_populates="settings")


class Trade(Base):
    """Trades linked to users."""
    __tablename__ = 'trades'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    symbol = Column(String)
    environment = Column(String, default="PAPER")
    
    entry_time = Column(DateTime)
    entry_price_opt = Column(Float)
    entry_price_stk = Column(Float)
    contract_details = Column(String)
    quantity = Column(Integer)
    
    exit_time = Column(DateTime, nullable=True)
    exit_price_opt = Column(Float, nullable=True)
    exit_price_stk = Column(Float, nullable=True)
    
    pnl = Column(Float, default=0.0)
    exit_reason = Column(String, nullable=True)
    
    initial_stop = Column(Float)
    final_stop = Column(Float)
    
    user = relationship("User", back_populates="trades")


# ============= Database Manager =============

class MultiTenantDB:
    """Database manager for multi-tenant platform."""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DB_URL")
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    # ===== User Operations =====
    
    def get_or_create_user(self, privy_did: str, email: str = None, name: str = None) -> User:
        """Get existing user or create new one from Privy login."""
        session = self.Session()
        try:
            user = session.query(User).filter_by(privy_did=privy_did).first()
            
            if not user:
                user = User(
                    privy_did=privy_did,
                    email=email,
                    name=name
                )
                session.add(user)
                
                # Create default settings
                settings = UserSettings(user_id=user.id)
                session.add(settings)
                
                session.commit()
                session.refresh(user)
            
            return user
        finally:
            session.close()
    
    def get_user_by_id(self, user_id: str) -> User:
        session = self.Session()
        try:
            return session.query(User).filter_by(id=user_id).first()
        finally:
            session.close()
    
    # ===== Settings Operations =====
    
    def get_user_settings(self, user_id: str) -> dict:
        session = self.Session()
        try:
            settings = session.query(UserSettings).filter_by(user_id=user_id).first()
            if settings:
                return {
                    'min_ai_score': settings.min_ai_score,
                    'auto_execute_score': settings.auto_execute_score,
                    'max_position_size': float(settings.max_position_size),
                    'max_positions': settings.max_positions,
                    'max_contracts': settings.max_contracts,
                    'stop_aggression': float(settings.stop_aggression),
                    'risk_tolerance': settings.risk_tolerance,
                    'target_dte_min': settings.target_dte_min,
                    'target_dte_max': settings.target_dte_max,
                    'target_delta': float(settings.target_delta)
                }
            return None
        finally:
            session.close()
    
    def update_user_settings(self, user_id: str, settings_data: dict):
        session = self.Session()
        try:
            settings = session.query(UserSettings).filter_by(user_id=user_id).first()
            if settings:
                for key, value in settings_data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
                session.commit()
        finally:
            session.close()
    
    # ===== Watchlist Operations =====
    
    def get_user_watchlist(self, user_id: str) -> list:
        session = self.Session()
        try:
            items = session.query(UserWatchlist).filter_by(
                user_id=user_id, 
                is_active=True
            ).all()
            return [item.symbol for item in items]
        finally:
            session.close()
    
    def add_to_watchlist(self, user_id: str, symbol: str):
        session = self.Session()
        try:
            existing = session.query(UserWatchlist).filter_by(
                user_id=user_id, 
                symbol=symbol.upper()
            ).first()
            
            if existing:
                existing.is_active = True
            else:
                item = UserWatchlist(user_id=user_id, symbol=symbol.upper())
                session.add(item)
            
            session.commit()
        finally:
            session.close()
    
    def remove_from_watchlist(self, user_id: str, symbol: str):
        session = self.Session()
        try:
            item = session.query(UserWatchlist).filter_by(
                user_id=user_id, 
                symbol=symbol.upper()
            ).first()
            if item:
                item.is_active = False
                session.commit()
        finally:
            session.close()
    
    # ===== IB Account Operations =====
    
    def save_ib_account(self, user_id: str, account_id: str, encrypted_creds: str, mode: str = 'paper'):
        session = self.Session()
        try:
            account = session.query(IBAccount).filter_by(user_id=user_id).first()
            
            if account:
                account.account_id = account_id
                account.credentials_encrypted = encrypted_creds
                account.trading_mode = mode
            else:
                account = IBAccount(
                    user_id=user_id,
                    account_id=account_id,
                    credentials_encrypted=encrypted_creds,
                    trading_mode=mode
                )
                session.add(account)
            
            session.commit()
        finally:
            session.close()
    
    def get_ib_account(self, user_id: str) -> IBAccount:
        session = self.Session()
        try:
            return session.query(IBAccount).filter_by(user_id=user_id).first()
        finally:
            session.close()
    
    def get_all_active_accounts(self) -> list:
        """Get all IB accounts for gateway management."""
        session = self.Session()
        try:
            accounts = session.query(IBAccount).join(User).filter(
                User.is_active == True
            ).all()
            return accounts
        finally:
            session.close()
    
    def update_gateway_status(self, user_id: str, port: int, connected: bool):
        session = self.Session()
        try:
            account = session.query(IBAccount).filter_by(user_id=user_id).first()
            if account:
                account.gateway_port = port
                account.is_connected = connected
                if connected:
                    account.last_connected_at = datetime.now()
                session.commit()
        finally:
            session.close()
    
    # ===== Aggregation (for Signal Service) =====
    
    def get_all_active_symbols(self) -> set:
        """Get unique set of all symbols across all active user watchlists."""
        session = self.Session()
        try:
            items = session.query(UserWatchlist.symbol).join(User).filter(
                User.is_active == True,
                UserWatchlist.is_active == True
            ).distinct().all()
            return {item.symbol for item in items}
        finally:
            session.close()
    
    def get_users_watching_symbol(self, symbol: str) -> list:
        """Get all user IDs watching a specific symbol."""
        session = self.Session()
        try:
            items = session.query(UserWatchlist.user_id).join(User).filter(
                User.is_active == True,
                UserWatchlist.is_active == True,
                UserWatchlist.symbol == symbol.upper()
            ).all()
            return [str(item.user_id) for item in items]
        finally:
            session.close()
