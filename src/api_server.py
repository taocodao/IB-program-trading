"""
Multi-Tenant API Server
=======================

FastAPI server with Privy authentication for multi-user trading platform.

Endpoints:
- /api/auth/user - Get current user info
- /api/settings - Get/update user settings
- /api/watchlist - Manage watchlist
- /api/ib-account - Link IB account
"""

import os
import jwt
import httpx
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from dotenv import load_dotenv
load_dotenv()

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.multi_tenant import MultiTenantDB, User
from security.encryption import encrypt_credentials, decrypt_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= Configuration =============

PRIVY_APP_ID = os.getenv("PRIVY_APP_ID", "")
PRIVY_APP_SECRET = os.getenv("PRIVY_APP_SECRET", "")
PRIVY_VERIFICATION_KEY = os.getenv("PRIVY_VERIFICATION_KEY", "")

# Database
db = MultiTenantDB()


# ============= FastAPI App =============

app = FastAPI(
    title="IB Trading Platform API",
    description="Multi-tenant options trading platform",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Pydantic Models =============

class UserResponse(BaseModel):
    id: str
    email: Optional[str]
    name: Optional[str]
    wallet_address: Optional[str]
    created_at: datetime
    has_ib_account: bool


class UserSettingsRequest(BaseModel):
    min_ai_score: Optional[int] = None
    auto_execute_score: Optional[int] = None
    max_position_size: Optional[float] = None
    max_positions: Optional[int] = None
    max_contracts: Optional[int] = None
    stop_aggression: Optional[float] = None
    risk_tolerance: Optional[str] = None
    target_dte_min: Optional[int] = None
    target_dte_max: Optional[int] = None
    target_delta: Optional[float] = None


class WatchlistItem(BaseModel):
    symbol: str


class IBAccountRequest(BaseModel):
    username: str
    password: str
    account_id: Optional[str] = None
    trading_mode: str = "paper"


# ============= Privy Authentication =============

async def verify_privy_token(authorization: str = Header(...)) -> dict:
    """
    Verify Privy JWT access token.
    
    Privy tokens are ES256-signed JWTs containing:
    - sub: Privy DID (user identifier)
    - iss: Privy app issuer
    - aud: App ID
    - exp: Expiration timestamp
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # For development: decode without verification
        # In production: verify with Privy's public key or API
        if PRIVY_VERIFICATION_KEY:
            claims = jwt.decode(
                token,
                PRIVY_VERIFICATION_KEY,
                algorithms=["ES256"],
                audience=PRIVY_APP_ID
            )
        else:
            # Development mode: decode unverified (NOT FOR PRODUCTION)
            claims = jwt.decode(token, options={"verify_signature": False})
        
        return claims
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )


async def get_current_user(claims: dict = Depends(verify_privy_token)) -> User:
    """Get or create user from Privy claims."""
    privy_did = claims.get("sub")
    if not privy_did:
        raise HTTPException(status_code=401, detail="Invalid token claims")
    
    # Extract email if available
    email = claims.get("email")
    
    # Get or create user
    user = db.get_or_create_user(
        privy_did=privy_did,
        email=email
    )
    
    return user


# ============= Health Check =============

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============= Auth Endpoints =============

@app.get("/api/auth/user", response_model=UserResponse)
async def get_user_info(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    ib_account = db.get_ib_account(str(user.id))
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        wallet_address=user.wallet_address,
        created_at=user.created_at,
        has_ib_account=ib_account is not None
    )


# ============= Settings Endpoints =============

@app.get("/api/settings")
async def get_settings(user: User = Depends(get_current_user)):
    """Get user trading settings."""
    settings = db.get_user_settings(str(user.id))
    if not settings:
        raise HTTPException(status_code=404, detail="Settings not found")
    return settings


@app.put("/api/settings")
async def update_settings(
    settings: UserSettingsRequest,
    user: User = Depends(get_current_user)
):
    """Update user trading settings."""
    # Filter out None values
    updates = {k: v for k, v in settings.dict().items() if v is not None}
    
    # Validate ranges
    if 'min_ai_score' in updates:
        if not 50 <= updates['min_ai_score'] <= 100:
            raise HTTPException(400, "min_ai_score must be 50-100")
    
    if 'stop_aggression' in updates:
        if not 0.5 <= updates['stop_aggression'] <= 2.0:
            raise HTTPException(400, "stop_aggression must be 0.5-2.0")
    
    if 'risk_tolerance' in updates:
        if updates['risk_tolerance'] not in ['conservative', 'moderate', 'aggressive']:
            raise HTTPException(400, "risk_tolerance must be conservative/moderate/aggressive")
    
    db.update_user_settings(str(user.id), updates)
    
    return {"status": "updated", "updates": updates}


# ============= Watchlist Endpoints =============

@app.get("/api/watchlist")
async def get_watchlist(user: User = Depends(get_current_user)):
    """Get user's watchlist."""
    symbols = db.get_user_watchlist(str(user.id))
    return {"symbols": symbols, "count": len(symbols)}


@app.post("/api/watchlist")
async def add_to_watchlist(
    item: WatchlistItem,
    user: User = Depends(get_current_user)
):
    """Add symbol to watchlist."""
    symbol = item.symbol.upper().strip()
    if len(symbol) > 10:
        raise HTTPException(400, "Invalid symbol")
    
    db.add_to_watchlist(str(user.id), symbol)
    return {"status": "added", "symbol": symbol}


@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(
    symbol: str,
    user: User = Depends(get_current_user)
):
    """Remove symbol from watchlist."""
    db.remove_from_watchlist(str(user.id), symbol.upper())
    return {"status": "removed", "symbol": symbol.upper()}


# ============= IB Account Endpoints =============

@app.post("/api/ib-account")
async def link_ib_account(
    account: IBAccountRequest,
    user: User = Depends(get_current_user)
):
    """
    Link IB account (one-time credential storage).
    
    Credentials are encrypted with AES-256 before storage.
    """
    # Encrypt credentials
    encrypted = encrypt_credentials(account.username, account.password)
    
    # Save to database
    db.save_ib_account(
        user_id=str(user.id),
        account_id=account.account_id or "",
        encrypted_creds=encrypted,
        mode=account.trading_mode
    )
    
    logger.info(f"IB account linked for user {user.id}")
    
    return {
        "status": "linked",
        "trading_mode": account.trading_mode,
        "message": "IB account linked successfully. Gateway will start automatically."
    }


@app.get("/api/ib-account/status")
async def get_ib_status(user: User = Depends(get_current_user)):
    """Get IB account connection status."""
    account = db.get_ib_account(str(user.id))
    
    if not account:
        return {"status": "not_linked"}
    
    return {
        "status": "linked",
        "trading_mode": account.trading_mode,
        "is_connected": account.is_connected,
        "gateway_port": account.gateway_port,
        "last_connected": account.last_connected_at.isoformat() if account.last_connected_at else None
    }


@app.delete("/api/ib-account")
async def unlink_ib_account(user: User = Depends(get_current_user)):
    """Unlink IB account (deletes encrypted credentials)."""
    # TODO: Stop gateway if running
    session = db.get_session()
    try:
        from models.multi_tenant import IBAccount
        account = session.query(IBAccount).filter_by(user_id=user.id).first()
        if account:
            session.delete(account)
            session.commit()
    finally:
        session.close()
    
    return {"status": "unlinked"}


# ============= Admin Endpoints (Future) =============

@app.get("/api/admin/users")
async def admin_list_users():
    """List all users (admin only - add auth check)."""
    # TODO: Add admin authentication
    session = db.get_session()
    try:
        users = session.query(User).filter_by(is_active=True).all()
        return {
            "count": len(users),
            "users": [
                {
                    "id": str(u.id),
                    "email": u.email,
                    "created_at": u.created_at.isoformat()
                } for u in users
            ]
        }
    finally:
        session.close()


@app.get("/api/admin/symbols")
async def admin_all_symbols():
    """Get all unique symbols across all watchlists."""
    symbols = db.get_all_active_symbols()
    return {"symbols": list(symbols), "count": len(symbols)}


# ============= Run Server =============

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("IB Trading Platform API Server")
    print("=" * 60)
    print(f"Privy App ID: {PRIVY_APP_ID[:10]}..." if PRIVY_APP_ID else "Privy: Not configured")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
