# Privy Setup Guide

## Step 1: Create Privy App

1. Go to [console.privy.io](https://console.privy.io)
2. Sign up / Login
3. Click "Create App"
4. Name: `IB Trading Platform`
5. Copy your **App ID** and **App Secret**

## Step 2: Configure .env

```bash
PRIVY_APP_ID=clxxxxxxxxxxxxxx
PRIVY_APP_SECRET=sk_xxxxxxxxxxxxxxxxxx
```

## Step 3: Configure Privy Dashboard

Enable these authentication methods:
- ✅ Email
- ✅ Google
- ✅ Wallet Connect

### Allowed Origins
- `http://localhost:3000`
- `https://your-domain.com`

## Step 4: Generate Encryption Key

```powershell
cd D:\Projects\IB-program-trading
python src/security/encryption.py
```

Copy key to `.env`:
```
IB_ENCRYPTION_KEY=your_generated_key
```

## Step 5: Start API Server

```powershell
python -m uvicorn src.api_server:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/auth/user | Get current user |
| GET | /api/settings | Get user settings |
| PUT | /api/settings | Update settings |
| GET | /api/watchlist | Get watchlist |
| POST | /api/ib-account | Link IB account |
