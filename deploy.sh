#!/bin/bash
# ============================================================
# IB Trading System - AWS EC2 Deployment Script
# ============================================================
# Run this script on a fresh Ubuntu 22.04 EC2 instance
# Usage: chmod +x deploy.sh && ./deploy.sh
# ============================================================

set -e  # Exit on any error

echo "=============================================="
echo "IB Trading System - EC2 Deployment"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============= Step 1: Update System =============
print_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ============= Step 2: Install Docker =============
print_step "Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt install -y docker.io docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# ============= Step 3: Install Nginx =============
print_step "Installing Nginx..."
if ! command -v nginx &> /dev/null; then
    sudo apt install -y nginx
    sudo systemctl enable nginx
    echo "Nginx installed successfully"
else
    echo "Nginx already installed"
fi

# ============= Step 4: Install Certbot for SSL =============
print_step "Installing Certbot for SSL..."
sudo apt install -y certbot python3-certbot-nginx

# ============= Step 5: Setup Application Directory =============
print_step "Setting up application directory..."
APP_DIR="/opt/ib-trading"
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
fi

# ============= Step 6: Copy Nginx Config =============
print_step "Configuring Nginx..."
sudo cp nginx/trading.conf /etc/nginx/sites-available/trading
sudo ln -sf /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ============= Step 7: Create .env if not exists =============
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Please create it with your credentials."
    cat > .env.template << 'EOF'
# Database Connection (already configured)
DB_URL=postgresql://your-user:your-password@your-rds-endpoint:5432/ib_trading

# IB Gateway Credentials
TWS_USERID=your_ib_username
TWS_PASSWORD=your_ib_password

# Trading Mode
TRADING_MODE=paper

# Privy Authentication
PRIVY_APP_ID=your_privy_app_id
PRIVY_APP_SECRET=your_privy_app_secret
PRIVY_VERIFICATION_KEY=

# IB Credential Encryption Key
IB_ENCRYPTION_KEY=

# Redis
REDIS_URL=redis://redis:6379
EOF
    echo "Created .env.template - copy to .env and fill in values"
fi

# ============= Step 8: Start Services =============
print_step "Starting Docker services..."
docker-compose up -d

# ============= Step 9: Check Status =============
print_step "Checking service status..."
docker-compose ps

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your credentials"
echo "2. Restart services: docker-compose restart"
echo "3. For SSL, run: sudo certbot --nginx -d yourdomain.com"
echo ""
echo "Access your dashboard at: http://$(curl -s ifconfig.me)"
echo "=============================================="
