@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: IB Trading System - Full Deployment Script
:: ============================================================
:: This script handles the entire deployment process:
:: 1. Commits and pushes code to Git (local)
:: 2. SSHs into EC2 and pulls/restarts services (remote)
:: ============================================================

:: ===== CONFIGURATION (EDIT THESE) =====
set KEY_FILE=tradecoin-bot-key.pem
set EC2_USER=ubuntu
set EC2_IP=34.235.119.67
set REMOTE_DIR=/home/ubuntu/ib-program-trading
set REPO_URL=https://github.com/taocodao/IB-program-trading.git
:: ======================================

echo.
echo ========================================================
echo  IB Trading System - Full Deployment
echo ========================================================
echo.

:: Check for user-provided IP
if not "%~1"=="" (
    set EC2_IP=%~1
)

if "!EC2_IP!"=="YOUR_EC2_IP" (
    echo [ERROR] EC2 IP not configured.
    echo Please edit this script and set EC2_IP, or provide it as argument:
    echo     deploy_full.bat 1.2.3.4
    pause
    exit /b 1
)

:: ===== STEP 1: Git Commit and Push =====
echo [STEP 1/4] Committing and pushing code to Git...
echo.

git add .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git add failed.
    pause
    exit /b 1
)

set timestamp=%date% %time%
git commit -m "Deploy: %timestamp%"
:: Commit may "fail" if nothing to commit - that's OK

git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Push to 'main' failed, trying 'master'...
    git push origin master
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Git push failed. Check your remote configuration.
        pause
        exit /b 1
    )
)
echo [OK] Code pushed to remote repository.
echo.

:: ===== STEP 2: SSH and Pull =====
echo [STEP 2/4] Connecting to EC2 and pulling latest code...
echo.

ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no %EC2_USER%@%EC2_IP% "cd %REMOTE_DIR% && git pull origin main || git pull origin master"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Pull failed. Server may need initial clone.
    echo Attempting initial clone...
    ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "git clone %REPO_URL% %REMOTE_DIR% || echo 'Clone skipped - directory may exist'"
)
echo [OK] Code updated on server.
echo.

:: ===== STEP 3: Docker Restart =====
echo [STEP 3/4] Restarting Docker services on EC2...
echo.

ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "cd %REMOTE_DIR% && docker-compose -f docker-compose.prod.yml up -d --build"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker restart failed. Check server logs.
    pause
    exit /b 1
)
echo [OK] Docker services restarted.
echo.

:: ===== STEP 4: Verify =====
echo [STEP 4/4] Verifying deployment...
echo.

ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "cd %REMOTE_DIR% && docker-compose -f docker-compose.prod.yml ps"

echo.
echo ========================================================
echo  DEPLOYMENT COMPLETE!
echo ========================================================
echo.
echo Server: %EC2_USER%@%EC2_IP%
echo.
echo To view logs:
echo     ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "cd %REMOTE_DIR% && docker-compose logs -f trading-system"
echo.
echo To SSH into server:
echo     scripts\ssh_login.bat %EC2_IP%
echo.
pause
