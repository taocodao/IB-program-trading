@echo off
echo ========================================================
echo  IB Trading System - Cloud Deployment Helper
echo ========================================================
echo.

:: 1. Check if git is initialized
if not exist ".git" (
    echo [ERROR] Git is not initialized.
    echo Please run: git init
    echo And set up your remote: git remote add origin YOUR_REPO_URL
    pause
    exit /b
)

:: 2. Add all files
echo [STEP 1] Adding files...
git add .

:: 3. Commit with timestamp
echo [STEP 2] Committing changes...
set dt=%date% %time%
git commit -m "Deployment Update: %dt%"

:: 4. Push to remote
echo [STEP 3] Pushing to cloud repository...
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Push failed. Trying 'master' branch...
    git push origin master
)

echo.
echo ========================================================
echo  DEPLOYMENT PUSH COMPLETE
echo ========================================================
echo.
echo NEXT STEPS (On Server):
echo 1. SSH into your EC2 instance: ssh user@your-ip
echo 2. Navigate to folder:         cd ib-program-trading
echo 3. Pull latest changes:        git pull
echo 4. Restart services:           docker-compose up -d --build
echo.
pause
