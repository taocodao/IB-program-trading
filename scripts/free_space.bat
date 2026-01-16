@echo off
setlocal

:: Configuration
set KEY_FILE=tradecoin-bot-key.pem
set EC2_USER=ubuntu
set EC2_IP=34.235.119.67

echo.
echo ========================================================
echo  Cleaning up Disk Space on EC2
echo ========================================================
echo.

:: Run cleanup commands
:: 1. Prune docker system (images, containers, networks)
:: 2. Clean apt cache
ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "echo 'cleaning docker...'; docker system prune -af; echo 'cleaning apt...'; sudo apt-get clean; echo 'Disk usage after cleanup:'; df -h"

echo.
echo ========================================================
echo  Cleanup Complete
echo ========================================================
echo.
pause
