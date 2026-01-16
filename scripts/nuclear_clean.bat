@echo off
setlocal

:: Configuration
set KEY_FILE=tradecoin-bot-key.pem
set EC2_USER=ubuntu
set EC2_IP=34.235.119.67

echo.
echo ========================================================
echo  NUCLEAR CLEANUP on EC2
echo ========================================================
echo.
echo WARNING: This will stop all containers and remove EVERYTHING not running.
echo.

:: Run cleanup commands
:: 1. Stop all containers
:: 2. Remove all system data (images, containers, volumes)
ssh -i "%KEY_FILE%" %EC2_USER%@%EC2_IP% "echo 'Stopping all...'; docker stop $(docker ps -aq); echo 'Pruning system...'; docker system prune -a --volumes -f; echo 'Disk usage:'; df -h"

echo.
echo ========================================================
echo  Cleanup Complete. Now try deploying again.
echo ========================================================
echo.
pause
