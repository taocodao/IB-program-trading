@echo off
setlocal

:: Configuration
set KEY_FILE=..\tradecoin-bot-key.pem
set DEFAULT_USER=ubuntu
set DEFAULT_IP=34.235.119.67

:: Check if IP provided as argument
if "%~1"=="" (
    if "%DEFAULT_IP%"=="YOUR_EC2_IP" (
        echo [ERROR] No IP address provided.
        echo Usage: ssh_login.bat [IP_ADDRESS]
        echo Or edit this file to set DEFAULT_IP
        echo.
        pause
        exit /b
    ) else (
        set IP=%DEFAULT_IP%
    )
) else (
    set IP=%~1
)

:: Check if key exists
if not exist "%KEY_FILE%" (
    echo [ERROR] Key file not found at: %KEY_FILE%
    pause
    exit /b
)

echo ========================================================
echo  Connecting to %DEFAULT_USER%@%IP%...
echo ========================================================
echo Key: %KEY_FILE%
echo.

:: Fix permissions on key file (common Windows SSH issue)
:: icacls "%KEY_FILE%" /inheritance:r >nul 2>&1
:: icacls "%KEY_FILE%" /grant:r "%USERNAME%":"(R)" >nul 2>&1

:: Connect
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no %DEFAULT_USER%@%IP%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Connection failed.
    echo 1. Check if the IP is correct.
    echo 2. Check if your Security Group allows port 22.
    echo 3. Check if the username '%DEFAULT_USER%' is correct (try 'ec2-user' if not Ubuntu).
    pause
)
