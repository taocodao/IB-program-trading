# Trailing Stop-Loss Manager - PowerShell Launcher
# Run from project root: .\scripts\run.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Trailing Stop-Loss Manager" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if TWS/Gateway is running
$twsProcess = Get-Process -Name "tws" -ErrorAction SilentlyContinue
$gatewayProcess = Get-Process -Name "ibgateway" -ErrorAction SilentlyContinue

if (-not $twsProcess -and -not $gatewayProcess) {
    Write-Host "[WARNING] TWS or IB Gateway not detected!" -ForegroundColor Yellow
    Write-Host "Please ensure TWS or IB Gateway is running before starting." -ForegroundColor Yellow
    Write-Host ""
}

# Run the application
Write-Host "Starting application..." -ForegroundColor Green
python src\trailing_stop_mgr.py

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
