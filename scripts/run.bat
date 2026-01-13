@echo off
REM Trailing Stop-Loss Manager - Windows Launcher
REM Run from project root: scripts\run.bat

cd /d "%~dp0.."
echo Starting Trailing Stop-Loss Manager...
echo.
python src\trailing_stop_mgr.py
pause
