@echo off
REM Run unit tests
cd /d "%~dp0.."
echo Running Tests...
echo.
python -m pytest tests/ -v
pause
