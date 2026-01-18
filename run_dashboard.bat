@echo off
title Polymarket Volatility Scanner
cd /d "%~dp0"

echo ========================================================
echo.
echo    POLYMARKET VOLATILITY SCANNER
echo.
echo    Starting dashboard...
echo.
echo ========================================================

python src/dashboard/app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error starting the dashboard. Press any key to exit.
    pause >nul
)
