@echo off
echo ========================================
echo    KrakenBot Live Trading Operation
echo ========================================
echo.
echo This will start:
echo - Live trading session with real data
echo - Dashboard on http://localhost:8505
echo - Real-time session monitoring
echo - Comprehensive data collection
echo.
echo Press any key to start or Ctrl+C to cancel...
pause >nul

echo.
echo Starting live trading operation...
echo.

REM Start the live trading session
start "KrakenBot Live Trading" cmd /k "cd /d %~dp0 && python start_live_trading_session.py"

REM Wait a moment for the session to initialize
timeout /t 10 /nobreak >nul

REM Start the session monitor
start "Session Monitor" cmd /k "cd /d %~dp0 && python session_monitor.py"

echo.
echo ========================================
echo  Live Trading Operation Started!
echo ========================================
echo.
echo Dashboard: http://localhost:8505
echo.
echo Two windows opened:
echo 1. Live Trading Session - Main trading loop
echo 2. Session Monitor - Real-time analysis
echo.
echo Press any key to exit this window...
pause >nul