@echo off
echo KrakenBot Triangular Arbitrage System
echo ===================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.6 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if requirements are installed
if not exist "venv\Lib\site-packages\ccxt" (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM Check if .env file exists
if not exist ".env" (
    echo .env file not found.
    echo Running setup wizard...
    python setup_wizard.py
    if %ERRORLEVEL% neq 0 (
        echo Setup failed.
        pause
        exit /b 1
    )
) else (
    REM Ask user what they want to do
    echo What would you like to do?
    echo 1. Run monitor (default)
    echo 2. Run dashboard
    echo 3. Run setup wizard
    echo 4. Run backtest
    echo 5. Generate report
    echo 6. Check system health
    echo.
    
    set /p choice="Enter your choice (1-6): "
    
    if "%choice%"=="" set choice=1
    
    if "%choice%"=="1" (
        echo Starting monitor...
        python main.py run --mode monitor --health-monitor --notifications
    ) else if "%choice%"=="2" (
        echo Starting dashboard...
        python main.py run --mode dashboard
    ) else if "%choice%"=="3" (
        echo Running setup wizard...
        python setup_wizard.py
    ) else if "%choice%"=="4" (
        echo Running backtest...
        python main.py backtest
    ) else if "%choice%"=="5" (
        echo Generating report...
        python main.py report
    ) else if "%choice%"=="6" (
        echo Checking system health...
        python main.py health
    ) else (
        echo Invalid choice.
        echo Starting monitor (default)...
        python main.py run --mode monitor --health-monitor --notifications
    )
)

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo KrakenBot session ended.
pause