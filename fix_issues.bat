@echo off
echo KrakenBot Fix Common Issues Tool
echo =============================
echo.
echo This tool will attempt to fix common issues with your KrakenBot installation.
echo.
echo Press any key to start fixing issues...
pause > nul

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.6 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found, running with system Python.
)

echo.
echo Running fixes...
echo.

python fix_common_issues.py

echo.
echo Fix process completed.
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\deactivate.bat
)

pause