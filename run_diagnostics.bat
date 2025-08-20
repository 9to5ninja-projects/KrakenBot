@echo off
echo KrakenBot Diagnostic Tool
echo =======================
echo.
echo This tool will run comprehensive diagnostics on your KrakenBot installation
echo and save the results to a log file.
echo.
echo Press any key to start diagnostics...
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
echo Running diagnostics...
echo This may take a few minutes. Please wait.
echo.

python diagnose.py

echo.
echo Diagnostics completed.
echo Please check the log file for results.
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\deactivate.bat
)

pause