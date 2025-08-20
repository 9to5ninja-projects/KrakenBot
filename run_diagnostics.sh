#!/bin/bash

echo "KrakenBot Diagnostic Tool"
echo "======================="
echo
echo "This tool will run comprehensive diagnostics on your KrakenBot installation"
echo "and save the results to a log file."
echo
echo "Press Enter to start diagnostics..."
read

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    echo "Please install Python 3.6 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found, running with system Python."
    PYTHON=python3
fi

echo
echo "Running diagnostics..."
echo "This may take a few minutes. Please wait."
echo

python diagnose.py

echo
echo "Diagnostics completed."
echo "Please check the log file for results."
echo

if [ -f "venv/bin/activate" ]; then
    deactivate
fi

echo "Press Enter to exit..."
read