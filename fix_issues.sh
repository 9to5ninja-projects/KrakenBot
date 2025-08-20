#!/bin/bash

echo "KrakenBot Fix Common Issues Tool"
echo "============================="
echo
echo "This tool will attempt to fix common issues with your KrakenBot installation."
echo
echo "Press Enter to start fixing issues..."
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
echo "Running fixes..."
echo

python fix_common_issues.py

echo
echo "Fix process completed."
echo

if [ -f "venv/bin/activate" ]; then
    deactivate
fi

echo "Press Enter to exit..."
read