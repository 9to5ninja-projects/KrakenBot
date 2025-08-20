#!/bin/bash

echo "KrakenBot Triangular Arbitrage System"
echo "==================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    echo "Please install Python 3.6 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if [ ! -d "venv/lib/python*/site-packages/ccxt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ".env file not found."
    echo "Running setup wizard..."
    python setup_wizard.py
    if [ $? -ne 0 ]; then
        echo "Setup failed."
        exit 1
    fi
else
    # Ask user what they want to do
    echo "What would you like to do?"
    echo "1. Run monitor (default)"
    echo "2. Run dashboard"
    echo "3. Run setup wizard"
    echo "4. Run backtest"
    echo "5. Generate report"
    echo "6. Check system health"
    echo
    
    read -p "Enter your choice (1-6): " choice
    
    if [ -z "$choice" ]; then
        choice=1
    fi
    
    case $choice in
        1)
            echo "Starting monitor..."
            python main.py run --mode monitor --health-monitor --notifications
            ;;
        2)
            echo "Starting dashboard..."
            python main.py run --mode dashboard
            ;;
        3)
            echo "Running setup wizard..."
            python setup_wizard.py
            ;;
        4)
            echo "Running backtest..."
            python main.py backtest
            ;;
        5)
            echo "Generating report..."
            python main.py report
            ;;
        6)
            echo "Checking system health..."
            python main.py health
            ;;
        *)
            echo "Invalid choice."
            echo "Starting monitor (default)..."
            python main.py run --mode monitor --health-monitor --notifications
            ;;
    esac
fi

# Deactivate virtual environment
deactivate

echo
echo "KrakenBot session ended."