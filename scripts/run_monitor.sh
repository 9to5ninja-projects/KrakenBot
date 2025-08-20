#!/bin/bash
# Script to run the KrakenBot monitor
# This script activates the virtual environment and starts the arbitrage monitor

# Exit on error
set -e

# Default values
MODE="monitor"
AMOUNT=10000
THRESHOLD=5
INTERVAL=10
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --amount)
            AMOUNT="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found: $VENV_DIR"
    echo "Please run install.sh first to set up the environment."
    exit 1
fi

# Start the monitor
MAIN_PATH="$PROJECT_ROOT/main.py"
if [ -f "$MAIN_PATH" ]; then
    echo "Starting KrakenBot in $MODE mode..."
    python "$MAIN_PATH" --mode "$MODE" --amount "$AMOUNT" --threshold "$THRESHOLD" --interval "$INTERVAL" --log-level "$LOG_LEVEL"
else
    echo "Error: Main file not found: $MAIN_PATH"
    exit 1
fi