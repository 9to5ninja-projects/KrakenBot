#!/bin/bash
# Script to run the KrakenBot dashboard
# This script activates the virtual environment and starts the Streamlit dashboard

# Exit on error
set -e

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

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed. Please install it with: pip install streamlit"
    exit 1
fi

# Start the dashboard
DASHBOARD_PATH="$PROJECT_ROOT/dashboard.py"
if [ -f "$DASHBOARD_PATH" ]; then
    echo "Starting Streamlit dashboard..."
    streamlit run "$DASHBOARD_PATH"
else
    echo "Error: Dashboard file not found: $DASHBOARD_PATH"
    exit 1
fi