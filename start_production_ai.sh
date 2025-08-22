#!/bin/bash

echo "========================================"
echo "    PRODUCTION AI TRADER STARTUP"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if AI models exist
if [ ! -f "data/emergency_training/trained_models/rf_classifier.pkl" ]; then
    echo "ERROR: AI models not found!"
    echo "Please train models first by running emergency_ai_training.py"
    exit 1
fi

# Show current configuration
echo
echo "Current Configuration:"
if [ -f "production_config.json" ]; then
    cat production_config.json
else
    echo "No config found, will create default configuration."
fi

echo
echo "========================================"
echo "Starting Production AI Trading System"
echo "========================================"
echo
echo "This will run for up to 72 hours with:"
echo "- Conservative risk management"
echo "- Automatic safety stops"
echo "- Continuous data logging"
echo "- Health monitoring"
echo
echo "Press Ctrl+C to stop gracefully"
echo

# Start the production trader
python production_ai_trader.py

echo
echo "========================================"
echo "Production AI Trading Session Ended"
echo "========================================"