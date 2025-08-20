#!/bin/bash
# Installation script for KrakenBot
# This script sets up the virtual environment and installs dependencies

# Exit on error
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create directories if they don't exist
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/logs"
echo "Created data and logs directories"

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Dependencies installed successfully."
else
    echo "Error: Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Create .env file from example if it doesn't exist
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE_FILE="$PROJECT_ROOT/.env.example"

if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE_FILE" ]; then
    echo "Creating .env file from example..."
    cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
    echo ".env file created. Please edit it with your settings."
fi

echo "Installation completed successfully!"
echo "To run the bot, use: python main.py"
echo "For more options, use: python main.py --help"