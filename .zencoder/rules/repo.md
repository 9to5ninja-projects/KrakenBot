---
description: Repository Information Overview
alwaysApply: true
---

# KrakenBot Information

## Summary
KrakenBot is a Python-based triangular arbitrage monitoring system for cryptocurrency trading on the Kraken exchange. It focuses on detecting profitable trading opportunities between BTC/CAD, ETH/CAD, and ETH/BTC pairs, with comprehensive monitoring, analysis, and reporting capabilities.

## Structure
- **main.py**: Entry point with command-line interface
- **config.py**: Configuration management using environment variables
- **exchange.py**: Kraken exchange API wrapper using ccxt
- **arbitrage.py**: Triangular arbitrage calculation engine
- **monitor.py**: Real-time monitoring system
- **health_monitor.py**: System health monitoring
- **notifications.py**: Notification system for profitable opportunities
- **reports.py**: Automated reporting module
- **backtest.py**: Backtesting engine for strategy evaluation
- **setup_wizard.py**: Interactive configuration wizard
- **dashboard.py**: Streamlit-based visualization dashboard
- **logger.py**: Logging and data storage functionality
- **trader.py**: Trading execution module (placeholder for future)
- **data/**: Data storage directory for opportunities and reports
- **logs/**: Log files directory
- **docs/**: Documentation including troubleshooting guides
- **.env.example**: Example environment configuration
- **requirements.txt**: Python dependencies

## Language & Runtime
**Language**: Python
**Version**: Python 3.6+
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- ccxt: Cryptocurrency exchange trading library
- python-dotenv: Environment variable management
- pandas: Data analysis and manipulation
- numpy: Numerical computing
- matplotlib: Data visualization
- loguru: Enhanced logging
- psutil: System monitoring
- requests: HTTP requests for notifications

**Development Dependencies**:
- streamlit: Web-based dashboard (optional)
- pytest: Testing framework
- pytest-mock: Mocking for tests

## Build & Installation
```bash
# Automated setup (recommended)
# Windows: run start_krakenbot.bat
# Linux/Mac: run ./start_krakenbot.sh

# Manual setup
python -m venv venv
# Activate virtual environment (Windows)
venv\Scripts\activate
# Or on Linux/Mac
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run setup wizard
python setup_wizard.py
```

## Usage
```bash
# Using command-line interface
python main.py run --mode monitor --health-monitor --notifications

# Monitor mode (default)
python main.py run --mode monitor --amount 10000 --threshold 5 --interval 10

# Dashboard mode
python main.py run --mode dashboard

# Simulation mode
python main.py run --mode simulate --amount 10000 --threshold 5

# Backtest mode
python main.py backtest --start-date 2023-01-01 --end-date 2023-01-31

# Generate reports
python main.py report --type daily
python main.py report --type weekly
python main.py report --type monthly --month 1 --year 2023

# Check system health
python main.py health
```

## Diagnostic Tools
The system includes comprehensive diagnostic and troubleshooting tools:
- **run_diagnostics.bat/sh**: Run comprehensive system diagnostics
- **fix_issues.bat/sh**: Automatically fix common issues
- **diagnose.py**: Detailed diagnostic script that logs to file
- **fix_common_issues.py**: Script to repair common configuration problems
- **docs/TROUBLESHOOTING.md**: Detailed troubleshooting guide

## Configuration
The system is configured through environment variables or .env file:
- Trading parameters: START_AMOUNT, PROFIT_THRESHOLD, MAKER_FEE, TAKER_FEE
- API credentials: KRAKEN_API_KEY, KRAKEN_SECRET (optional for monitoring)
- Trading pairs: PAIR_1, PAIR_2, PAIR_3
- Monitoring settings: CHECK_INTERVAL, LOG_LEVEL
- Notification settings: ENABLE_NOTIFICATIONS, NOTIFICATION_METHOD
- Trading mode: TRADING_MODE