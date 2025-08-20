# KrakenBot Advanced Features

This document describes the advanced features of KrakenBot that enhance its usability, monitoring capabilities, and analysis tools.

## Setup Wizard

The setup wizard provides an interactive way to configure KrakenBot without manually editing the `.env` file.

```bash
python setup_wizard.py
```

The wizard will guide you through configuring:
- API keys (optional)
- Trading parameters
- Monitoring settings
- Trading pairs
- Notification preferences
- Trading mode

## Health Monitoring

The health monitoring system tracks system resources, exchange connectivity, and bot performance.

```bash
# Run a one-time health check
python main.py health

# Enable continuous health monitoring while running the bot
python main.py run --mode monitor --health-monitor
```

The health monitor tracks:
- CPU, memory, and disk usage
- Exchange API latency and status
- Bot performance metrics
- Potential issues and alerts

Health data is saved to `data/health.json` and can be viewed in the dashboard.

## Notifications

KrakenBot can send notifications when profitable arbitrage opportunities are detected.

```bash
# Enable notifications while running the bot
python main.py run --mode monitor --notifications
```

Supported notification methods:
- Console (default)
- Email
- Telegram
- Discord

Configure notification settings in the `.env` file or using the setup wizard.

## Backtesting

The backtesting module allows you to test arbitrage strategies on historical data.

```bash
# Run a backtest with default settings (last 30 days)
python main.py backtest

# Run a backtest with custom settings
python main.py backtest --start-date 2023-01-01 --end-date 2023-01-31 --amount 10000 --threshold 5 --download
```

Backtest results include:
- Total opportunities and profitable opportunities
- Win rate and profit statistics
- Detailed logs of all opportunities
- Visualization of profit over time

Results are saved to `data/backtest_results_*.json` and plots to `data/backtest_plot_*.png`.

## Reporting

The reporting module generates detailed reports on arbitrage opportunities and system performance.

```bash
# Generate a daily report for yesterday
python main.py report --type daily

# Generate a daily report for a specific date
python main.py report --type daily --date 2023-01-01

# Generate a weekly report
python main.py report --type weekly --date 2023-01-07

# Generate a monthly report
python main.py report --type monthly --month 1 --year 2023
```

Reports include:
- Summary statistics
- Path analysis
- Profitable opportunities
- Daily/weekly breakdowns (for weekly/monthly reports)

Reports are saved as both JSON and HTML files in the `data/reports` directory.

## Command Line Interface

KrakenBot has a comprehensive command line interface with multiple commands:

```bash
# Run the bot
python main.py run --mode monitor --health-monitor --notifications

# Run the dashboard
python main.py run --mode dashboard

# Run the setup wizard
python main.py setup

# Run a backtest
python main.py backtest

# Generate a report
python main.py report

# Check system health
python main.py health
```

## Startup Scripts

For convenience, KrakenBot includes startup scripts for Windows and Linux/Mac:

- Windows: `start_krakenbot.bat`
- Linux/Mac: `start_krakenbot.sh`

These scripts handle:
- Checking for Python installation
- Creating and activating a virtual environment
- Installing dependencies
- Running the setup wizard if needed
- Providing a menu of options to run different features