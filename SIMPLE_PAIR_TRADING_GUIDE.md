# Simple Pair Trading System Guide

## Overview

We've successfully implemented a comprehensive simple pair trading system for CAD/ETH and CAD/BTC pairs with the following features:

- **Real-time monitoring** with accurate Kraken maker/taker fees
- **Mock investment simulation** for safe testing
- **Enhanced dashboard** with both triangular arbitrage and simple pair trading views
- **Comprehensive data collection** for future analysis
- **Risk management** with position sizing and stop-loss mechanisms

## System Components

### 1. Core Trading Engine (`simple_pair_trader.py`)
- Implements buy-low, sell-high strategy with technical indicators
- Uses accurate Kraken fees (0.25% maker, 0.40% taker)
- Position sizing with risk management
- Real-time portfolio tracking

### 2. Monitoring Scripts
- `run_simple_pair_monitor.py` - Basic monitoring
- `run_full_day_monitor.py` - Comprehensive 24-hour monitoring
- `test_simple_pair_trading.py` - Testing with different modes

### 3. Enhanced Dashboard (`enhanced_dashboard.py`)
- Real-time portfolio visualization
- Trading history analysis
- Performance metrics
- Price charts with trade markers
- Both triangular arbitrage and simple pair trading views

### 4. Data Collection
- All trades stored in JSON format
- Portfolio history tracking
- Price data collection
- Performance analytics
- Comprehensive reporting

## Configuration

The system uses accurate Kraken fees from your `.env` file:
```
MAKER_FEE=0.0025    # 0.25%
TAKER_FEE=0.004     # 0.40%
SLIPPAGE_BUFFER=0.001  # 0.10%
START_AMOUNT=100.0
```

## Trading Strategy Parameters

### Conservative Settings (Default)
- **Buy Threshold**: -1.5% (buy when price drops 1.5% from recent high)
- **Sell Threshold**: +2.0% (sell when price rises 2.0% from recent low)
- **Lookback Period**: 12 intervals (1 hour for 5-minute checks)
- **Max Position Size**: 25% of portfolio per position
- **Min Trade Amount**: $25 CAD

### Risk Management
- Maximum 25% of portfolio per position
- Automatic stop-loss when positions move against you
- Fee-aware profit calculations
- Slippage buffer included

## How to Use

### 1. Run a Quick Test (5 minutes)
```bash
python run_simple_pair_monitor.py --test
```

### 2. Run Full Day Monitoring (24 hours, 5-minute intervals)
```bash
python run_full_day_monitor.py
```

### 3. Run Test with Different Duration
```bash
python run_full_day_monitor.py --duration 2 --interval 120
# Runs for 2 hours with 2-minute intervals
```

### 4. Generate Sample Data for Dashboard Testing
```bash
python generate_sample_data.py
```

### 5. Run Enhanced Dashboard
```bash
streamlit run enhanced_dashboard.py
```
Then open http://localhost:8503 in your browser

### 6. Test with Simulated Volatility
```bash
python test_simple_pair_trading.py --mode simulate
```

## Dashboard Features

### Overview Tab
- Combined view of both strategies
- Portfolio performance over time
- Key metrics summary

### Simple Pair Trading Tab
- Performance summary with returns
- Current positions
- Trading history
- Price charts with buy/sell markers
- Risk metrics

### Triangular Arbitrage Tab
- Opportunity analysis
- Profit distribution
- Time series analysis
- Recent opportunities

### Live Monitoring Tab
- Current market prices
- Live arbitrage checking
- System status

## Data Output

Each monitoring session creates a timestamped directory in `data/` containing:

- `trades.json` - All executed trades
- `portfolio_history.json` - Portfolio value over time
- `performance_summary.json` - Performance metrics
- `price_history.json` - Price data collected
- `monitoring_report.json` - Complete session report
- `comprehensive_analysis.json` - Detailed analysis (full day monitor)
- `README.md` - Session summary

## Key Features

### 1. Accurate Fee Calculation
- Uses real Kraken maker/taker fees
- Includes slippage buffer
- Fee-aware profit calculations

### 2. Risk Management
- Position sizing limits
- Maximum portfolio exposure per pair
- Minimum trade amounts
- Stop-loss mechanisms

### 3. Technical Analysis
- Simple moving average
- Recent high/low analysis
- Configurable lookback periods
- Multiple signal confirmation

### 4. Real-time Monitoring
- Live price feeds from Kraken
- Rate limiting to avoid API limits
- Graceful error handling
- Comprehensive logging

### 5. Data Collection
- All trades recorded
- Portfolio snapshots
- Price history
- Performance analytics

## Performance Metrics

The system tracks:
- **Total Return %** - Overall portfolio performance
- **Realized P&L** - Profit/loss from completed trades
- **Unrealized P&L** - Current position values
- **Win Rate** - Percentage of profitable trades
- **Trade Frequency** - Trades per hour
- **Risk Metrics** - Volatility, max drawdown, Sharpe ratio

## Safety Features

### 1. Mock Trading Mode
- All trades are simulated by default
- No real money at risk
- Full functionality for testing

### 2. Rate Limiting
- Respects Kraken API limits
- Automatic backoff on rate limit hits
- Configurable check intervals

### 3. Error Handling
- Graceful shutdown on interruption
- Data saved on exit
- Comprehensive error logging

## Next Steps

1. **Run Test Session**: Start with a short test to see the system in action
2. **Analyze Results**: Use the dashboard to review performance
3. **Adjust Parameters**: Modify thresholds based on market conditions
4. **Scale Up**: Run longer sessions to collect more data
5. **Live Trading**: When ready, modify for live trading (requires API keys)

## Files Created

- `simple_pair_trader.py` - Core trading engine
- `run_simple_pair_monitor.py` - Basic monitoring
- `run_full_day_monitor.py` - Comprehensive monitoring
- `test_simple_pair_trading.py` - Testing utilities
- `enhanced_dashboard.py` - Streamlit dashboard
- `generate_sample_data.py` - Sample data generator
- `SIMPLE_PAIR_TRADING_GUIDE.md` - This guide

## Example Commands

```bash
# Quick 5-minute test
python run_simple_pair_monitor.py --test

# 1-hour test with 2-minute intervals
python run_full_day_monitor.py --test

# Full 24-hour monitoring
python run_full_day_monitor.py

# Generate sample data and run dashboard
python generate_sample_data.py
streamlit run enhanced_dashboard.py

# Test with simulated market volatility
python test_simple_pair_trading.py --mode simulate
```

The system is now ready for comprehensive testing and data collection! 🚀