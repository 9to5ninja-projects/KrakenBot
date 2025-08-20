# KrakenBot Live Trading Operation Guide

## Overview

This guide covers running KrakenBot as a live trading operation with comprehensive monitoring, analysis, and data collection - simulating real trading that started today.

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
```bash
# Windows PowerShell (Recommended)
.\start_live_operation.ps1

# Windows Command Prompt
start_live_operation.bat
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start live trading session
python start_live_trading_session.py

# Terminal 2: Start session monitor
python session_monitor.py

# Terminal 3: Dashboard (optional - auto-started)
streamlit run simple_dashboard.py --server.port 8505
```

## 📊 What Gets Started

### 1. Live Trading Session
- **File**: `start_live_trading_session.py`
- **Function**: Main trading loop with 5-minute intervals
- **Features**:
  - Real market data collection from Kraken
  - Mock trading execution with accurate fees
  - Automatic data saving every 15 minutes
  - Hourly performance reports
  - Graceful shutdown handling

### 2. Session Monitor
- **File**: `session_monitor.py`
- **Function**: Real-time analysis and reporting
- **Features**:
  - Live performance tracking
  - Trading activity analysis
  - Market condition monitoring
  - Return projections (hourly, daily, weekly, monthly)
  - Updates every 60 seconds

### 3. Dashboard
- **URL**: http://localhost:8505
- **Function**: Visual monitoring interface
- **Features**:
  - Real-time portfolio visualization
  - Trading history with charts
  - Performance metrics
  - Market data display

## 📁 Data Structure

Each live session creates a timestamped directory:
```
data/live_session_YYYY-MM-DD_HH-MM/
├── session_metadata.json      # Session configuration
├── README.md                  # Session documentation
├── trades.json               # All executed trades
├── portfolio_history.json    # Portfolio value over time
├── performance_summary.json  # Current performance metrics
├── price_history.json       # Market price data
├── final_session_report.json # Final comprehensive report
└── hourly_reports/           # Hourly analysis reports
    ├── hour_01_14-00.json
    ├── hour_02_15-00.json
    └── ...
```

## ⚙️ Trading Parameters

### Conservative Live Settings
- **Buy Threshold**: -0.8% (buy when price drops 0.8% from recent high)
- **Sell Threshold**: +1.2% (sell when price rises 1.2% from recent low)
- **Lookback Period**: 8 intervals (40 minutes)
- **Min Trade Amount**: $25 CAD
- **Max Position Size**: 30% of portfolio
- **Check Interval**: 5 minutes

### Accurate Kraken Fees
- **Maker Fee**: 0.25%
- **Taker Fee**: 0.40%
- **Slippage Buffer**: 0.10%

## 📈 Monitoring Features

### Real-Time Session Monitor Output
```
================================================================================
🚀 LIVE TRADING SESSION STATUS
================================================================================
📅 Session: live_session_2025-08-20_13-15
⏰ Started: 2025-08-20 13:15:30
🕐 Runtime: 2:45:00

💰 PORTFOLIO PERFORMANCE:
   Initial Balance: $100.00
   Current Value:   $102.35
   Total Return:    +2.35% (+$2.35)
   Total Trades:    8
   Total Fees:      $0.32

📊 RETURN PROJECTIONS:
   Hourly:   +0.855%
   Daily:    +20.52%
   Weekly:   +143.64%
   Monthly:  +615.60%

🔄 TRADING ACTIVITY:
   Total Trades: 8
   Recent (1h):  3
   ETH/CAD: 2 buys, 2 sells, $40.00 volume
   BTC/CAD: 2 buys, 2 sells, $40.00 volume
   Latest: SELL ETH/CAD at 14:32:15

📈 MARKET CONDITIONS:
   ETH/CAD:
     Current: $5,967.39
     Session: +0.15%
     Volatility: 0.45%
     Range: $5,945.60 - $5,995.58
   BTC/CAD:
     Current: $157,781.10
     Session: -0.23%
     Volatility: 0.38%
     Range: $157,041.60 - $158,453.75
================================================================================
```

### Dashboard Features
- **Overview Tab**: Combined strategy performance
- **Simple Pair Trading Tab**: 
  - Portfolio performance charts
  - Trading history with buy/sell markers
  - Current positions
  - Performance metrics
- **Live Monitoring Tab**: 
  - Current market prices
  - Active session detection
  - System status
  - Quick action buttons

## 📊 Reports Generated

### Hourly Reports
- Performance summary
- Market analysis
- Trading activity
- Risk metrics
- Return projections

### Final Session Report
- Complete session summary
- Comprehensive performance analysis
- Market movement analysis
- Trading statistics
- Risk assessment

## 🛑 Stopping the Operation

### Graceful Shutdown
- Press `Ctrl+C` in the trading session window
- All data will be automatically saved
- Final report will be generated
- Dashboard will be stopped

### Emergency Stop
```bash
# Kill all processes
taskkill /f /im python.exe
taskkill /f /im streamlit.exe
```

## 📋 Best Practices

### 1. Monitor Regularly
- Check the session monitor output every hour
- Review dashboard for visual confirmation
- Watch for unusual market conditions

### 2. Data Management
- Sessions create significant data
- Archive old sessions periodically
- Monitor disk space usage

### 3. Performance Analysis
- Review hourly reports for trends
- Compare actual vs projected returns
- Analyze trading frequency and success rate

### 4. Risk Management
- Monitor max drawdown
- Watch for excessive trading
- Review fee impact on returns

## 🔧 Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Restart dashboard manually
streamlit run simple_dashboard.py --server.port 8505
```

#### Session Monitor Not Finding Session
- Check if trading session is still running
- Verify data directory exists
- Ensure session files are being created

#### API Rate Limits
- System includes automatic rate limiting
- If issues persist, increase check intervals
- Monitor exchange connection status

### Log Files
- Trading session logs to console and files
- Check `logs/` directory for detailed logs
- Monitor for error messages or warnings

## 📈 Expected Behavior

### Normal Operation
- Regular price updates every 5 minutes
- Occasional trades based on market conditions
- Steady data collection and reporting
- Gradual portfolio value changes

### Market Conditions Impact
- **Volatile Markets**: More frequent trading
- **Stable Markets**: Fewer trades, steady monitoring
- **Trending Markets**: Directional position building

## 🎯 Success Metrics

### Short-term (1-4 hours)
- System stability and data collection
- Successful trade execution
- Accurate fee calculations
- Proper risk management

### Medium-term (4-24 hours)
- Portfolio performance vs market
- Trading frequency optimization
- Risk-adjusted returns
- Data quality and completeness

### Long-term (1+ days)
- Consistent profitability
- Risk management effectiveness
- Strategy parameter optimization
- Comprehensive data for analysis

## 🚀 Ready to Start

Run the startup script and begin your live trading operation:

```bash
.\start_live_operation.ps1
```

The system will handle everything automatically and provide comprehensive monitoring and analysis throughout the session.

**Dashboard**: http://localhost:8505
**Data Directory**: `data/live_session_*`
**Monitoring**: Real-time console output

Good luck with your live trading operation! 📈🚀