# KrakenBot: Advanced Cryptocurrency Trading System

A comprehensive Python-based trading system for Kraken exchange featuring triangular arbitrage monitoring and simple pair trading strategies with live operation capabilities.

## 🚀 Overview

KrakenBot has evolved from a simple arbitrage monitor into a full-featured trading system that:

### Triangular Arbitrage Strategy
1. **Real-time Monitoring**: Continuously scans for arbitrage opportunities
2. **Multi-path Analysis**: Analyzes BTC/CAD, ETH/CAD, and ETH/BTC triangular paths
3. **Fee-aware Calculations**: Accounts for Kraken's maker/taker fees and slippage
4. **Statistical Analysis**: Identifies optimal entry points and market conditions

### Simple Pair Trading Strategy
1. **Technical Analysis**: Uses moving averages and price momentum indicators
2. **Risk Management**: Position sizing, exposure limits, and stop-loss mechanisms
3. **Live Operation**: Real-time trading simulation with comprehensive monitoring
4. **Performance Tracking**: Detailed analytics and return projections

## ✨ Key Features

- **🔄 Live Trading Operation**: Complete trading session management with real-time monitoring
- **📊 Advanced Dashboard**: Multi-strategy visualization with real-time charts and metrics
- **📈 Performance Analytics**: Comprehensive tracking of returns, risk metrics, and projections
- **🛡️ Risk Management**: Position sizing, maximum exposure limits, and fee-aware calculations
- **📋 Comprehensive Logging**: All trades, portfolio changes, and market data recorded
- **⏰ Automated Reporting**: Hourly performance reports and session summaries
- **🎯 Multiple Strategies**: Both triangular arbitrage and simple pair trading
- **🔍 Real-time Monitoring**: Live session analysis and market condition tracking

## 🏗️ Project Structure

```
KrakenBot/
├── Core Trading System
│   ├── main.py                      # Original entry point
│   ├── simple_pair_trader.py        # Main trading engine
│   ├── start_live_trading_session.py # Live session manager
│   └── session_monitor.py           # Real-time monitoring
│
├── Strategies
│   ├── arbitrage.py                 # Triangular arbitrage logic
│   ├── triangle_optimizer.py       # Optimal path discovery
│   └── statistical_analyzer.py     # Statistical analysis
│
├── Infrastructure
│   ├── config.py                   # Configuration management
│   ├── exchange.py                 # Kraken API wrapper
│   ├── monitor.py                  # Monitoring system
│   └── logger.py                   # Logging utilities
│
├── Monitoring & Testing
│   ├── run_simple_pair_monitor.py  # Basic monitoring
│   ├── run_full_day_monitor.py     # 24-hour monitoring
│   ├── test_simple_pair_trading.py # Testing utilities
│   ├── run_demo_trading.py         # Demo trading
│   └── generate_sample_data.py     # Sample data generation
│
├── Dashboard & Visualization
│   ├── simple_dashboard.py         # Reliable dashboard
│   ├── enhanced_dashboard.py       # Full-featured dashboard
│   └── dashboard.py                # Original dashboard
│
├── Operations
│   ├── start_live_operation.ps1    # PowerShell startup
│   ├── start_live_operation.bat    # Windows batch startup
│   └── trader.py                   # Trading execution
│
├── Documentation
│   ├── SIMPLE_PAIR_TRADING_GUIDE.md # Usage guide
│   ├── LIVE_TRADING_GUIDE.md       # Live operation guide
│   ├── CHANGELOG.md                # Version history
│   └── README.md                   # This file
│
└── Data & Configuration
    ├── requirements.txt            # Dependencies
    ├── .env.example               # Configuration template
    ├── data/                      # Data storage
    │   ├── live_session_*/        # Live trading sessions
    │   ├── historical/            # Historical data
    │   └── statistical/           # Analysis results
    └── logs/                      # Log files
```

## 🚀 Quick Start

### Option 1: Live Trading Operation (Recommended)
```bash
# Start complete live trading system
.\start_live_operation.ps1
```

This launches:
- Live trading session with 5-minute intervals
- Real-time dashboard on http://localhost:8505
- Session monitor with live analysis
- Comprehensive data collection

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Run specific components
python start_live_trading_session.py  # Live trading
python session_monitor.py             # Real-time monitoring
streamlit run simple_dashboard.py     # Dashboard
```

## 📊 Trading Strategies

### 1. Simple Pair Trading
**Pairs**: ETH/CAD, BTC/CAD
**Strategy**: Buy low, sell high with technical indicators
**Parameters**:
- Buy Threshold: -0.8% (buy on price drops)
- Sell Threshold: +1.2% (sell on price rises)
- Lookback Period: 8 intervals (40 minutes)
- Max Position: 30% of portfolio
- Min Trade: $25 CAD

### 2. Triangular Arbitrage
**Paths**: CAD→BTC→ETH→CAD, CAD→ETH→BTC→CAD
**Strategy**: Exploit price differences across three currencies
**Features**:
- Real-time opportunity detection
- Fee-aware profit calculations
- Statistical analysis of optimal paths
- Historical performance tracking

## 🎯 Live Operation Features

### Real-time Monitoring
- **5-minute intervals**: Regular market data collection
- **Live dashboard**: Visual portfolio and performance tracking
- **Session monitor**: Real-time analysis and projections
- **Automatic reporting**: Hourly performance summaries

### Data Collection
- **Trade History**: All executed trades with timestamps
- **Portfolio Tracking**: Value changes over time
- **Market Data**: Price history and volatility analysis
- **Performance Metrics**: Returns, risk metrics, projections

### Risk Management
- **Position Sizing**: Dynamic sizing based on portfolio value
- **Exposure Limits**: Maximum 30% per position
- **Fee Integration**: Accurate Kraken fees (0.25% maker, 0.40% taker)
- **Stop-loss**: Automatic position management

## 📈 Dashboard Features

Access the dashboard at **http://localhost:8505**

### Overview Tab
- Combined strategy performance
- Portfolio value over time
- Key performance metrics

### Simple Pair Trading Tab
- Trading history with buy/sell markers
- Current positions and P&L
- Performance analytics
- Risk metrics

### Triangular Arbitrage Tab
- Opportunity analysis
- Profit distribution charts
- Historical performance
- Recent opportunities

### Live Monitoring Tab
- Current market prices
- Active session status
- System configuration
- Quick action buttons

## 🔧 Configuration

### Environment Variables (.env file)
```bash
# Trading Parameters
START_AMOUNT=100.0
MAKER_FEE=0.0025      # 0.25%
TAKER_FEE=0.004       # 0.40%
SLIPPAGE_BUFFER=0.001 # 0.10%

# Trading Pairs
PAIR_1=BTC/CAD
PAIR_2=ETH/CAD
PAIR_3=ETH/BTC

# API Configuration (optional for monitoring)
KRAKEN_API_KEY=your_api_key
KRAKEN_SECRET=your_secret_key

# System Settings
LOG_LEVEL=INFO
CHECK_INTERVAL=300    # 5 minutes
```

### Trading Parameters
- **Conservative**: Lower risk, fewer trades
- **Aggressive**: Higher risk, more frequent trading
- **Demo**: High sensitivity for testing

## 📋 Usage Examples

### Live Trading Session
```bash
# Start live operation
python start_live_trading_session.py

# Monitor session
python session_monitor.py --interval 30

# View dashboard
streamlit run simple_dashboard.py --server.port 8505
```

### Testing & Demo
```bash
# Quick 5-minute test
python run_simple_pair_monitor.py --test

# Full day monitoring
python run_full_day_monitor.py

# Demo with sensitive parameters
python run_demo_trading.py

# Generate sample data
python generate_sample_data.py
```

### Original Arbitrage System
```bash
# Monitor arbitrage opportunities
python main.py --mode monitor --amount 100 --min-profit-pct 1.5

# Run dashboard
python main.py --mode dashboard

# Optimize triangular paths
python main.py optimize --days 90 --base CAD
```

## 📊 Performance Tracking

### Metrics Tracked
- **Portfolio Value**: Real-time portfolio tracking
- **Total Returns**: Percentage and absolute returns
- **Trade Statistics**: Win rate, frequency, average size
- **Risk Metrics**: Volatility, maximum drawdown, Sharpe ratio
- **Fee Analysis**: Total fees paid and impact on returns

### Projections
- **Hourly**: Short-term performance trends
- **Daily**: 24-hour return projections
- **Weekly**: 7-day performance estimates
- **Monthly**: Long-term return projections

## 🛠️ Development & Testing

### Test Modes
- **Demo Mode**: High-sensitivity parameters for quick results
- **Simulation Mode**: Market volatility simulation
- **Live Mode**: Real market data with mock trading

### Data Analysis
- **Historical Analysis**: Price movements and patterns
- **Performance Validation**: Strategy effectiveness
- **Risk Assessment**: Drawdown and volatility analysis

## 🔄 Operational Workflow

1. **Session Start**: Initialize trading session with metadata
2. **Data Collection**: Real-time market data every 5 minutes
3. **Signal Analysis**: Technical indicators and trading signals
4. **Trade Execution**: Mock trades with accurate fee calculations
5. **Performance Tracking**: Portfolio value and metrics updates
6. **Reporting**: Hourly reports and continuous monitoring
7. **Session End**: Final report and data archival

## 📁 Data Structure

Each session creates organized data:
```
data/live_session_YYYY-MM-DD_HH-MM/
├── session_metadata.json      # Configuration
├── README.md                  # Session documentation
├── trades.json               # Trade history
├── portfolio_history.json    # Portfolio tracking
├── performance_summary.json  # Metrics
├── price_history.json       # Market data
├── final_session_report.json # Final analysis
└── hourly_reports/           # Hourly summaries
```

## 🚨 Risk Disclaimer

**This software is for educational and simulation purposes only.**

- All trading is currently **MOCK TRADING** - no real money is used
- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk
- Always test thoroughly before considering real trading
- The authors are not responsible for any financial losses

## 🔮 Future Roadmap

### Phase 3: Live Trading Implementation
- Real money trading capabilities
- Advanced order types and execution
- Enhanced risk management
- Multi-exchange support

### Phase 4: Advanced Features
- Machine learning integration
- Additional trading strategies
- Portfolio optimization
- Advanced analytics

## 📄 License

MIT License - See LICENSE file for details

---

**Ready to start?** Run `.\start_live_operation.ps1` and begin your trading journey! 🚀