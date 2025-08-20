"""
Prepare repository for commit with comprehensive documentation and organization.
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def create_commit_summary():
    """Create a comprehensive commit summary."""
    
    summary = {
        "commit_info": {
            "version": "2.0.0",
            "date": datetime.now().isoformat(),
            "title": "Live Trading System Implementation",
            "description": "Major release implementing simple pair trading with live operation capabilities"
        },
        "new_files": [
            # Core Trading System
            "simple_pair_trader.py",
            "start_live_trading_session.py", 
            "session_monitor.py",
            
            # Monitoring & Testing
            "run_simple_pair_monitor.py",
            "run_full_day_monitor.py",
            "test_simple_pair_trading.py",
            "run_demo_trading.py",
            "generate_sample_data.py",
            
            # Dashboard & Visualization
            "simple_dashboard.py",
            "enhanced_dashboard.py",
            
            # Operations
            "start_live_operation.ps1",
            "start_live_operation.bat",
            
            # Documentation
            "SIMPLE_PAIR_TRADING_GUIDE.md",
            "LIVE_TRADING_GUIDE.md",
            "CHANGELOG.md",
            "prepare_commit.py"
        ],
        "updated_files": [
            "README.md",
            "requirements.txt",
            ".env.example"
        ],
        "features_added": [
            "Simple pair trading strategy",
            "Live trading session management",
            "Real-time monitoring and analysis",
            "Enhanced dashboard with multiple strategies",
            "Comprehensive data collection and reporting",
            "Risk management and position sizing",
            "Accurate fee calculations",
            "Hourly performance reporting",
            "Session documentation and metadata",
            "Multiple testing and demo modes"
        ],
        "technical_improvements": [
            "Rate limiting and API management",
            "Structured data storage",
            "Background processing",
            "Error handling and recovery",
            "Performance optimization",
            "Memory management",
            "Caching mechanisms"
        ]
    }
    
    return summary

def organize_data_directory():
    """Organize data directory and create archive of sample sessions."""
    data_dir = Path("data")
    
    # Create archive directory for sample data
    archive_dir = data_dir / "sample_sessions_archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Move sample sessions to archive
    sample_patterns = [
        "sample_simple_pair_*",
        "demo_trading_*",
        "simulated_volatile_*",
        "test_simple_pair_*"
    ]
    
    archived_count = 0
    for pattern in sample_patterns:
        for session_dir in data_dir.glob(pattern):
            if session_dir.is_dir():
                dest = archive_dir / session_dir.name
                if not dest.exists():
                    shutil.move(str(session_dir), str(dest))
                    archived_count += 1
    
    print(f"📁 Archived {archived_count} sample sessions to {archive_dir}")
    
    # Create data directory README
    readme_content = """# KrakenBot Data Directory

## Structure

### Active Sessions
- `live_session_*` - Live trading sessions with real market data
- `simple_pair_monitor_*` - Monitoring sessions

### Archives
- `sample_sessions_archive/` - Sample and demo sessions
- `historical/` - Historical price data
- `statistical/` - Statistical analysis results

### System Data
- `opportunities.csv` - Triangular arbitrage opportunities
- `trades.csv` - Historical trades
- `stats.json` - System statistics
- `health.json` - System health data

## Session Data Format

Each session directory contains:
- `session_metadata.json` - Configuration and parameters
- `README.md` - Session documentation
- `trades.json` - All executed trades
- `portfolio_history.json` - Portfolio value over time
- `performance_summary.json` - Performance metrics
- `price_history.json` - Market price data
- `hourly_reports/` - Hourly analysis reports
- `final_session_report.json` - Final comprehensive report

## Data Retention

- Live sessions: Keep indefinitely for analysis
- Sample sessions: Archived automatically
- Historical data: Retained for backtesting
- Logs: Rotated based on size and age
"""
    
    with open(data_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return archived_count

def create_git_commit_message():
    """Create a comprehensive git commit message."""
    
    message = """feat: Implement live trading system with simple pair trading strategy

🚀 Major Release: KrakenBot v2.0.0

## New Features
- ✨ Simple pair trading strategy for ETH/CAD and BTC/CAD
- 🔄 Live trading session management with 5-minute intervals
- 📊 Enhanced dashboard with multi-strategy visualization
- 📈 Real-time monitoring and performance analysis
- 🛡️ Comprehensive risk management and position sizing
- ⏰ Automated hourly reporting and session documentation
- 🎯 Multiple testing modes (demo, simulation, live)

## Technical Improvements
- 🔧 Accurate Kraken fee integration (0.25% maker, 0.40% taker)
- 📁 Structured data storage with JSON-based trade history
- 🚦 Rate limiting and API connection management
- 💾 Incremental data saving and graceful shutdown handling
- 🎨 Caching mechanisms for improved dashboard performance
- 🔍 Real-time session monitoring with live analysis

## New Files Added
### Core System
- simple_pair_trader.py - Main trading engine
- start_live_trading_session.py - Live session manager
- session_monitor.py - Real-time monitoring

### Monitoring & Testing
- run_simple_pair_monitor.py - Basic monitoring
- run_full_day_monitor.py - 24-hour monitoring
- test_simple_pair_trading.py - Testing utilities
- run_demo_trading.py - Demo trading
- generate_sample_data.py - Sample data generation

### Dashboard & UI
- simple_dashboard.py - Reliable dashboard
- enhanced_dashboard.py - Full-featured dashboard

### Operations
- start_live_operation.ps1 - PowerShell startup
- start_live_operation.bat - Windows batch startup

### Documentation
- SIMPLE_PAIR_TRADING_GUIDE.md - Usage guide
- LIVE_TRADING_GUIDE.md - Live operation guide
- CHANGELOG.md - Version history

## Updated Files
- README.md - Comprehensive documentation update
- requirements.txt - New dependencies
- .env.example - Updated configuration template

## Data & Analytics
- 📊 Portfolio performance tracking with returns and risk metrics
- 📈 Hourly performance reports with projections
- 🎯 Trading activity analysis and market condition monitoring
- 📋 Comprehensive session documentation and metadata

## Risk Management
- Position sizing with maximum 30% exposure per pair
- Fee-aware profit calculations with slippage buffers
- Stop-loss mechanisms and portfolio value protection
- Real-time risk metrics and drawdown monitoring

## Breaking Changes
- None - Fully backward compatible with existing arbitrage system

## Migration Notes
- Existing arbitrage functionality remains unchanged
- New simple pair trading runs independently
- All new features are opt-in and don't affect existing workflows

This release transforms KrakenBot from a simple arbitrage monitor into a 
comprehensive trading system capable of live operation with professional-grade 
monitoring, analysis, and risk management.

Closes: #trading-system-implementation
Refs: #live-monitoring #dashboard-enhancement #risk-management"""

    return message

def update_requirements():
    """Update requirements.txt with any new dependencies."""
    
    # Read current requirements
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, 'r') as f:
            current_reqs = f.read()
    else:
        current_reqs = ""
    
    # Add any new requirements that might be missing
    new_reqs = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    updated = False
    for req in new_reqs:
        package_name = req.split('>=')[0]
        if package_name not in current_reqs:
            current_reqs += f"\n{req}"
            updated = True
    
    if updated:
        with open(req_file, 'w') as f:
            f.write(current_reqs.strip() + "\n")
        print(f"📦 Updated requirements.txt with new dependencies")
    
    return updated

def create_env_example_update():
    """Update .env.example with new configuration options."""
    
    env_example = """# KrakenBot Configuration

# Trading Parameters
START_AMOUNT=100.0
MAKER_FEE=0.0025      # Kraken maker fee (0.25%)
TAKER_FEE=0.004       # Kraken taker fee (0.40%)
SLIPPAGE_BUFFER=0.001 # Slippage buffer (0.10%)

# Profit Thresholds
MIN_PROFIT_PCT=1.5    # Minimum profit percentage for arbitrage
PROFIT_THRESHOLD=1.50 # Legacy profit threshold in CAD

# Trading Pairs
PAIR_1=BTC/CAD
PAIR_2=ETH/CAD
PAIR_3=ETH/BTC

# System Settings
CHECK_INTERVAL=300    # Check interval in seconds (5 minutes)
LOG_LEVEL=INFO        # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# API Configuration (Optional - only needed for live trading)
KRAKEN_API_KEY=your_api_key_here
KRAKEN_SECRET=your_secret_key_here

# Notifications (Optional)
ENABLE_NOTIFICATIONS=false
NOTIFICATION_METHOD=Console Only

# Simple Pair Trading Parameters
SIMPLE_PAIR_BUY_THRESHOLD=-0.008   # Buy on 0.8% price drop
SIMPLE_PAIR_SELL_THRESHOLD=0.012   # Sell on 1.2% price rise
SIMPLE_PAIR_LOOKBACK_PERIODS=8     # Lookback periods for analysis
SIMPLE_PAIR_MIN_TRADE_AMOUNT=25.0  # Minimum trade amount in CAD
SIMPLE_PAIR_MAX_POSITION_SIZE=0.3  # Maximum position size (30% of portfolio)

# Dashboard Settings
DASHBOARD_PORT=8505
DASHBOARD_AUTO_REFRESH=false
DASHBOARD_REFRESH_INTERVAL=30

# Data Settings
DATA_RETENTION_DAYS=90
ENABLE_HOURLY_REPORTS=true
ENABLE_SESSION_DOCUMENTATION=true
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    
    print("📝 Updated .env.example with new configuration options")

def main():
    """Main preparation function."""
    print("🚀 Preparing KrakenBot v2.0.0 for commit...")
    print("=" * 60)
    
    # Create commit summary
    summary = create_commit_summary()
    
    # Organize data directory
    archived_count = organize_data_directory()
    
    # Update requirements
    req_updated = update_requirements()
    
    # Update .env.example
    create_env_example_update()
    
    # Create commit message
    commit_message = create_git_commit_message()
    
    # Save commit information
    with open("COMMIT_INFO.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open("COMMIT_MESSAGE.txt", "w", encoding='utf-8') as f:
        f.write(commit_message)
    
    print("\n✅ Repository preparation complete!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   New files: {len(summary['new_files'])}")
    print(f"   Updated files: {len(summary['updated_files'])}")
    print(f"   Features added: {len(summary['features_added'])}")
    print(f"   Data archived: {archived_count} sessions")
    print(f"   Requirements updated: {'Yes' if req_updated else 'No'}")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review COMMIT_MESSAGE.txt for the commit message")
    print(f"   2. Check COMMIT_INFO.json for detailed summary")
    print(f"   3. Run: git add .")
    print(f"   4. Run: git commit -F COMMIT_MESSAGE.txt")
    print(f"   5. Run: git push origin main")
    
    print(f"\n🎯 Live trading session status:")
    live_sessions = list(Path("data").glob("live_session_*"))
    if live_sessions:
        latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
        print(f"   Active session: {latest_session.name}")
        print(f"   Dashboard: http://localhost:8505")
        print(f"   Data directory: {latest_session}")
    else:
        print(f"   No active sessions found")
    
    print("\n🚀 Ready for commit and push to repository!")

if __name__ == "__main__":
    main()