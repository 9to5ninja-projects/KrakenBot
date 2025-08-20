# KrakenBot Data Directory

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
