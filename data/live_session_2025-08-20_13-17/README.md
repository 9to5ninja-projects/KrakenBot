# Live Trading Session: live_session_2025-08-20_13-17

## Session Overview
- **Start Time**: 2025-08-20 13:17:48
- **Strategy**: Simple Pair Trading (ETH/CAD, BTC/CAD)
- **Initial Balance**: $100.00 CAD
- **Mode**: Live Simulation (Mock Trading)

## Trading Parameters
- **Buy Threshold**: -0.8% (buy when price drops)
- **Sell Threshold**: 1.2% (sell when price rises)
- **Lookback Period**: 8 intervals (40 minutes)
- **Min Trade Amount**: $25.0
- **Max Position Size**: 30% of portfolio

## Fees (Kraken Accurate)
- **Maker Fee**: 0.25%
- **Taker Fee**: 0.40%

## Files Generated
- `session_metadata.json` - Session configuration
- `trades.json` - All executed trades
- `portfolio_history.json` - Portfolio value over time
- `performance_summary.json` - Performance metrics
- `price_history.json` - Market price data
- `hourly_reports/` - Hourly performance reports
- `analysis/` - Detailed analysis files

## Dashboard
Access the live dashboard at: http://localhost:8505

## Status
Session is actively running and collecting real market data.
