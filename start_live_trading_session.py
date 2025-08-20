"""
Start a live trading session with comprehensive monitoring and analysis.
This simulates starting active trading today with full data collection.
"""
import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import threading
import signal

from simple_pair_trader import SimplePairTrader
import config

class LiveTradingSession:
    """Manages a complete live trading session with monitoring and analysis."""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("live_session_%Y-%m-%d_%H-%M")
        self.base_dir = Path("data") / self.session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trader with realistic live trading parameters
        self.trader = SimplePairTrader(
            pairs=['ETH/CAD', 'BTC/CAD'],
            initial_balance=config.START_AMOUNT
        )
        
        # Live trading parameters (more conservative than demo)
        self.trader.buy_threshold = -0.008   # Buy on 0.8% drop
        self.trader.sell_threshold = 0.012   # Sell on 1.2% rise
        self.trader.lookback_periods = 8     # 40-minute lookback (8 * 5min)
        self.trader.min_trade_amount = 25.0  # Minimum $25 per trade
        self.trader.max_position_size = 0.3  # Max 30% per position
        
        # Use accurate Kraken fees
        self.trader.maker_fee = config.MAKER_FEE
        self.trader.taker_fee = config.TAKER_FEE
        
        self.running = True
        self.dashboard_process = None
        
        logger.info(f"🚀 LIVE TRADING SESSION INITIALIZED: {self.session_id}")
        logger.info(f"📁 Data directory: {self.base_dir}")
        logger.info(f"💰 Initial balance: ${self.trader.initial_balance:.2f}")
        logger.info(f"📊 Trading parameters:")
        logger.info(f"   Buy threshold: {self.trader.buy_threshold*100:.1f}%")
        logger.info(f"   Sell threshold: {self.trader.sell_threshold*100:.1f}%")
        logger.info(f"   Lookback periods: {self.trader.lookback_periods}")
        logger.info(f"   Min trade amount: ${self.trader.min_trade_amount}")
        logger.info(f"   Max position size: {self.trader.max_position_size*100:.0f}%")
    
    def start_dashboard(self):
        """Start the dashboard in a separate process."""
        try:
            # Kill any existing streamlit processes
            os.system("taskkill /f /im streamlit.exe 2>nul")
            time.sleep(2)
            
            # Start new dashboard
            cmd = ["streamlit", "run", "simple_dashboard.py", "--server.port", "8505"]
            self.dashboard_process = subprocess.Popen(
                cmd, 
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            logger.info("📊 Dashboard starting on http://localhost:8505")
            time.sleep(5)  # Give dashboard time to start
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    def create_session_metadata(self):
        """Create session metadata file."""
        metadata = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "trading_pairs": self.trader.pairs,
            "initial_balance": self.trader.initial_balance,
            "parameters": {
                "buy_threshold": self.trader.buy_threshold,
                "sell_threshold": self.trader.sell_threshold,
                "lookback_periods": self.trader.lookback_periods,
                "min_trade_amount": self.trader.min_trade_amount,
                "max_position_size": self.trader.max_position_size,
                "maker_fee": self.trader.maker_fee,
                "taker_fee": self.trader.taker_fee
            },
            "strategy": "Simple Pair Trading",
            "mode": "Live Simulation",
            "description": "Live trading session started today with comprehensive monitoring"
        }
        
        with open(self.base_dir / "session_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create README for the session
        readme_content = f"""# Live Trading Session: {self.session_id}

## Session Overview
- **Start Time**: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
- **Strategy**: Simple Pair Trading (ETH/CAD, BTC/CAD)
- **Initial Balance**: ${self.trader.initial_balance:.2f} CAD
- **Mode**: Live Simulation (Mock Trading)

## Trading Parameters
- **Buy Threshold**: {self.trader.buy_threshold*100:.1f}% (buy when price drops)
- **Sell Threshold**: {self.trader.sell_threshold*100:.1f}% (sell when price rises)
- **Lookback Period**: {self.trader.lookback_periods} intervals (40 minutes)
- **Min Trade Amount**: ${self.trader.min_trade_amount}
- **Max Position Size**: {self.trader.max_position_size*100:.0f}% of portfolio

## Fees (Kraken Accurate)
- **Maker Fee**: {self.trader.maker_fee*100:.2f}%
- **Taker Fee**: {self.trader.taker_fee*100:.2f}%

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
"""
        
        with open(self.base_dir / "README.md", "w") as f:
            f.write(readme_content)
    
    def run_trading_loop(self):
        """Main trading loop with 5-minute intervals."""
        logger.info("🔄 Starting live trading loop (5-minute intervals)")
        
        step_count = 0
        last_hourly_report = self.session_start
        last_save = self.session_start
        
        # Create hourly reports directory
        hourly_dir = self.base_dir / "hourly_reports"
        hourly_dir.mkdir(exist_ok=True)
        
        try:
            while self.running:
                step_count += 1
                current_time = datetime.now()
                elapsed_hours = (current_time - self.session_start).total_seconds() / 3600
                
                logger.info(f"=== LIVE STEP {step_count} - {current_time.strftime('%H:%M:%S')} ===")
                logger.info(f"⏱️  Session runtime: {elapsed_hours:.2f} hours")
                
                # Run strategy step
                step_result = self.trader.run_strategy_step()
                
                if step_result:
                    portfolio_value = step_result['portfolio_value']
                    total_return = (portfolio_value - self.trader.initial_balance) / self.trader.initial_balance * 100
                    
                    logger.info(f"💰 Portfolio: ${portfolio_value:.2f} (Return: {total_return:+.2f}%)")
                    logger.info(f"💵 CAD balance: ${step_result['cad_balance']:.2f}")
                    
                    if step_result['trades_executed'] > 0:
                        logger.info(f"🔥 TRADES EXECUTED: {step_result['trades_executed']}")
                        
                        # Log trade details
                        for trade in self.trader.trades[-step_result['trades_executed']:]:
                            logger.info(f"   {trade.side.upper()} {trade.amount:.6f} {trade.pair} at ${trade.price:.2f}")
                    
                    # Log current market prices and positions
                    logger.info("📈 Current positions:")
                    for pair, price in step_result['prices'].items():
                        position_amount = step_result['positions'].get(pair, 0)
                        if position_amount > 0:
                            position_value = position_amount * price
                            logger.info(f"   {pair}: ${price:,.2f} | Position: {position_amount:.6f} (${position_value:.2f})")
                        else:
                            logger.info(f"   {pair}: ${price:,.2f} | No position")
                
                # Generate hourly report
                if (current_time - last_hourly_report).total_seconds() >= 3600:  # Every hour
                    self.generate_hourly_report(current_time, step_count)
                    last_hourly_report = current_time
                
                # Save data every 15 minutes
                if (current_time - last_save).total_seconds() >= 900:  # 15 minutes
                    self.trader.save_data(str(self.base_dir))
                    logger.info(f"💾 Data saved to {self.base_dir}")
                    last_save = current_time
                
                # Wait 5 minutes for next check
                if self.running:
                    logger.info("⏳ Waiting 5 minutes until next check...")
                    time.sleep(300)  # 5 minutes
        
        except KeyboardInterrupt:
            logger.info("⚠️  Trading session interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}")
        finally:
            self.cleanup()
    
    def generate_hourly_report(self, current_time, step_count):
        """Generate detailed hourly performance report."""
        logger.info("📊 Generating hourly report...")
        
        summary = self.trader.get_performance_summary()
        runtime_hours = (current_time - self.session_start).total_seconds() / 3600
        
        report = {
            "timestamp": current_time.isoformat(),
            "session_runtime_hours": runtime_hours,
            "steps_completed": step_count,
            "performance": summary,
            "market_analysis": {
                "price_movements": self.analyze_price_movements(),
                "trading_activity": self.analyze_trading_activity(),
                "risk_metrics": self.calculate_risk_metrics()
            },
            "projections": {
                "daily_return_projection": summary['total_return_pct'] / runtime_hours * 24 if runtime_hours > 0 else 0,
                "weekly_return_projection": summary['total_return_pct'] / runtime_hours * 24 * 7 if runtime_hours > 0 else 0,
                "monthly_return_projection": summary['total_return_pct'] / runtime_hours * 24 * 30 if runtime_hours > 0 else 0
            }
        }
        
        # Save hourly report
        report_filename = f"hour_{int(runtime_hours):02d}_{current_time.strftime('%H-%M')}.json"
        with open(self.base_dir / "hourly_reports" / report_filename, "w") as f:
            json.dump(report, f, indent=2)
        
        # Log key metrics
        logger.info("📊 HOURLY PERFORMANCE SUMMARY:")
        logger.info(f"   Runtime: {runtime_hours:.1f} hours")
        logger.info(f"   Portfolio Value: ${summary['current_value']:.2f}")
        logger.info(f"   Total Return: {summary['total_return_pct']:.2f}%")
        logger.info(f"   Total Trades: {summary['total_trades']}")
        logger.info(f"   Daily Projection: {report['projections']['daily_return_projection']:.2f}%")
    
    def analyze_price_movements(self):
        """Analyze price movements during the session."""
        if not self.trader.price_history:
            return {}
        
        analysis = {}
        for pair, price_data in self.trader.price_history.items():
            if len(price_data) > 1:
                prices = [p['price'] for p in price_data]
                start_price = prices[0]
                current_price = prices[-1]
                min_price = min(prices)
                max_price = max(prices)
                
                analysis[pair] = {
                    "start_price": start_price,
                    "current_price": current_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "total_movement_pct": (current_price - start_price) / start_price * 100,
                    "volatility_range_pct": (max_price - min_price) / min_price * 100
                }
        
        return analysis
    
    def analyze_trading_activity(self):
        """Analyze trading activity patterns."""
        if not self.trader.trades:
            return {"total_trades": 0}
        
        buy_trades = [t for t in self.trader.trades if t.side == 'buy']
        sell_trades = [t for t in self.trader.trades if t.side == 'sell']
        
        return {
            "total_trades": len(self.trader.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "avg_trade_size": sum(t.cost for t in self.trader.trades) / len(self.trader.trades),
            "total_fees_paid": sum(t.fee for t in self.trader.trades),
            "pairs_traded": list(set(t.pair for t in self.trader.trades))
        }
    
    def calculate_risk_metrics(self):
        """Calculate risk metrics for the session."""
        if not self.trader.portfolio_history:
            return {}
        
        portfolio_values = [entry['portfolio_value'] for entry in self.trader.portfolio_history]
        
        if len(portfolio_values) < 2:
            return {}
        
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        
        max_value = max(portfolio_values)
        min_value = min(portfolio_values)
        current_value = portfolio_values[-1]
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "max_portfolio_value": max_value,
            "min_portfolio_value": min_value,
            "current_drawdown": (max_value - current_value) / max_value if max_value > 0 else 0,
            "max_drawdown": max_drawdown,
            "volatility": (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns))**0.5 if returns else 0
        }
    
    def cleanup(self):
        """Clean up resources and save final data."""
        logger.info("🧹 Cleaning up trading session...")
        
        # Save final data
        self.trader.save_data(str(self.base_dir))
        
        # Generate final session report
        self.generate_final_report()
        
        # Stop dashboard
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                logger.info("📊 Dashboard stopped")
            except:
                pass
        
        logger.info(f"✅ Session data saved to: {self.base_dir}")
        logger.info("🏁 Live trading session completed!")
    
    def generate_final_report(self):
        """Generate comprehensive final session report."""
        end_time = datetime.now()
        runtime = end_time - self.session_start
        
        summary = self.trader.get_performance_summary()
        
        final_report = {
            "session_summary": {
                "session_id": self.session_id,
                "start_time": self.session_start.isoformat(),
                "end_time": end_time.isoformat(),
                "total_runtime": str(runtime),
                "runtime_hours": runtime.total_seconds() / 3600
            },
            "performance": summary,
            "market_analysis": {
                "price_movements": self.analyze_price_movements(),
                "trading_activity": self.analyze_trading_activity(),
                "risk_metrics": self.calculate_risk_metrics()
            },
            "session_statistics": {
                "data_points_collected": len(self.trader.portfolio_history),
                "price_updates": sum(len(prices) for prices in self.trader.price_history.values()),
                "average_check_interval": runtime.total_seconds() / len(self.trader.portfolio_history) if self.trader.portfolio_history else 0
            }
        }
        
        with open(self.base_dir / "final_session_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        logger.info("📋 FINAL SESSION REPORT:")
        logger.info(f"   Session ID: {self.session_id}")
        logger.info(f"   Runtime: {runtime}")
        logger.info(f"   Final Portfolio Value: ${summary['current_value']:.2f}")
        logger.info(f"   Total Return: {summary['total_return_pct']:.2f}%")
        logger.info(f"   Total Trades: {summary['total_trades']}")
        logger.info(f"   Data Points: {len(self.trader.portfolio_history)}")
    
    def start(self):
        """Start the complete live trading session."""
        logger.info("🚀 STARTING LIVE TRADING SESSION")
        logger.info("=" * 60)
        
        # Create session files
        self.create_session_metadata()
        
        # Start dashboard
        self.start_dashboard()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        
        logger.info("📊 Dashboard: http://localhost:8505")
        logger.info("📁 Data Directory: " + str(self.base_dir))
        logger.info("⏰ Check Interval: 5 minutes")
        logger.info("🛑 Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)
        
        # Start trading loop
        self.run_trading_loop()

def main():
    """Main function to start live trading session."""
    print("🚀 KrakenBot Live Trading Session")
    print("=" * 50)
    print("This will start a live trading simulation with:")
    print("• Real market data collection")
    print("• 5-minute trading intervals")
    print("• Live dashboard on port 8505")
    print("• Comprehensive hourly reports")
    print("• Full data logging and analysis")
    print("=" * 50)
    
    response = input("Start live trading session? (y/N): ").strip().lower()
    
    if response == 'y':
        session = LiveTradingSession()
        session.start()
    else:
        print("Session cancelled.")

if __name__ == "__main__":
    main()