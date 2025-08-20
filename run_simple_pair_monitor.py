"""
Monitor and execute simple pair trading strategy for CAD/ETH and CAD/BTC.
Collects real data for analysis and runs mock investment simulation.
"""
import os
import sys
import time
import json
import signal
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from simple_pair_trader import SimplePairTrader
import config

class SimplePairMonitor:
    """Monitor for simple pair trading strategy."""
    
    def __init__(self, duration_hours: float = 24, check_interval: int = 60):
        """
        Initialize the monitor.
        
        Args:
            duration_hours: How long to run the monitor (default: 24 hours)
            check_interval: Seconds between checks (default: 60 seconds)
        """
        self.duration_hours = duration_hours
        self.check_interval = check_interval
        self.trader = SimplePairTrader()
        self.running = True
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create data directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.data_dir = Path("data") / f"simple_pair_monitor_{timestamp}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Simple pair monitor initialized:")
        logger.info(f"  Duration: {duration_hours} hours")
        logger.info(f"  Check interval: {check_interval} seconds")
        logger.info(f"  Data directory: {self.data_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Run the monitoring loop."""
        logger.info("Starting simple pair trading monitor...")
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(hours=self.duration_hours)
        
        step_count = 0
        last_summary_time = self.start_time
        
        try:
            while self.running and datetime.now() < end_time:
                step_count += 1
                current_time = datetime.now()
                
                logger.info(f"=== Step {step_count} - {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                
                # Run strategy step
                step_result = self.trader.run_strategy_step()
                
                if step_result:
                    logger.info(f"Portfolio value: ${step_result['portfolio_value']:.2f}")
                    logger.info(f"CAD balance: ${step_result['cad_balance']:.2f}")
                    logger.info(f"Trades executed: {step_result['trades_executed']}")
                    
                    # Log current prices
                    for pair, price in step_result['prices'].items():
                        position_amount = step_result['positions'].get(pair, 0)
                        logger.info(f"  {pair}: ${price:.2f} (Position: {position_amount:.6f})")
                
                # Print summary every hour
                if (current_time - last_summary_time).total_seconds() >= 3600:  # 1 hour
                    self._print_summary()
                    last_summary_time = current_time
                
                # Save data periodically (every 10 steps)
                if step_count % 10 == 0:
                    self.trader.save_data(str(self.data_dir))
                
                # Wait for next check
                if self.running and datetime.now() < end_time:
                    logger.info(f"Waiting {self.check_interval} seconds until next check...")
                    time.sleep(self.check_interval)
            
            # Final summary and data save
            logger.info("=== MONITORING COMPLETED ===")
            self._print_final_summary()
            self.trader.save_data(str(self.data_dir))
            self._generate_report()
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            # Always save data on exit
            self.trader.save_data(str(self.data_dir))
            logger.info(f"Final data saved to {self.data_dir}")
    
    def _print_summary(self):
        """Print current performance summary."""
        summary = self.trader.get_performance_summary()
        
        if summary:
            logger.info("=== PERFORMANCE SUMMARY ===")
            logger.info(f"Initial balance: ${summary['initial_balance']:.2f}")
            logger.info(f"Current value: ${summary['current_value']:.2f}")
            logger.info(f"Total return: {summary['total_return_pct']:.2f}% (${summary['total_return_cad']:.2f})")
            logger.info(f"Total trades: {summary['total_trades']}")
            logger.info(f"Total fees paid: ${summary['total_fees_paid']:.2f}")
            logger.info(f"Realized P&L: ${summary['total_realized_pnl']:.2f}")
            logger.info(f"Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
            
            logger.info("Current positions:")
            for pair, pos_info in summary['positions'].items():
                if pos_info['amount'] > 0:
                    logger.info(f"  {pair}: {pos_info['amount']:.6f} @ ${pos_info['avg_buy_price']:.2f} (P&L: ${pos_info['unrealized_pnl']:.2f})")
                else:
                    logger.info(f"  {pair}: No position (Realized P&L: ${pos_info['realized_pnl']:.2f})")
    
    def _print_final_summary(self):
        """Print final performance summary."""
        runtime = datetime.now() - self.start_time
        
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Data collected in: {self.data_dir}")
        
        self._print_summary()
        
        # Additional statistics
        if self.trader.trades:
            buy_trades = [t for t in self.trader.trades if t.side == 'buy']
            sell_trades = [t for t in self.trader.trades if t.side == 'sell']
            
            logger.info(f"Buy trades: {len(buy_trades)}")
            logger.info(f"Sell trades: {len(sell_trades)}")
            
            if buy_trades:
                avg_buy_price = sum(t.price for t in buy_trades) / len(buy_trades)
                logger.info(f"Average buy price: ${avg_buy_price:.2f}")
            
            if sell_trades:
                avg_sell_price = sum(t.price for t in sell_trades) / len(sell_trades)
                logger.info(f"Average sell price: ${avg_sell_price:.2f}")
    
    def _generate_report(self):
        """Generate a comprehensive report."""
        report = {
            'monitoring_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': self.duration_hours,
                'check_interval': self.check_interval,
                'data_directory': str(self.data_dir)
            },
            'performance': self.trader.get_performance_summary(),
            'configuration': {
                'pairs': self.trader.pairs,
                'initial_balance': self.trader.initial_balance,
                'min_trade_amount': self.trader.min_trade_amount,
                'max_position_size': self.trader.max_position_size,
                'buy_threshold': self.trader.buy_threshold,
                'sell_threshold': self.trader.sell_threshold,
                'maker_fee': self.trader.maker_fee,
                'taker_fee': self.trader.taker_fee
            }
        }
        
        # Save report
        with open(self.data_dir / 'monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate README
        readme_content = f"""# Simple Pair Trading Monitor Report

## Monitoring Session
- **Start Time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {self.duration_hours} hours
- **Check Interval**: {self.check_interval} seconds

## Strategy Configuration
- **Trading Pairs**: {', '.join(self.trader.pairs)}
- **Initial Balance**: ${self.trader.initial_balance:.2f} CAD
- **Buy Threshold**: {self.trader.buy_threshold*100:.1f}% drop from recent high
- **Sell Threshold**: {self.trader.sell_threshold*100:.1f}% rise from recent low
- **Max Position Size**: {self.trader.max_position_size*100:.0f}% of portfolio
- **Maker Fee**: {self.trader.maker_fee*100:.2f}%
- **Taker Fee**: {self.trader.taker_fee*100:.2f}%

## Performance Summary
"""
        
        summary = self.trader.get_performance_summary()
        if summary:
            readme_content += f"""- **Final Portfolio Value**: ${summary['current_value']:.2f}
- **Total Return**: {summary['total_return_pct']:.2f}% (${summary['total_return_cad']:.2f})
- **Total Trades**: {summary['total_trades']}
- **Total Fees Paid**: ${summary['total_fees_paid']:.2f}
- **Realized P&L**: ${summary['total_realized_pnl']:.2f}
- **Unrealized P&L**: ${summary['total_unrealized_pnl']:.2f}

## Files Generated
- `trades.json` - All executed trades
- `portfolio_history.json` - Portfolio value over time
- `performance_summary.json` - Performance metrics
- `price_history.json` - Price data collected
- `monitoring_report.json` - Complete monitoring report
"""
        
        with open(self.data_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Report generated: {self.data_dir / 'README.md'}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Pair Trading Monitor")
    parser.add_argument('--duration', type=float, default=24, 
                       help='Duration to run in hours (default: 24)')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Check interval in seconds (default: 60)')
    parser.add_argument('--test', action='store_true', 
                       help='Run a short test (5 minutes)')
    
    args = parser.parse_args()
    
    if args.test:
        duration = 5/60  # 5 minutes
        interval = 30    # 30 seconds
        print("Running in TEST mode (5 minutes, 30-second intervals)")
    else:
        duration = args.duration
        interval = args.interval
    
    monitor = SimplePairMonitor(duration_hours=duration, check_interval=interval)
    monitor.run()

if __name__ == "__main__":
    main()