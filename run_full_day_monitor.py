"""
Run a full day of simple pair trading monitoring with real data collection.
This script implements the accurate maker/taker fees and collects real market data.
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
from run_simple_pair_monitor import SimplePairMonitor
import config

class FullDayMonitor(SimplePairMonitor):
    """Extended monitor for full day data collection."""
    
    def __init__(self, duration_hours: float = 24, check_interval: int = 300):
        """
        Initialize the full day monitor.
        
        Args:
            duration_hours: How long to run (default: 24 hours)
            check_interval: Seconds between checks (default: 300 = 5 minutes)
        """
        # Initialize with more conservative settings for real trading
        super().__init__(duration_hours, check_interval)
        
        # Override trader with more realistic settings
        self.trader = SimplePairTrader(
            pairs=['ETH/CAD', 'BTC/CAD'],
            initial_balance=config.START_AMOUNT
        )
        
        # Adjust trading parameters for real market conditions
        self.trader.buy_threshold = -0.015   # Buy on 1.5% drop (more conservative)
        self.trader.sell_threshold = 0.02    # Sell on 2% rise (more conservative)
        self.trader.lookback_periods = 12    # 1 hour lookback (12 * 5min intervals)
        self.trader.min_trade_amount = 25.0  # Minimum $25 CAD per trade
        self.trader.max_position_size = 0.25 # Maximum 25% of portfolio per position
        
        # Use accurate Kraken fees
        self.trader.maker_fee = config.MAKER_FEE  # 0.25%
        self.trader.taker_fee = config.TAKER_FEE  # 0.40%
        
        logger.info("Full day monitor initialized with realistic parameters:")
        logger.info(f"  Buy threshold: {self.trader.buy_threshold*100:.1f}%")
        logger.info(f"  Sell threshold: {self.trader.sell_threshold*100:.1f}%")
        logger.info(f"  Lookback periods: {self.trader.lookback_periods}")
        logger.info(f"  Maker fee: {self.trader.maker_fee*100:.2f}%")
        logger.info(f"  Taker fee: {self.trader.taker_fee*100:.2f}%")
    
    def run(self):
        """Run the full day monitoring with enhanced logging."""
        logger.info("🚀 Starting FULL DAY simple pair trading monitor...")
        logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏰ Duration: {self.duration_hours} hours")
        logger.info(f"🔄 Check interval: {self.check_interval} seconds ({self.check_interval/60:.1f} minutes)")
        
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(hours=self.duration_hours)
        
        step_count = 0
        last_summary_time = self.start_time
        last_save_time = self.start_time
        
        # Log initial market conditions
        initial_prices = self.trader.get_current_prices()
        logger.info("📊 Initial market prices:")
        for pair, price in initial_prices.items():
            logger.info(f"  {pair}: ${price:,.2f}")
        
        try:
            while self.running and datetime.now() < end_time:
                step_count += 1
                current_time = datetime.now()
                elapsed_hours = (current_time - self.start_time).total_seconds() / 3600
                remaining_hours = self.duration_hours - elapsed_hours
                
                logger.info(f"=== Step {step_count} - {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                logger.info(f"⏱️  Elapsed: {elapsed_hours:.1f}h, Remaining: {remaining_hours:.1f}h")
                
                # Run strategy step
                step_result = self.trader.run_strategy_step()
                
                if step_result:
                    portfolio_value = step_result['portfolio_value']
                    initial_value = self.trader.initial_balance
                    total_return = (portfolio_value - initial_value) / initial_value * 100
                    
                    logger.info(f"💰 Portfolio: ${portfolio_value:.2f} (Return: {total_return:+.2f}%)")
                    logger.info(f"💵 CAD balance: ${step_result['cad_balance']:.2f}")
                    
                    if step_result['trades_executed'] > 0:
                        logger.info(f"🔥 Trades executed: {step_result['trades_executed']}")
                    
                    # Log current prices and positions
                    logger.info("📈 Current market:")
                    for pair, price in step_result['prices'].items():
                        position_amount = step_result['positions'].get(pair, 0)
                        if position_amount > 0:
                            position_value = position_amount * price
                            logger.info(f"  {pair}: ${price:,.2f} (Position: {position_amount:.6f} = ${position_value:.2f})")
                        else:
                            logger.info(f"  {pair}: ${price:,.2f} (No position)")
                
                # Print detailed summary every 2 hours
                if (current_time - last_summary_time).total_seconds() >= 7200:  # 2 hours
                    self._print_detailed_summary(step_count, elapsed_hours)
                    last_summary_time = current_time
                
                # Save data every hour
                if (current_time - last_save_time).total_seconds() >= 3600:  # 1 hour
                    self.trader.save_data(str(self.data_dir))
                    logger.info(f"💾 Data saved to {self.data_dir}")
                    last_save_time = current_time
                
                # Wait for next check
                if self.running and datetime.now() < end_time:
                    logger.info(f"⏳ Waiting {self.check_interval} seconds until next check...")
                    time.sleep(self.check_interval)
            
            # Final summary and data save
            logger.info("🏁 FULL DAY MONITORING COMPLETED!")
            self._print_final_detailed_summary()
            self.trader.save_data(str(self.data_dir))
            self._generate_comprehensive_report()
            
        except KeyboardInterrupt:
            logger.info("⚠️  Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in monitoring loop: {e}")
        finally:
            # Always save data on exit
            self.trader.save_data(str(self.data_dir))
            logger.info(f"💾 Final data saved to {self.data_dir}")
    
    def _print_detailed_summary(self, step_count, elapsed_hours):
        """Print detailed performance summary."""
        summary = self.trader.get_performance_summary()
        
        if summary:
            logger.info("=" * 50)
            logger.info("📊 DETAILED PERFORMANCE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"⏱️  Runtime: {elapsed_hours:.1f} hours ({step_count} checks)")
            logger.info(f"💰 Initial balance: ${summary['initial_balance']:.2f}")
            logger.info(f"💰 Current value: ${summary['current_value']:.2f}")
            logger.info(f"📈 Total return: {summary['total_return_pct']:.2f}% (${summary['total_return_cad']:+.2f})")
            logger.info(f"🔄 Total trades: {summary['total_trades']}")
            logger.info(f"💸 Total fees paid: ${summary['total_fees_paid']:.2f}")
            logger.info(f"💵 Realized P&L: ${summary['total_realized_pnl']:+.2f}")
            logger.info(f"📊 Unrealized P&L: ${summary['total_unrealized_pnl']:+.2f}")
            logger.info(f"💵 CAD balance: ${summary['cad_balance']:.2f}")
            
            logger.info("📋 Current positions:")
            for pair, pos_info in summary['positions'].items():
                if pos_info['amount'] > 0:
                    logger.info(f"  {pair}: {pos_info['amount']:.6f} @ ${pos_info['avg_buy_price']:.2f}")
                    logger.info(f"    Unrealized P&L: ${pos_info['unrealized_pnl']:+.2f}")
                else:
                    logger.info(f"  {pair}: No position (Realized P&L: ${pos_info['realized_pnl']:+.2f})")
            
            # Calculate performance metrics
            if elapsed_hours > 0:
                hourly_return = summary['total_return_pct'] / elapsed_hours
                daily_return_projection = hourly_return * 24
                logger.info(f"📊 Hourly return rate: {hourly_return:.4f}%")
                logger.info(f"📊 Daily return projection: {daily_return_projection:.2f}%")
            
            logger.info("=" * 50)
    
    def _print_final_detailed_summary(self):
        """Print final comprehensive summary."""
        runtime = datetime.now() - self.start_time
        
        logger.info("🏆 FINAL COMPREHENSIVE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📅 Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total runtime: {runtime}")
        logger.info(f"📁 Data directory: {self.data_dir}")
        
        self._print_summary()
        
        # Additional detailed statistics
        if self.trader.trades:
            buy_trades = [t for t in self.trader.trades if t.side == 'buy']
            sell_trades = [t for t in self.trader.trades if t.side == 'sell']
            
            logger.info("📊 TRADING STATISTICS:")
            logger.info(f"  Buy trades: {len(buy_trades)}")
            logger.info(f"  Sell trades: {len(sell_trades)}")
            
            if buy_trades:
                avg_buy_price = sum(t.price for t in buy_trades) / len(buy_trades)
                logger.info(f"  Average buy price: ${avg_buy_price:.2f}")
            
            if sell_trades:
                avg_sell_price = sum(t.price for t in sell_trades) / len(sell_trades)
                logger.info(f"  Average sell price: ${avg_sell_price:.2f}")
            
            # Trade frequency
            total_hours = runtime.total_seconds() / 3600
            if total_hours > 0:
                trades_per_hour = len(self.trader.trades) / total_hours
                logger.info(f"  Trade frequency: {trades_per_hour:.2f} trades/hour")
        
        logger.info("=" * 60)
    
    def _generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        runtime = datetime.now() - self.start_time
        
        # Base report
        super()._generate_report()
        
        # Add comprehensive analysis
        analysis = {
            'runtime_analysis': {
                'total_runtime_seconds': runtime.total_seconds(),
                'total_runtime_hours': runtime.total_seconds() / 3600,
                'checks_performed': len(self.trader.portfolio_history),
                'average_check_interval': runtime.total_seconds() / len(self.trader.portfolio_history) if self.trader.portfolio_history else 0
            },
            'market_analysis': {
                'price_volatility': self._calculate_price_volatility(),
                'trading_opportunities': self._analyze_trading_opportunities(),
                'market_conditions': self._analyze_market_conditions()
            },
            'strategy_performance': {
                'win_rate': self._calculate_win_rate(),
                'average_trade_size': self._calculate_average_trade_size(),
                'risk_metrics': self._calculate_risk_metrics()
            }
        }
        
        # Save comprehensive analysis
        with open(self.data_dir / 'comprehensive_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"📊 Comprehensive analysis saved to {self.data_dir / 'comprehensive_analysis.json'}")
    
    def _calculate_price_volatility(self):
        """Calculate price volatility metrics."""
        if not self.trader.price_history:
            return {}
        
        volatility = {}
        for pair, price_data in self.trader.price_history.items():
            if len(price_data) > 1:
                prices = [p['price'] for p in price_data]
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                
                volatility[pair] = {
                    'min_price': min(prices),
                    'max_price': max(prices),
                    'price_range_pct': (max(prices) - min(prices)) / min(prices) * 100,
                    'average_return': sum(returns) / len(returns) if returns else 0,
                    'volatility_std': (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns))**0.5 if returns else 0
                }
        
        return volatility
    
    def _analyze_trading_opportunities(self):
        """Analyze trading opportunities and signals."""
        total_checks = len(self.trader.portfolio_history)
        total_trades = len(self.trader.trades)
        
        return {
            'total_checks': total_checks,
            'total_trades': total_trades,
            'trade_frequency': total_trades / total_checks if total_checks > 0 else 0,
            'trades_per_pair': {
                pair: len([t for t in self.trader.trades if t.pair == pair])
                for pair in self.trader.pairs
            }
        }
    
    def _analyze_market_conditions(self):
        """Analyze overall market conditions during the monitoring period."""
        if not self.trader.portfolio_history:
            return {}
        
        first_entry = self.trader.portfolio_history[0]
        last_entry = self.trader.portfolio_history[-1]
        
        market_movement = {}
        for pair in self.trader.pairs:
            if pair in first_entry['prices'] and pair in last_entry['prices']:
                start_price = first_entry['prices'][pair]
                end_price = last_entry['prices'][pair]
                movement = (end_price - start_price) / start_price * 100
                market_movement[pair] = {
                    'start_price': start_price,
                    'end_price': end_price,
                    'total_movement_pct': movement
                }
        
        return market_movement
    
    def _calculate_win_rate(self):
        """Calculate win rate of trades."""
        if not self.trader.trades:
            return 0
        
        profitable_trades = 0
        total_completed_trades = 0
        
        # Group trades by pair to match buy/sell pairs
        for pair in self.trader.pairs:
            pair_trades = [t for t in self.trader.trades if t.pair == pair]
            buy_trades = [t for t in pair_trades if t.side == 'buy']
            sell_trades = [t for t in pair_trades if t.side == 'sell']
            
            # Match buy/sell pairs
            for sell_trade in sell_trades:
                # Find corresponding buy trade (simplified matching)
                for buy_trade in buy_trades:
                    if buy_trade.timestamp < sell_trade.timestamp:
                        profit = sell_trade.cost - sell_trade.fee - buy_trade.cost - buy_trade.fee
                        if profit > 0:
                            profitable_trades += 1
                        total_completed_trades += 1
                        break
        
        return profitable_trades / total_completed_trades if total_completed_trades > 0 else 0
    
    def _calculate_average_trade_size(self):
        """Calculate average trade size."""
        if not self.trader.trades:
            return 0
        
        total_cost = sum(t.cost for t in self.trader.trades)
        return total_cost / len(self.trader.trades)
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics."""
        if not self.trader.portfolio_history:
            return {}
        
        portfolio_values = [entry['portfolio_value'] for entry in self.trader.portfolio_history]
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        
        if not returns:
            return {}
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return)**2 for r in returns) / len(returns)
        volatility = variance**0.5
        
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': avg_return / volatility if volatility > 0 else 0,
            'max_portfolio_value': max(portfolio_values),
            'min_portfolio_value': min(portfolio_values)
        }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Day Simple Pair Trading Monitor")
    parser.add_argument('--duration', type=float, default=24, 
                       help='Duration to run in hours (default: 24)')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Check interval in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--test', action='store_true', 
                       help='Run a short test (1 hour, 2-minute intervals)')
    
    args = parser.parse_args()
    
    if args.test:
        duration = 1      # 1 hour
        interval = 120    # 2 minutes
        print("🧪 Running in TEST mode (1 hour, 2-minute intervals)")
    else:
        duration = args.duration
        interval = args.interval
        print(f"🚀 Running FULL DAY monitor ({duration} hours, {interval/60:.1f}-minute intervals)")
    
    monitor = FullDayMonitor(duration_hours=duration, check_interval=interval)
    monitor.run()

if __name__ == "__main__":
    main()