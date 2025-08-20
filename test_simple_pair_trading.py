"""
Test script for simple pair trading with more sensitive parameters for demonstration.
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from simple_pair_trader import SimplePairTrader
import config

def run_test_trading(duration_minutes=30, check_interval=60):
    """Run a test trading session with more sensitive parameters."""
    
    # Create a custom trader with more sensitive thresholds for testing
    trader = SimplePairTrader()
    
    # Override thresholds for testing (more sensitive)
    trader.buy_threshold = -0.005   # Buy when price drops 0.5% from recent high
    trader.sell_threshold = 0.01    # Sell when price rises 1% from recent low
    trader.lookback_periods = 5     # Shorter lookback for more frequent signals
    trader.min_trade_amount = 10.0  # Lower minimum trade amount
    
    logger.info("=== TEST TRADING SESSION ===")
    logger.info(f"Duration: {duration_minutes} minutes")
    logger.info(f"Check interval: {check_interval} seconds")
    logger.info(f"Buy threshold: {trader.buy_threshold*100:.1f}%")
    logger.info(f"Sell threshold: {trader.sell_threshold*100:.1f}%")
    logger.info(f"Lookback periods: {trader.lookback_periods}")
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir = Path("data") / f"test_simple_pair_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    step_count = 0
    
    try:
        while datetime.now() < end_time:
            step_count += 1
            current_time = datetime.now()
            
            logger.info(f"=== Step {step_count} - {current_time.strftime('%H:%M:%S')} ===")
            
            # Run strategy step
            step_result = trader.run_strategy_step()
            
            if step_result:
                logger.info(f"Portfolio value: ${step_result['portfolio_value']:.2f}")
                logger.info(f"CAD balance: ${step_result['cad_balance']:.2f}")
                logger.info(f"Trades executed: {step_result['trades_executed']}")
                
                # Log current prices and positions
                for pair, price in step_result['prices'].items():
                    position_amount = step_result['positions'].get(pair, 0)
                    if position_amount > 0:
                        logger.info(f"  {pair}: ${price:.2f} (Position: {position_amount:.6f})")
                    else:
                        logger.info(f"  {pair}: ${price:.2f} (No position)")
                
                # Log any trades executed
                if step_result['trades_executed'] > 0:
                    logger.info("🔥 TRADES EXECUTED!")
            
            # Save data every 5 steps
            if step_count % 5 == 0:
                trader.save_data(str(data_dir))
                logger.info(f"Data saved to {data_dir}")
            
            # Wait for next check
            if datetime.now() < end_time:
                logger.info(f"Waiting {check_interval} seconds...")
                time.sleep(check_interval)
        
        # Final save and summary
        trader.save_data(str(data_dir))
        
        # Print final summary
        summary = trader.get_performance_summary()
        
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Runtime: {datetime.now() - start_time}")
        logger.info(f"Initial balance: ${summary['initial_balance']:.2f}")
        logger.info(f"Final value: ${summary['current_value']:.2f}")
        logger.info(f"Total return: {summary['total_return_pct']:.2f}%")
        logger.info(f"Total trades: {summary['total_trades']}")
        logger.info(f"Total fees: ${summary['total_fees_paid']:.2f}")
        
        if summary['total_trades'] > 0:
            logger.info("✅ Trades were executed during the test!")
        else:
            logger.info("ℹ️ No trades executed - market conditions were stable")
        
        logger.info(f"Data saved to: {data_dir}")
        
        return data_dir
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        trader.save_data(str(data_dir))
        return data_dir
    except Exception as e:
        logger.error(f"Error during test: {e}")
        trader.save_data(str(data_dir))
        return data_dir

def simulate_volatile_market():
    """Simulate trading with artificial price volatility for demonstration."""
    
    logger.info("=== SIMULATED VOLATILE MARKET TEST ===")
    
    # Create trader with very sensitive settings
    trader = SimplePairTrader()
    trader.buy_threshold = -0.001   # Buy on 0.1% drop
    trader.sell_threshold = 0.002   # Sell on 0.2% rise
    trader.lookback_periods = 3     # Very short lookback
    trader.min_trade_amount = 5.0   # Very low minimum
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir = Path("data") / f"simulated_volatile_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get initial prices
    initial_prices = trader.get_current_prices()
    
    # Simulate 20 steps with artificial price movements
    for step in range(20):
        logger.info(f"=== Simulation Step {step + 1} ===")
        
        # Get real prices as base
        real_prices = trader.get_current_prices()
        
        # Add artificial volatility
        simulated_prices = {}
        for pair, price in real_prices.items():
            # Add random volatility between -1% and +1%
            import random
            volatility = random.uniform(-0.01, 0.01)
            simulated_prices[pair] = price * (1 + volatility)
        
        # Manually update price history with simulated prices
        trader.update_price_history(simulated_prices)
        
        # Check for signals and execute trades
        trades_executed = 0
        for pair in trader.pairs:
            should_buy, should_sell, reason = trader.calculate_signals(pair)
            
            logger.info(f"{pair}: {reason}")
            
            if should_buy:
                amount = trader.calculate_trade_amount(pair, 'buy', simulated_prices[pair])
                if amount > 0:
                    if trader.execute_trade(pair, 'buy', amount, simulated_prices[pair], reason):
                        trades_executed += 1
            
            elif should_sell:
                amount = trader.calculate_trade_amount(pair, 'sell', simulated_prices[pair])
                if amount > 0:
                    if trader.execute_trade(pair, 'sell', amount, simulated_prices[pair], reason):
                        trades_executed += 1
        
        # Update portfolio value
        portfolio_value = trader.get_portfolio_value(simulated_prices)
        logger.info(f"Portfolio value: ${portfolio_value:.2f}, Trades: {trades_executed}")
        
        # Small delay
        time.sleep(1)
    
    # Save final data
    trader.save_data(str(data_dir))
    
    # Print summary
    summary = trader.get_performance_summary()
    
    logger.info("=== SIMULATION SUMMARY ===")
    logger.info(f"Initial balance: ${summary['initial_balance']:.2f}")
    logger.info(f"Final value: ${summary['current_value']:.2f}")
    logger.info(f"Total return: {summary['total_return_pct']:.2f}%")
    logger.info(f"Total trades: {summary['total_trades']}")
    logger.info(f"Data saved to: {data_dir}")
    
    return data_dir

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Simple Pair Trading")
    parser.add_argument('--mode', choices=['real', 'simulate'], default='real',
                       help='Test mode: real market data or simulated volatility')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in minutes for real mode (default: 30)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds for real mode (default: 60)')
    
    args = parser.parse_args()
    
    if args.mode == 'simulate':
        data_dir = simulate_volatile_market()
    else:
        data_dir = run_test_trading(args.duration, args.interval)
    
    print(f"\n🎉 Test completed! Data saved to: {data_dir}")
    print(f"📊 Run the enhanced dashboard to view results:")
    print(f"   streamlit run enhanced_dashboard.py")

if __name__ == "__main__":
    main()