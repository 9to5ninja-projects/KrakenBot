"""
Demo trading script with very sensitive parameters to generate trades for dashboard demonstration.
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

def run_demo_trading():
    """Run demo trading with very sensitive parameters to generate trades."""
    
    logger.info("🎯 DEMO TRADING SESSION - Generating trades for dashboard")
    
    # Create trader with very sensitive parameters for demo
    trader = SimplePairTrader(
        pairs=['ETH/CAD', 'BTC/CAD'],
        initial_balance=100.0
    )
    
    # Override with very sensitive thresholds for demo
    trader.buy_threshold = -0.002   # Buy on 0.2% drop (very sensitive)
    trader.sell_threshold = 0.003   # Sell on 0.3% rise (very sensitive)
    trader.lookback_periods = 3     # Very short lookback
    trader.min_trade_amount = 5.0   # Lower minimum for more trades
    trader.max_position_size = 0.4  # Allow larger positions
    
    logger.info("Demo parameters:")
    logger.info(f"  Buy threshold: {trader.buy_threshold*100:.1f}%")
    logger.info(f"  Sell threshold: {trader.sell_threshold*100:.1f}%")
    logger.info(f"  Lookback periods: {trader.lookback_periods}")
    logger.info(f"  Min trade amount: ${trader.min_trade_amount}")
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir = Path("data") / f"demo_trading_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run for 20 steps (about 10 minutes with delays)
    for step in range(20):
        logger.info(f"=== Demo Step {step + 1}/20 ===")
        
        # Get real prices
        real_prices = trader.get_current_prices()
        
        # Add small artificial volatility to trigger trades
        import random
        simulated_prices = {}
        for pair, price in real_prices.items():
            # Add small random movement
            volatility = random.uniform(-0.005, 0.005)  # ±0.5%
            simulated_prices[pair] = price * (1 + volatility)
            logger.info(f"{pair}: Real: ${price:.2f}, Simulated: ${simulated_prices[pair]:.2f}")
        
        # Update price history with simulated prices
        trader.update_price_history(simulated_prices)
        
        # Check for trading signals
        trades_executed = 0
        for pair in trader.pairs:
            should_buy, should_sell, reason = trader.calculate_signals(pair)
            
            logger.info(f"{pair}: {reason}")
            
            if should_buy:
                amount = trader.calculate_trade_amount(pair, 'buy', simulated_prices[pair])
                if amount > 0:
                    if trader.execute_trade(pair, 'buy', amount, simulated_prices[pair], reason):
                        trades_executed += 1
                        logger.info(f"✅ BUY executed: {amount:.6f} {pair.split('/')[0]} at ${simulated_prices[pair]:.2f}")
            
            elif should_sell:
                amount = trader.calculate_trade_amount(pair, 'sell', simulated_prices[pair])
                if amount > 0:
                    if trader.execute_trade(pair, 'sell', amount, simulated_prices[pair], reason):
                        trades_executed += 1
                        logger.info(f"✅ SELL executed: {amount:.6f} {pair.split('/')[0]} at ${simulated_prices[pair]:.2f}")
        
        # Update portfolio
        portfolio_value = trader.get_portfolio_value(simulated_prices)
        total_return = (portfolio_value - trader.initial_balance) / trader.initial_balance * 100
        
        logger.info(f"Portfolio: ${portfolio_value:.2f} (Return: {total_return:+.2f}%), Trades: {trades_executed}")
        
        # Save data every 5 steps
        if step % 5 == 0:
            trader.save_data(str(data_dir))
        
        # Short delay
        time.sleep(2)
    
    # Final save
    trader.save_data(str(data_dir))
    
    # Print summary
    summary = trader.get_performance_summary()
    
    logger.info("🎯 DEMO TRADING COMPLETED!")
    logger.info(f"📁 Data saved to: {data_dir}")
    logger.info(f"💰 Initial balance: ${summary['initial_balance']:.2f}")
    logger.info(f"💰 Final value: ${summary['current_value']:.2f}")
    logger.info(f"📈 Total return: {summary['total_return_pct']:.2f}%")
    logger.info(f"🔄 Total trades: {summary['total_trades']}")
    logger.info(f"💸 Total fees: ${summary['total_fees_paid']:.2f}")
    
    if summary['total_trades'] > 0:
        logger.info("✅ SUCCESS: Trades were generated for dashboard demonstration!")
    else:
        logger.info("ℹ️  No trades generated - try running again for different market conditions")
    
    return data_dir

if __name__ == "__main__":
    run_demo_trading()