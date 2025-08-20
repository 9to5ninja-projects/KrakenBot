"""
Generate sample trading data for dashboard demonstration.
"""
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

def generate_sample_data():
    """Generate sample simple pair trading data."""
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir = Path("data") / f"sample_simple_pair_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample trades
    trades = []
    portfolio_history = []
    
    start_time = datetime.now() - timedelta(hours=2)
    current_time = start_time
    balance = 100.0
    eth_position = 0.0
    btc_position = 0.0
    eth_avg_price = 0.0
    btc_avg_price = 0.0
    
    # Initial prices
    eth_price = 5950.0
    btc_price = 157000.0
    
    for i in range(60):  # 60 data points over 2 hours
        current_time += timedelta(minutes=2)
        
        # Simulate price movements
        eth_price *= (1 + random.uniform(-0.01, 0.01))  # ±1% movement
        btc_price *= (1 + random.uniform(-0.01, 0.01))  # ±1% movement
        
        # Simulate some trades
        if random.random() < 0.1:  # 10% chance of trade
            if random.random() < 0.5 and balance > 10:  # Buy
                pair = random.choice(['ETH/CAD', 'BTC/CAD'])
                if pair == 'ETH/CAD':
                    amount = 10 / eth_price
                    cost = 10.0
                    fee = cost * 0.004
                    balance -= (cost + fee)
                    eth_position += amount
                    eth_avg_price = eth_price
                    
                    trades.append({
                        'timestamp': current_time.isoformat(),
                        'pair': pair,
                        'side': 'buy',
                        'amount': amount,
                        'price': eth_price,
                        'cost': cost,
                        'fee': fee,
                        'portfolio_value_before': balance + cost + fee,
                        'portfolio_value_after': balance + eth_position * eth_price + btc_position * btc_price,
                        'reason': 'Sample buy signal'
                    })
                else:  # BTC/CAD
                    amount = 10 / btc_price
                    cost = 10.0
                    fee = cost * 0.004
                    balance -= (cost + fee)
                    btc_position += amount
                    btc_avg_price = btc_price
                    
                    trades.append({
                        'timestamp': current_time.isoformat(),
                        'pair': pair,
                        'side': 'buy',
                        'amount': amount,
                        'price': btc_price,
                        'cost': cost,
                        'fee': fee,
                        'portfolio_value_before': balance + cost + fee,
                        'portfolio_value_after': balance + eth_position * eth_price + btc_position * btc_price,
                        'reason': 'Sample buy signal'
                    })
            
            elif random.random() < 0.3:  # Sell
                if eth_position > 0 and random.random() < 0.5:
                    # Sell ETH
                    amount = eth_position
                    proceeds = amount * eth_price
                    fee = proceeds * 0.004
                    balance += (proceeds - fee)
                    eth_position = 0
                    
                    trades.append({
                        'timestamp': current_time.isoformat(),
                        'pair': 'ETH/CAD',
                        'side': 'sell',
                        'amount': amount,
                        'price': eth_price,
                        'cost': proceeds,
                        'fee': fee,
                        'portfolio_value_before': balance - proceeds + fee + amount * eth_price,
                        'portfolio_value_after': balance + btc_position * btc_price,
                        'reason': 'Sample sell signal'
                    })
                
                elif btc_position > 0:
                    # Sell BTC
                    amount = btc_position
                    proceeds = amount * btc_price
                    fee = proceeds * 0.004
                    balance += (proceeds - fee)
                    btc_position = 0
                    
                    trades.append({
                        'timestamp': current_time.isoformat(),
                        'pair': 'BTC/CAD',
                        'side': 'sell',
                        'amount': amount,
                        'price': btc_price,
                        'cost': proceeds,
                        'fee': fee,
                        'portfolio_value_before': balance - proceeds + fee + amount * btc_price,
                        'portfolio_value_after': balance + eth_position * eth_price,
                        'reason': 'Sample sell signal'
                    })
        
        # Record portfolio state
        portfolio_value = balance + eth_position * eth_price + btc_position * btc_price
        total_return = (portfolio_value - 100) / 100 * 100
        
        portfolio_history.append({
            'timestamp': current_time.isoformat(),
            'cad_balance': balance,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'positions': {
                'ETH/CAD': {
                    'base_amount': eth_position,
                    'avg_buy_price': eth_avg_price,
                    'current_price': eth_price,
                    'unrealized_pnl': (eth_price - eth_avg_price) * eth_position if eth_position > 0 else 0,
                    'realized_pnl': 0
                },
                'BTC/CAD': {
                    'base_amount': btc_position,
                    'avg_buy_price': btc_avg_price,
                    'current_price': btc_price,
                    'unrealized_pnl': (btc_price - btc_avg_price) * btc_position if btc_position > 0 else 0,
                    'realized_pnl': 0
                }
            },
            'prices': {
                'ETH/CAD': eth_price,
                'BTC/CAD': btc_price
            }
        })
    
    # Calculate final performance
    final_portfolio_value = balance + eth_position * eth_price + btc_position * btc_price
    total_fees = sum(trade['fee'] for trade in trades)
    
    performance_summary = {
        'initial_balance': 100.0,
        'current_value': final_portfolio_value,
        'total_return_pct': (final_portfolio_value - 100) / 100 * 100,
        'total_return_cad': final_portfolio_value - 100,
        'total_trades': len(trades),
        'total_fees_paid': total_fees,
        'total_realized_pnl': 0,  # Simplified for sample
        'total_unrealized_pnl': (eth_price - eth_avg_price) * eth_position + (btc_price - btc_avg_price) * btc_position,
        'cad_balance': balance,
        'positions': {
            'ETH/CAD': {
                'amount': eth_position,
                'avg_buy_price': eth_avg_price,
                'unrealized_pnl': (eth_price - eth_avg_price) * eth_position if eth_position > 0 else 0,
                'realized_pnl': 0
            },
            'BTC/CAD': {
                'amount': btc_position,
                'avg_buy_price': btc_avg_price,
                'unrealized_pnl': (btc_price - btc_avg_price) * btc_position if btc_position > 0 else 0,
                'realized_pnl': 0
            }
        }
    }
    
    # Generate price history
    price_history = {
        'ETH/CAD': [],
        'BTC/CAD': []
    }
    
    for entry in portfolio_history:
        price_history['ETH/CAD'].append({
            'timestamp': entry['timestamp'],
            'price': entry['prices']['ETH/CAD']
        })
        price_history['BTC/CAD'].append({
            'timestamp': entry['timestamp'],
            'price': entry['prices']['BTC/CAD']
        })
    
    # Save all data
    with open(data_dir / 'trades.json', 'w') as f:
        json.dump(trades, f, indent=2)
    
    with open(data_dir / 'portfolio_history.json', 'w') as f:
        json.dump(portfolio_history, f, indent=2)
    
    with open(data_dir / 'performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    with open(data_dir / 'price_history.json', 'w') as f:
        json.dump(price_history, f, indent=2)
    
    # Generate README
    readme_content = f"""# Sample Simple Pair Trading Data

This is sample data generated for dashboard demonstration.

## Summary
- **Initial Balance**: $100.00 CAD
- **Final Portfolio Value**: ${final_portfolio_value:.2f} CAD
- **Total Return**: {(final_portfolio_value - 100) / 100 * 100:.2f}%
- **Total Trades**: {len(trades)}
- **Total Fees**: ${total_fees:.2f}

## Files
- `trades.json` - All executed trades
- `portfolio_history.json` - Portfolio value over time
- `performance_summary.json` - Performance metrics
- `price_history.json` - Price data
"""
    
    with open(data_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"Sample data generated in: {data_dir}")
    print(f"Total trades: {len(trades)}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {(final_portfolio_value - 100) / 100 * 100:.2f}%")
    
    return data_dir

if __name__ == "__main__":
    generate_sample_data()