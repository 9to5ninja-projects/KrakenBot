"""
Direct test script for KrakenBot development and analysis.
Provides comprehensive testing and analysis capabilities.
"""
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Import the necessary modules
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator
from monitor import ArbitrageMonitor
import config

class KrakenBotTester:
    """Comprehensive testing and analysis tool for KrakenBot."""
    
    def __init__(self):
        """Initialize the tester."""
        self.exchange = ExchangeManager(use_api_keys=False)
        self.calculator = ArbitrageCalculator(self.exchange)
        self.results = []
        
    def test_single_opportunity(self, verbose=True):
        """Test a single arbitrage opportunity calculation."""
        if verbose:
            print("=== Testing Single Arbitrage Opportunity ===")
        
        try:
            # Get current prices
            prices = self.calculator.get_current_prices()
            if verbose:
                print("Current prices:")
                for pair, price in prices.items():
                    print(f"  {pair}: {price}")
            
            # Find opportunities
            opportunities = self.calculator.find_opportunities()
            
            if verbose:
                print(f"\nFound {len(opportunities)} opportunities:")
                for i, opp in enumerate(opportunities, 1):
                    status = "✅ PROFITABLE" if opp.is_profitable else "❌ Not Profitable"
                    print(f"{i}. {status}")
                    print(f"   Path: {' → '.join(opp.path)}")
                    print(f"   Profit: {opp.profit_percentage:.4f}%")
                    print(f"   Amount: {opp.start_amount:.2f} → {opp.end_amount:.2f}")
            
            return opportunities
            
        except Exception as e:
            print(f"Error testing single opportunity: {e}")
            return []
    
    def test_custom_path(self, path, amount=None, verbose=True):
        """Test a custom arbitrage path."""
        if amount is None:
            amount = config.START_AMOUNT
            
        if verbose:
            print(f"=== Testing Custom Path: {' → '.join(path)} ===")
        
        try:
            # Get current prices
            prices = self.calculator.get_current_prices()
            
            # Calculate opportunity for custom path
            opportunity = self.calculator.calculate_custom_path(path, prices)
            
            if verbose:
                status = "✅ PROFITABLE" if opportunity.is_profitable else "❌ Not Profitable"
                print(f"Result: {status}")
                print(f"Path: {' → '.join(opportunity.path)}")
                print(f"Profit: {opportunity.profit_percentage:.4f}%")
                print(f"Amount: {opportunity.start_amount:.2f} → {opportunity.end_amount:.2f}")
                print(f"Absolute profit: {opportunity.profit:.2f}")
            
            return opportunity
            
        except Exception as e:
            print(f"Error testing custom path: {e}")
            return None
    
    def monitor_opportunities(self, duration_minutes=5, interval_seconds=30):
        """Monitor opportunities for a specified duration."""
        print(f"=== Monitoring Opportunities for {duration_minutes} minutes ===")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        opportunities_log = []
        
        try:
            while datetime.now() < end_time:
                current_time = datetime.now()
                print(f"\n[{current_time.strftime('%H:%M:%S')}] Checking opportunities...")
                
                opportunities = self.test_single_opportunity(verbose=False)
                
                # Log all opportunities
                for opp in opportunities:
                    opp_data = {
                        'timestamp': current_time.isoformat(),
                        'path': ' → '.join(opp.path),
                        'profit_percentage': opp.profit_percentage,
                        'is_profitable': opp.is_profitable,
                        'start_amount': opp.start_amount,
                        'end_amount': opp.end_amount
                    }
                    opportunities_log.append(opp_data)
                
                # Show best opportunity
                if opportunities:
                    best_opp = max(opportunities, key=lambda x: x.profit_percentage)
                    status = "✅ PROFITABLE" if best_opp.is_profitable else "❌ Not Profitable"
                    print(f"Best: {status} - {' → '.join(best_opp.path)} - {best_opp.profit_percentage:.4f}%")
                else:
                    print("No opportunities found")
                
                # Wait for next check
                if datetime.now() < end_time:
                    print(f"Waiting {interval_seconds} seconds...")
                    time.sleep(interval_seconds)
            
            # Save results
            timestamp = start_time.strftime("%Y-%m-%d_%H-%M")
            results_dir = Path("data") / f"monitor_test_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save opportunities log
            with open(results_dir / "opportunities.json", "w") as f:
                json.dump(opportunities_log, f, indent=2)
            
            # Generate summary
            if opportunities_log:
                df = pd.DataFrame(opportunities_log)
                summary = {
                    'total_checks': len(df),
                    'profitable_count': len(df[df['is_profitable'] == True]),
                    'best_profit': df['profit_percentage'].max(),
                    'worst_profit': df['profit_percentage'].min(),
                    'average_profit': df['profit_percentage'].mean(),
                    'paths_tested': df['path'].unique().tolist()
                }
                
                with open(results_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n=== Monitoring Summary ===")
                print(f"Total checks: {summary['total_checks']}")
                print(f"Profitable opportunities: {summary['profitable_count']}")
                print(f"Best profit: {summary['best_profit']:.4f}%")
                print(f"Worst profit: {summary['worst_profit']:.4f}%")
                print(f"Average profit: {summary['average_profit']:.4f}%")
                print(f"Results saved to: {results_dir}")
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")
    
    def test_all_possible_paths(self):
        """Test all possible triangular arbitrage paths."""
        print("=== Testing All Possible Paths ===")
        
        try:
            # Get available markets
            markets = self.exchange.get_markets()
            
            # Extract all currencies
            currencies = set()
            available_pairs = set()
            
            for market in markets:
                if market['active'] and market['type'] == 'spot':
                    symbol = market['symbol']
                    base, quote = symbol.split('/')
                    currencies.add(base)
                    currencies.add(quote)
                    available_pairs.add(symbol)
            
            print(f"Found {len(currencies)} currencies and {len(available_pairs)} pairs")
            
            # Focus on major currencies
            major_currencies = {'BTC', 'ETH', 'CAD', 'USD', 'EUR', 'USDT', 'USDC'}
            test_currencies = currencies.intersection(major_currencies)
            
            print(f"Testing with major currencies: {', '.join(sorted(test_currencies))}")
            
            # Test triangular paths
            profitable_paths = []
            all_results = []
            
            for base in test_currencies:
                for c1 in test_currencies:
                    for c2 in test_currencies:
                        if len({base, c1, c2}) == 3:  # All different currencies
                            path = [base, c1, c2, base]
                            
                            # Check if all required pairs exist
                            pair1 = f"{base}/{c1}" in available_pairs or f"{c1}/{base}" in available_pairs
                            pair2 = f"{c1}/{c2}" in available_pairs or f"{c2}/{c1}" in available_pairs
                            pair3 = f"{c2}/{base}" in available_pairs or f"{base}/{c2}" in available_pairs
                            
                            if pair1 and pair2 and pair3:
                                try:
                                    opportunity = self.test_custom_path(path, verbose=False)
                                    if opportunity:
                                        result = {
                                            'path': ' → '.join(path),
                                            'profit_percentage': opportunity.profit_percentage,
                                            'is_profitable': opportunity.is_profitable,
                                            'start_amount': opportunity.start_amount,
                                            'end_amount': opportunity.end_amount
                                        }
                                        all_results.append(result)
                                        
                                        if opportunity.is_profitable:
                                            profitable_paths.append(result)
                                            print(f"✅ PROFITABLE: {' → '.join(path)} - {opportunity.profit_percentage:.4f}%")
                                        
                                        # Add delay to avoid rate limits
                                        time.sleep(2)
                                        
                                except Exception as e:
                                    print(f"Error testing path {' → '.join(path)}: {e}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            results_dir = Path("data") / f"all_paths_test_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            
            if profitable_paths:
                with open(results_dir / "profitable_paths.json", "w") as f:
                    json.dump(profitable_paths, f, indent=2)
            
            print(f"\n=== Path Testing Summary ===")
            print(f"Total paths tested: {len(all_results)}")
            print(f"Profitable paths found: {len(profitable_paths)}")
            
            if all_results:
                df = pd.DataFrame(all_results)
                print(f"Best profit: {df['profit_percentage'].max():.4f}%")
                print(f"Worst profit: {df['profit_percentage'].min():.4f}%")
                print(f"Average profit: {df['profit_percentage'].mean():.4f}%")
            
            print(f"Results saved to: {results_dir}")
            
        except Exception as e:
            print(f"Error testing all paths: {e}")
    
    def analyze_market_conditions(self):
        """Analyze current market conditions."""
        print("=== Market Conditions Analysis ===")
        
        try:
            # Get current prices
            prices = self.calculator.get_current_prices()
            
            print("Current Market Prices:")
            for pair, price in prices.items():
                print(f"  {pair}: {price:,.2f}")
            
            # Calculate spreads and volatility indicators
            print("\nMarket Analysis:")
            
            # BTC/ETH ratio analysis
            if 'BTC/CAD' in prices and 'ETH/CAD' in prices:
                btc_cad = prices['BTC/CAD']
                eth_cad = prices['ETH/CAD']
                btc_eth_ratio = btc_cad / eth_cad
                print(f"  BTC/ETH ratio: {btc_eth_ratio:.4f}")
            
            if 'ETH/BTC' in prices:
                eth_btc_direct = prices['ETH/BTC']
                print(f"  ETH/BTC direct: {eth_btc_direct:.6f}")
                
                if 'BTC/CAD' in prices and 'ETH/CAD' in prices:
                    eth_btc_calculated = eth_cad / btc_cad
                    spread = abs(eth_btc_direct - eth_btc_calculated) / eth_btc_direct * 100
                    print(f"  ETH/BTC calculated: {eth_btc_calculated:.6f}")
                    print(f"  ETH/BTC spread: {spread:.4f}%")
            
            # Fee analysis
            print(f"\nFee Structure:")
            print(f"  Taker fee: {config.TAKER_FEE * 100:.2f}%")
            print(f"  Slippage buffer: {config.SLIPPAGE_BUFFER * 100:.2f}%")
            print(f"  Total cost per trade: {(config.TAKER_FEE + config.SLIPPAGE_BUFFER) * 100:.2f}%")
            print(f"  Breakeven threshold: {config.BREAKEVEN_PCT:.2f}%")
            print(f"  Minimum profit target: {config.MIN_PROFIT_PCT:.2f}%")
            
        except Exception as e:
            print(f"Error analyzing market conditions: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="KrakenBot Direct Testing Tool")
    parser.add_argument('command', choices=['test', 'monitor', 'paths', 'market', 'custom'], 
                       help='Command to execute')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Duration for monitoring in minutes (default: 5)')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Interval between checks in seconds (default: 30)')
    parser.add_argument('--path', type=str, 
                       help='Custom path for testing (e.g., "CAD,ETH,BTC,CAD")')
    parser.add_argument('--amount', type=float, default=100, 
                       help='Amount to test with (default: 100)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = KrakenBotTester()
    
    try:
        if args.command == 'test':
            # Test single opportunity
            tester.test_single_opportunity()
            
        elif args.command == 'monitor':
            # Monitor opportunities
            tester.monitor_opportunities(args.duration, args.interval)
            
        elif args.command == 'paths':
            # Test all possible paths
            tester.test_all_possible_paths()
            
        elif args.command == 'market':
            # Analyze market conditions
            tester.analyze_market_conditions()
            
        elif args.command == 'custom':
            # Test custom path
            if not args.path:
                print("Error: --path is required for custom command")
                print("Example: python direct_test.py custom --path CAD,ETH,BTC,CAD")
                return
            
            path = args.path.split(',')
            if len(path) < 3:
                print("Error: Path must have at least 3 currencies")
                return
            
            tester.test_custom_path(path, args.amount)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("=== KrakenBot Direct Testing Tool ===")
        print("Available commands:")
        print("  test    - Test single arbitrage opportunity")
        print("  monitor - Monitor opportunities over time")
        print("  paths   - Test all possible triangular paths")
        print("  market  - Analyze current market conditions")
        print("  custom  - Test custom arbitrage path")
        print("\nExamples:")
        print("  python direct_test.py test")
        print("  python direct_test.py monitor --duration 10 --interval 20")
        print("  python direct_test.py custom --path CAD,ETH,BTC,CAD")
        print("  python direct_test.py market")
        
        # Run a quick test by default
        print("\nRunning quick test...")
        tester = KrakenBotTester()
        tester.test_single_opportunity()
    else:
        main()