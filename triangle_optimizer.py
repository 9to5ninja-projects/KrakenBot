"""
Triangle Optimizer module for KrakenBot.
Analyzes historical data to identify the most profitable triangular arbitrage paths.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from pathlib import Path
from loguru import logger
import itertools

import config
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator

class TriangleOptimizer:
    """
    Analyzes historical data to identify optimal triangular arbitrage paths.
    """
    
    def __init__(self, exchange_manager: ExchangeManager = None):
        """
        Initialize the triangle optimizer.
        
        Args:
            exchange_manager: Exchange manager instance (optional)
        """
        self.exchange = exchange_manager or ExchangeManager(use_api_keys=False)
        self.data_dir = config.DATA_DIR / 'historical'
        self.data_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.start_amount = config.START_AMOUNT
        self.min_profit_pct = config.MIN_PROFIT_PCT
        self.maker_fee = config.MAKER_FEE
        self.taker_fee = config.TAKER_FEE
        self.slippage_buffer = config.SLIPPAGE_BUFFER
        
        # Additional trading pairs to consider
        self.additional_pairs = [
            'XRP/CAD', 'LTC/CAD', 'XLM/CAD', 'USDC/CAD', 'USDT/CAD',
            'XRP/BTC', 'LTC/BTC', 'XLM/BTC', 'XRP/ETH', 'LTC/ETH',
            'XLM/ETH', 'USDC/BTC', 'USDT/BTC', 'USDC/ETH', 'USDT/ETH'
        ]
        
        # Fee adjustment factor (includes both fee and slippage)
        self.fee_factor = 1 - self.taker_fee - self.slippage_buffer
        
        # Results storage
        self.triangle_stats = {}
        self.optimal_triangles = []
    
    def download_historical_data(self, symbol: str, start_date: str, end_date: str) -> str:
        """
        Download historical data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/CAD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Path to the saved data file
        """
        logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create filename
        symbol_file = symbol.replace('/', '_')
        filename = f"{symbol_file}_{start_date}_{end_date}.csv"
        file_path = self.data_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"Historical data file already exists: {file_path}")
            return str(file_path)
        
        # Download data
        try:
            # Get OHLCV data (Open, High, Low, Close, Volume)
            data = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
                logger.info(f"Fetching data for {symbol} at {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get daily data
                daily_data = self.exchange.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe='1d',
                    since=int(current_dt.timestamp() * 1000)
                )
                
                if daily_data:
                    data.extend(daily_data)
                    # Move to next day
                    current_dt += timedelta(days=1)
                else:
                    # If no data, move to next day anyway to avoid infinite loop
                    logger.warning(f"No data available for {symbol} on {current_dt.strftime('%Y-%m-%d')}")
                    current_dt += timedelta(days=1)
            
            if not data:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Historical data saved to {file_path}")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading historical data for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """
        Get all available trading pairs from the exchange.
        
        Returns:
            List of trading pair symbols
        """
        try:
            markets = self.exchange.get_markets()
            symbols = [market['symbol'] for market in markets if market.get('active', True)]
            
            # Add default pairs
            default_pairs = list(config.TRADING_PAIRS.values())
            
            # Combine with additional pairs
            all_pairs = set(symbols + default_pairs + self.additional_pairs)
            
            # Filter to only include pairs with correct format
            valid_pairs = [pair for pair in all_pairs if '/' in pair]
            
            logger.info(f"Found {len(valid_pairs)} available trading pairs")
            return valid_pairs
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            # Return default pairs as fallback
            return list(config.TRADING_PAIRS.values()) + self.additional_pairs
    
    def find_triangular_paths(self, base_currency: str = 'CAD', max_paths: int = 20) -> List[List[str]]:
        """
        Find all possible triangular arbitrage paths starting and ending with the base currency.
        
        Args:
            base_currency: Base currency to start and end with
            max_paths: Maximum number of paths to return
            
        Returns:
            List of triangular paths (each path is a list of currencies)
        """
        # Get all trading pairs
        symbols = self.get_available_symbols()
        
        # Extract currencies from symbols
        currencies = set()
        for symbol in symbols:
            base, quote = symbol.split('/')
            currencies.add(base)
            currencies.add(quote)
        
        # Create a graph of currency connections
        graph = {}
        for symbol in symbols:
            base, quote = symbol.split('/')
            
            if base not in graph:
                graph[base] = set()
            if quote not in graph:
                graph[quote] = set()
            
            graph[base].add(quote)
            graph[quote].add(base)
        
        # Find all triangular paths
        paths = []
        
        def dfs(current, path, depth):
            if depth == 3 and current == base_currency:
                paths.append(path[:])
                return
            
            if depth >= 3:
                return
            
            if current in graph:
                for next_currency in graph[current]:
                    if next_currency != current and (depth > 0 or next_currency != base_currency):
                        path.append(next_currency)
                        dfs(next_currency, path, depth + 1)
                        path.pop()
        
        dfs(base_currency, [base_currency], 0)
        
        # Sort paths by potential profitability (heuristic: fewer exotic currencies)
        def path_score(path):
            # Prefer paths with major currencies
            major_currencies = {'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'DOT', 'ADA', 'LINK', 'XLM', 'USDT', 'USDC', 'DAI'}
            return sum(1 for currency in path if currency in major_currencies)
        
        paths.sort(key=path_score, reverse=True)
        
        # Return top paths
        return paths[:max_paths]
    
    def calculate_triangle_profitability(self, path: List[str], prices: Dict[str, float]) -> Dict:
        """
        Calculate the profitability of a triangular path.
        
        Args:
            path: List of currencies in the path
            prices: Dictionary of prices for each trading pair
            
        Returns:
            Dictionary with profitability metrics
        """
        # Create the trading pairs from the path
        pairs = []
        for i in range(len(path) - 1):
            # Determine the correct symbol format (base/quote)
            base, quote = path[i], path[i+1]
            
            # Check if the pair exists in both directions
            if f"{base}/{quote}" in prices:
                pairs.append((f"{base}/{quote}", True))  # True means direct
            elif f"{quote}/{base}" in prices:
                pairs.append((f"{quote}/{base}", False))  # False means inverse
            else:
                # If the pair doesn't exist, return zero profit
                return {
                    'path': path,
                    'pairs': [],
                    'start_amount': self.start_amount,
                    'end_amount': self.start_amount,
                    'profit': 0,
                    'profit_pct': 0,
                    'is_profitable': False
                }
        
        # Calculate the conversion
        amount = self.start_amount
        
        for pair, is_direct in pairs:
            price = prices[pair]
            
            if is_direct:
                # Selling base for quote
                amount = amount * price * self.fee_factor
            else:
                # Buying base with quote
                amount = amount / price * self.fee_factor
        
        # Calculate profit
        profit = amount - self.start_amount
        profit_pct = (profit / self.start_amount) * 100
        
        return {
            'path': path,
            'pairs': [pair for pair, _ in pairs],
            'start_amount': self.start_amount,
            'end_amount': amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'is_profitable': profit_pct > self.min_profit_pct
        }
    
    def analyze_historical_data(self, days: int = 90, base_currency: str = 'CAD') -> List[Dict]:
        """
        Analyze historical data to find the most profitable triangular paths.
        
        Args:
            days: Number of days of historical data to analyze
            base_currency: Base currency to start and end with
            
        Returns:
            List of optimal triangular paths with statistics
        """
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"Analyzing historical data from {start_date} to {end_date}")
        
        # Get all available symbols
        all_symbols = self.get_available_symbols()
        logger.info(f"Found {len(all_symbols)} trading pairs on the exchange")
        
        # Find potential triangular paths
        potential_paths = self.find_triangular_paths(base_currency=base_currency)
        logger.info(f"Found {len(potential_paths)} potential triangular paths")
        
        # Extract all unique trading pairs needed for these paths
        needed_pairs = set()
        for path in potential_paths:
            for i in range(len(path) - 1):
                base, quote = path[i], path[i+1]
                if f"{base}/{quote}" in all_symbols:
                    needed_pairs.add(f"{base}/{quote}")
                elif f"{quote}/{base}" in all_symbols:
                    needed_pairs.add(f"{quote}/{base}")
        
        logger.info(f"Need to download data for {len(needed_pairs)} trading pairs")
        
        # Download historical data for each pair
        for symbol in needed_pairs:
            self.download_historical_data(symbol, start_date, end_date)
        
        # Load all historical data
        historical_data = {}
        for symbol in needed_pairs:
            symbol_file = symbol.replace('/', '_')
            file_path = self.data_dir / f"{symbol_file}_{start_date}_{end_date}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    historical_data[symbol] = df
                except Exception as e:
                    logger.error(f"Error loading historical data for {symbol}: {e}")
        
        # Analyze each path
        path_stats = {}
        
        # Get all unique timestamps across all data
        all_timestamps = set()
        for symbol, df in historical_data.items():
            all_timestamps.update(df['timestamp'].dt.floor('D').unique())
        
        all_timestamps = sorted(all_timestamps)
        logger.info(f"Analyzing {len(all_timestamps)} days of historical data")
        
        # For each timestamp, calculate profitability of each path
        for timestamp in all_timestamps:
            # Get prices for this timestamp
            prices = {}
            for symbol, df in historical_data.items():
                # Get the row for this timestamp
                day_data = df[df['timestamp'].dt.floor('D') == timestamp]
                if not day_data.empty:
                    # Use closing price
                    prices[symbol] = day_data['close'].iloc[0]
            
            # Skip if we don't have enough prices
            if len(prices) < len(needed_pairs) * 0.8:  # Allow some missing data
                continue
            
            # Calculate profitability for each path
            for path in potential_paths:
                result = self.calculate_triangle_profitability(path, prices)
                
                # Skip if we couldn't calculate (missing prices)
                if not result['pairs']:
                    continue
                
                # Create a key for this path
                path_key = '→'.join(path)
                
                if path_key not in path_stats:
                    path_stats[path_key] = {
                        'path': path,
                        'pairs': result['pairs'],
                        'profitable_days': 0,
                        'total_days': 0,
                        'max_profit_pct': 0,
                        'avg_profit_pct': 0,
                        'total_profit_pct': 0,
                        'success_rate': 0
                    }
                
                # Update statistics
                path_stats[path_key]['total_days'] += 1
                
                if result['is_profitable']:
                    path_stats[path_key]['profitable_days'] += 1
                
                path_stats[path_key]['total_profit_pct'] += result['profit_pct']
                
                if result['profit_pct'] > path_stats[path_key]['max_profit_pct']:
                    path_stats[path_key]['max_profit_pct'] = result['profit_pct']
        
        # Calculate final statistics
        for path_key, stats in path_stats.items():
            if stats['total_days'] > 0:
                stats['avg_profit_pct'] = stats['total_profit_pct'] / stats['total_days']
                stats['success_rate'] = (stats['profitable_days'] / stats['total_days']) * 100
        
        # Sort by success rate and average profit
        sorted_paths = sorted(
            path_stats.values(),
            key=lambda x: (x['success_rate'], x['avg_profit_pct']),
            reverse=True
        )
        
        # Store results
        self.triangle_stats = path_stats
        self.optimal_triangles = sorted_paths[:10]  # Top 10 triangles
        
        # Save results to file
        results_file = self.data_dir / f"optimal_triangles_{start_date}_{end_date}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'base_currency': base_currency,
                'optimal_triangles': self.optimal_triangles
            }, f, indent=2)
        
        logger.info(f"Saved optimal triangles to {results_file}")
        
        return self.optimal_triangles
    
    def get_optimal_triangles(self, count: int = 5) -> List[Dict]:
        """
        Get the top optimal triangular paths.
        
        Args:
            count: Number of optimal paths to return
            
        Returns:
            List of optimal triangular paths
        """
        if not self.optimal_triangles:
            # Try to load from file
            latest_file = None
            latest_time = 0
            
            for file in self.data_dir.glob("optimal_triangles_*.json"):
                file_time = file.stat().st_mtime
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file
            
            if latest_file:
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                        self.optimal_triangles = data['optimal_triangles']
                        logger.info(f"Loaded optimal triangles from {latest_file}")
                except Exception as e:
                    logger.error(f"Error loading optimal triangles: {e}")
        
        return self.optimal_triangles[:count]
    
    def print_optimal_triangles(self, count: int = 5):
        """
        Print the top optimal triangular paths.
        
        Args:
            count: Number of optimal paths to print
        """
        optimal_triangles = self.get_optimal_triangles(count)
        
        if not optimal_triangles:
            print("No optimal triangles found. Run analyze_historical_data() first.")
            return
        
        print("\n" + "=" * 80)
        print("Optimal Triangular Arbitrage Paths".center(80))
        print("=" * 80)
        
        for i, triangle in enumerate(optimal_triangles, 1):
            print(f"\n{i}. Path: {' → '.join(triangle['path'])}")
            print(f"   Trading Pairs: {', '.join(triangle['pairs'])}")
            print(f"   Success Rate: {triangle['success_rate']:.2f}%")
            print(f"   Avg Profit: {triangle['avg_profit_pct']:.4f}%")
            print(f"   Max Profit: {triangle['max_profit_pct']:.4f}%")
            print(f"   Profitable Days: {triangle['profitable_days']} / {triangle['total_days']}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    optimizer = TriangleOptimizer()
    
    # Analyze 90 days of historical data
    optimal_triangles = optimizer.analyze_historical_data(days=90, base_currency='CAD')
    
    # Print the top 5 optimal triangles
    optimizer.print_optimal_triangles(count=5)