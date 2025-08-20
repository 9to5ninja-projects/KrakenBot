"""
Backtesting module for KrakenBot.
Allows testing arbitrage strategies on historical data.
"""
import os
import sys
import json
import csv
import time
import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from arbitrage import ArbitrageCalculator, ArbitrageOpportunity
from exchange import ExchangeManager

class HistoricalDataManager:
    """Manages historical price data for backtesting."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the historical data manager."""
        self.data_dir = data_dir or config.DATA_DIR / "historical"
        self.data_dir.mkdir(exist_ok=True)
    
    def download_historical_data(self, symbol: str, start_date: str, end_date: str):
        """
        Download historical data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/CAD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date}")
        
        # Create exchange manager
        exchange = ExchangeManager(use_api_keys=False)
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Prepare file path
        file_path = self.data_dir / f"{symbol.replace('/', '_')}_{start_date}_{end_date}.csv"
        
        try:
            # Fetch OHLCV data
            data = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                logger.info(f"Fetching data for {symbol} at {datetime.datetime.fromtimestamp(current_timestamp/1000)}")
                
                # Fetch data (1 day timeframe)
                ohlcv = exchange.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe='1d',
                    since=current_timestamp,
                    limit=1000  # Maximum limit
                )
                
                if not ohlcv:
                    logger.warning(f"No data returned for {symbol}")
                    break
                
                # Add to data list
                data.extend(ohlcv)
                
                # Update timestamp for next batch
                current_timestamp = ohlcv[-1][0] + 86400000  # Add 1 day in milliseconds
                
                # Avoid rate limits
                time.sleep(1)
            
            # Save to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for row in data:
                    writer.writerow(row)
            
            logger.info(f"Historical data saved to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            return None
    
    def load_historical_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/CAD')
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical data
        """
        # Find matching files
        symbol_prefix = symbol.replace('/', '_')
        matching_files = list(self.data_dir.glob(f"{symbol_prefix}_*.csv"))
        
        if not matching_files:
            logger.error(f"No historical data found for {symbol}")
            return None
        
        # Load all matching files
        dfs = []
        for file_path in matching_files:
            try:
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            logger.error(f"Failed to load any data for {symbol}")
            return None
        
        # Combine all data
        combined_df = pd.concat(dfs).drop_duplicates().sort_values('timestamp')
        
        # Filter by date if specified
        if start_date:
            start_timestamp = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            combined_df = combined_df[combined_df['timestamp'] >= start_timestamp]
        
        if end_date:
            end_timestamp = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            combined_df = combined_df[combined_df['timestamp'] <= end_timestamp]
        
        return combined_df


class BacktestEngine:
    """Engine for backtesting arbitrage strategies."""
    
    def __init__(self):
        """Initialize the backtest engine."""
        self.data_manager = HistoricalDataManager()
        self.results = {
            'opportunities': [],
            'profitable_opportunities': [],
            'total_profit': 0.0,
            'max_profit': 0.0,
            'avg_profit': 0.0,
            'win_rate': 0.0,
            'start_date': None,
            'end_date': None,
            'pairs': []
        }
    
    def prepare_data(self, pairs: List[str], start_date: str, end_date: str, download: bool = False):
        """
        Prepare historical data for backtesting.
        
        Args:
            pairs: List of trading pair symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            download: Whether to download new data
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        data = {}
        
        for symbol in pairs:
            # Download data if requested
            if download:
                self.data_manager.download_historical_data(symbol, start_date, end_date)
            
            # Load data
            df = self.data_manager.load_historical_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                data[symbol] = df
            else:
                logger.error(f"No data available for {symbol}")
                return None
        
        return data
    
    def run_backtest(self, start_date: str, end_date: str, start_amount: float = 10000.0,
                    profit_threshold: float = 5.0, maker_fee: float = 0.0025,
                    slippage_buffer: float = 0.001, download: bool = False):
        """
        Run a backtest for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            start_amount: Starting amount in CAD
            profit_threshold: Minimum profit threshold in CAD
            maker_fee: Maker fee percentage
            slippage_buffer: Slippage buffer percentage
            download: Whether to download new data
            
        Returns:
            Dictionary of backtest results
        """
        # Set up pairs
        pairs = [
            config.TRADING_PAIRS['PAIR_1'],
            config.TRADING_PAIRS['PAIR_2'],
            config.TRADING_PAIRS['PAIR_3']
        ]
        
        # Prepare data
        data = self.prepare_data(pairs, start_date, end_date, download)
        
        if not data:
            logger.error("Failed to prepare data for backtest")
            return None
        
        # Initialize results
        self.results = {
            'opportunities': [],
            'profitable_opportunities': [],
            'total_profit': 0.0,
            'max_profit': 0.0,
            'avg_profit': 0.0,
            'win_rate': 0.0,
            'start_date': start_date,
            'end_date': end_date,
            'pairs': pairs,
            'start_amount': start_amount,
            'profit_threshold': profit_threshold,
            'maker_fee': maker_fee,
            'slippage_buffer': slippage_buffer
        }
        
        # Get common timestamps across all pairs
        timestamps = set(data[pairs[0]]['timestamp'])
        for symbol in pairs[1:]:
            timestamps = timestamps.intersection(set(data[symbol]['timestamp']))
        
        timestamps = sorted(timestamps)
        logger.info(f"Found {len(timestamps)} common timestamps for backtesting")
        
        # Run backtest for each timestamp
        for ts in timestamps:
            # Get prices at this timestamp
            prices = {}
            for symbol in pairs:
                price_row = data[symbol][data[symbol]['timestamp'] == ts].iloc[0]
                prices[symbol] = price_row['close']
            
            # Calculate arbitrage opportunities
            self._calculate_arbitrage_at_timestamp(
                timestamp=ts,
                prices=prices,
                start_amount=start_amount,
                profit_threshold=profit_threshold,
                maker_fee=maker_fee,
                slippage_buffer=slippage_buffer
            )
        
        # Calculate summary statistics
        self._calculate_summary_statistics()
        
        return self.results
    
    def _calculate_arbitrage_at_timestamp(self, timestamp: int, prices: Dict[str, float],
                                         start_amount: float, profit_threshold: float,
                                         maker_fee: float, slippage_buffer: float):
        """
        Calculate arbitrage opportunities at a specific timestamp.
        
        Args:
            timestamp: Timestamp in milliseconds
            prices: Dictionary of prices keyed by symbol
            start_amount: Starting amount in CAD
            profit_threshold: Minimum profit threshold in CAD
            maker_fee: Maker fee percentage
            slippage_buffer: Slippage buffer percentage
        """
        # Extract prices for the triangular arbitrage
        btc_cad_price = prices.get(config.TRADING_PAIRS['PAIR_1'])
        eth_cad_price = prices.get(config.TRADING_PAIRS['PAIR_2'])
        eth_btc_price = prices.get(config.TRADING_PAIRS['PAIR_3'])
        
        if not all([btc_cad_price, eth_cad_price, eth_btc_price]):
            logger.warning(f"Missing prices at timestamp {timestamp}")
            return
        
        # Fee adjustment factor
        fee_factor = 1 - maker_fee - slippage_buffer
        
        # Calculate CAD → BTC → ETH → CAD path
        cad_to_btc = start_amount / btc_cad_price * fee_factor
        btc_to_eth = cad_to_btc * eth_btc_price * fee_factor
        eth_to_cad = btc_to_eth * eth_cad_price * fee_factor
        
        profit1 = eth_to_cad - start_amount
        profit_percentage1 = (profit1 / start_amount) * 100
        
        # Calculate CAD → ETH → BTC → CAD path
        cad_to_eth = start_amount / eth_cad_price * fee_factor
        eth_to_btc = cad_to_eth * (1 / eth_btc_price) * fee_factor
        btc_to_cad = eth_to_btc * btc_cad_price * fee_factor
        
        profit2 = btc_to_cad - start_amount
        profit_percentage2 = (profit2 / start_amount) * 100
        
        # Create opportunity objects
        dt = datetime.datetime.fromtimestamp(timestamp / 1000)
        
        opportunity1 = {
            'timestamp': timestamp,
            'datetime': dt.isoformat(),
            'start_amount': start_amount,
            'end_amount': eth_to_cad,
            'profit': profit1,
            'profit_percentage': profit_percentage1,
            'path': ['CAD', 'BTC', 'ETH', 'CAD'],
            'prices': {
                'BTC/CAD': btc_cad_price,
                'ETH/CAD': eth_cad_price,
                'ETH/BTC': eth_btc_price
            },
            'is_profitable': profit1 > profit_threshold
        }
        
        opportunity2 = {
            'timestamp': timestamp,
            'datetime': dt.isoformat(),
            'start_amount': start_amount,
            'end_amount': btc_to_cad,
            'profit': profit2,
            'profit_percentage': profit_percentage2,
            'path': ['CAD', 'ETH', 'BTC', 'CAD'],
            'prices': {
                'BTC/CAD': btc_cad_price,
                'ETH/CAD': eth_cad_price,
                'ETH/BTC': eth_btc_price
            },
            'is_profitable': profit2 > profit_threshold
        }
        
        # Add opportunities to results
        self.results['opportunities'].append(opportunity1)
        self.results['opportunities'].append(opportunity2)
        
        # Add profitable opportunities
        if opportunity1['is_profitable']:
            self.results['profitable_opportunities'].append(opportunity1)
        
        if opportunity2['is_profitable']:
            self.results['profitable_opportunities'].append(opportunity2)
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics for the backtest results."""
        # Total opportunities
        self.results['total_opportunities'] = len(self.results['opportunities'])
        self.results['total_profitable'] = len(self.results['profitable_opportunities'])
        
        # Win rate
        if self.results['total_opportunities'] > 0:
            self.results['win_rate'] = (self.results['total_profitable'] / self.results['total_opportunities']) * 100
        else:
            self.results['win_rate'] = 0.0
        
        # Profit statistics
        if self.results['profitable_opportunities']:
            profits = [op['profit'] for op in self.results['profitable_opportunities']]
            self.results['total_profit'] = sum(profits)
            self.results['max_profit'] = max(profits)
            self.results['avg_profit'] = sum(profits) / len(profits)
            self.results['min_profit'] = min(profits)
        else:
            self.results['total_profit'] = 0.0
            self.results['max_profit'] = 0.0
            self.results['avg_profit'] = 0.0
            self.results['min_profit'] = 0.0
    
    def save_results(self, file_path: Optional[Path] = None):
        """
        Save backtest results to a file.
        
        Args:
            file_path: Path to save results (default: data_dir/backtest_results_YYYY-MM-DD.json)
        """
        if not file_path:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = config.DATA_DIR / f"backtest_results_{timestamp}.json"
        
        try:
            with open(file_path, 'w') as f:
                # Create a copy of results with datetime objects converted to strings
                results_copy = self.results.copy()
                
                # Save to file
                json.dump(results_copy, f, indent=2)
            
            logger.info(f"Backtest results saved to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return None
    
    def plot_results(self, save_path: Optional[Path] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot (default: data_dir/backtest_plot_YYYY-MM-DD.png)
        """
        if not self.results['opportunities']:
            logger.error("No results to plot")
            return None
        
        try:
            # Create DataFrame from opportunities
            df = pd.DataFrame(self.results['opportunities'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime')
            
            # Group by datetime and path, and get the max profit
            grouped = df.groupby(['datetime', 'path'])['profit'].max().unstack()
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Plot profit for each path
            for path in grouped.columns:
                path_str = '→'.join(path)
                plt.plot(grouped.index, grouped[path], label=path_str)
            
            # Plot profit threshold
            plt.axhline(y=self.results['profit_threshold'], color='r', linestyle='--', 
                       label=f"Profit Threshold (${self.results['profit_threshold']})")
            
            # Plot zero line
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Profit (CAD)')
            plt.title('Triangular Arbitrage Backtest Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Add summary statistics as text
            stats_text = (
                f"Total Opportunities: {self.results['total_opportunities']}\n"
                f"Profitable: {self.results['total_profitable']} ({self.results['win_rate']:.2f}%)\n"
                f"Total Profit: ${self.results['total_profit']:.2f}\n"
                f"Avg Profit: ${self.results['avg_profit']:.2f}\n"
                f"Max Profit: ${self.results['max_profit']:.2f}"
            )
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                # Generate default path
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = config.DATA_DIR / f"backtest_plot_{timestamp}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            # Show plot
            plt.tight_layout()
            plt.show()
            
            return save_path
        
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KrakenBot Backtesting Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
        help="Start date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.datetime.now().strftime('%Y-%m-%d'),
        help="End date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--amount",
        type=float,
        default=config.START_AMOUNT,
        help="Starting amount in CAD"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.PROFIT_THRESHOLD,
        help="Profit threshold in CAD"
    )
    
    parser.add_argument(
        "--fee",
        type=float,
        default=config.MAKER_FEE,
        help="Maker fee percentage"
    )
    
    parser.add_argument(
        "--slippage",
        type=float,
        default=config.SLIPPAGE_BUFFER,
        help="Slippage buffer percentage"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download new historical data"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        config.LOG_DIR / "backtest.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(lambda msg: print(msg), level="INFO")
    
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("KrakenBot Backtesting Tool".center(80))
    print("=" * 80)
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Starting Amount: ${args.amount}")
    print(f"Profit Threshold: ${args.threshold}")
    print(f"Maker Fee: {args.fee * 100}%")
    print(f"Slippage Buffer: {args.slippage * 100}%")
    print(f"Download New Data: {args.download}")
    print("=" * 80)
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        start_amount=args.amount,
        profit_threshold=args.threshold,
        maker_fee=args.fee,
        slippage_buffer=args.slippage,
        download=args.download
    )
    
    if results:
        # Save results
        results_file = engine.save_results()
        
        # Plot results
        plot_file = engine.plot_results()
        
        # Print summary
        print("\nBacktest Summary:")
        print("=" * 80)
        print(f"Total Opportunities: {results['total_opportunities']}")
        print(f"Profitable Opportunities: {results['total_profitable']} ({results['win_rate']:.2f}%)")
        print(f"Total Profit: ${results['total_profit']:.2f}")
        print(f"Average Profit: ${results['avg_profit']:.2f}")
        print(f"Maximum Profit: ${results['max_profit']:.2f}")
        print("=" * 80)
        
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        if plot_file:
            print(f"Plot saved to: {plot_file}")
    else:
        print("\nBacktest failed. Check the logs for details.")