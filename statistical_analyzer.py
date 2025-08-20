"""
Statistical Analyzer module for KrakenBot.
Analyzes historical data to identify optimal entry points and trading patterns.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from scipy import stats
import statsmodels.api as sm

import config
from exchange import ExchangeManager

class StatisticalAnalyzer:
    """
    Analyzes historical data to identify optimal entry points and trading patterns.
    """
    
    def __init__(self, exchange_manager: ExchangeManager = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            exchange_manager: Exchange manager instance (optional)
        """
        self.exchange = exchange_manager or ExchangeManager(use_api_keys=False)
        self.data_dir = config.DATA_DIR / 'statistical'
        self.data_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.min_profit_pct = config.MIN_PROFIT_PCT
        self.maker_fee = config.MAKER_FEE
        self.taker_fee = config.TAKER_FEE
        self.slippage_buffer = config.SLIPPAGE_BUFFER
        
        # Results storage
        self.time_patterns = {}
        self.volatility_patterns = {}
        self.correlation_data = {}
        self.optimal_entry_times = {}
    
    def load_historical_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        Load historical data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/CAD')
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Create filename
        symbol_file = symbol.replace('/', '_')
        filename = f"{symbol_file}_{start_date}_{end_date}.csv"
        file_path = config.DATA_DIR / 'historical' / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Historical data file not found: {file_path}")
            logger.info(f"Downloading historical data for {symbol}")
            
            # Try to import the triangle optimizer to use its download function
            try:
                from triangle_optimizer import TriangleOptimizer
                optimizer = TriangleOptimizer(self.exchange)
                optimizer.download_historical_data(symbol, start_date, end_date)
            except ImportError:
                logger.error("Could not import TriangleOptimizer. Please run the optimizer first.")
                return pd.DataFrame()
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_time_patterns(self, symbols: List[str], days: int = 90) -> Dict:
        """
        Analyze time-based patterns in arbitrage opportunities.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with time-based pattern analysis
        """
        logger.info(f"Analyzing time patterns for {len(symbols)} symbols over {days} days")
        
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            df = self.load_historical_data(symbol, days)
            if not df.empty:
                data[symbol] = df
        
        if not data:
            logger.error("No historical data available for analysis")
            return {}
        
        # Extract time components
        time_patterns = {
            'hourly': {},
            'daily': {},
            'weekly': {}
        }
        
        # Analyze each symbol
        for symbol, df in data.items():
            # Add time components
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day_name()
            df['week'] = df['timestamp'].dt.isocalendar().week
            
            # Calculate hourly volatility (using close price)
            hourly_volatility = df.groupby('hour')['close'].agg(['mean', 'std', 'count'])
            hourly_volatility['volatility'] = hourly_volatility['std'] / hourly_volatility['mean'] * 100
            
            # Calculate daily volatility
            daily_volatility = df.groupby('day')['close'].agg(['mean', 'std', 'count'])
            daily_volatility['volatility'] = daily_volatility['std'] / daily_volatility['mean'] * 100
            
            # Store results
            time_patterns['hourly'][symbol] = hourly_volatility.to_dict()
            time_patterns['daily'][symbol] = daily_volatility.to_dict()
        
        # Find optimal trading hours (highest volatility)
        optimal_hours = {}
        for symbol in symbols:
            if symbol in time_patterns['hourly']:
                volatility_by_hour = {
                    hour: stats['volatility'] 
                    for hour, stats in time_patterns['hourly'][symbol]['volatility'].items()
                }
                # Sort by volatility
                sorted_hours = sorted(volatility_by_hour.items(), key=lambda x: x[1], reverse=True)
                optimal_hours[symbol] = [hour for hour, _ in sorted_hours[:5]]
        
        # Find common optimal hours across symbols
        if optimal_hours:
            all_optimal_hours = []
            for symbol, hours in optimal_hours.items():
                all_optimal_hours.extend(hours)
            
            # Count frequency of each hour
            hour_counts = {}
            for hour in all_optimal_hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            # Sort by frequency
            common_optimal_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            time_patterns['optimal_hours'] = [hour for hour, _ in common_optimal_hours[:5]]
        
        # Store results
        self.time_patterns = time_patterns
        
        # Save results to file
        results_file = self.data_dir / f"time_patterns_{start_date}_{end_date}.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_patterns = {}
            for key, value in time_patterns.items():
                if isinstance(value, dict):
                    serializable_patterns[key] = {
                        k: str(v) if not isinstance(v, (dict, list, int, float, str, bool, type(None))) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_patterns[key] = value
            
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'time_patterns': serializable_patterns
            }, f, indent=2)
        
        logger.info(f"Saved time patterns to {results_file}")
        
        return time_patterns
    
    def analyze_volatility(self, symbols: List[str], days: int = 90) -> Dict:
        """
        Analyze volatility patterns to identify optimal entry points.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with volatility analysis
        """
        logger.info(f"Analyzing volatility for {len(symbols)} symbols over {days} days")
        
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            df = self.load_historical_data(symbol, days)
            if not df.empty:
                data[symbol] = df
        
        if not data:
            logger.error("No historical data available for analysis")
            return {}
        
        # Calculate volatility metrics
        volatility_patterns = {}
        
        for symbol, df in data.items():
            # Calculate daily returns
            df['return'] = df['close'].pct_change() * 100
            
            # Calculate rolling volatility (20-day window)
            df['volatility_20d'] = df['return'].rolling(window=20).std()
            
            # Calculate Bollinger Bands (20-day, 2 standard deviations)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['upper_band'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
            df['lower_band'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)
            
            # Calculate distance from bands
            df['band_distance'] = (df['close'] - df['sma_20']) / (df['upper_band'] - df['sma_20'])
            
            # Identify potential entry points (when price is near lower band)
            df['entry_signal'] = (df['band_distance'] < -0.8) & (df['band_distance'] > -1.2)
            
            # Store results
            volatility_patterns[symbol] = {
                'mean_volatility': df['volatility_20d'].mean(),
                'max_volatility': df['volatility_20d'].max(),
                'min_volatility': df['volatility_20d'].min(),
                'entry_points': df[df['entry_signal']]['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            }
        
        # Store results
        self.volatility_patterns = volatility_patterns
        
        # Save results to file
        results_file = self.data_dir / f"volatility_patterns_{start_date}_{end_date}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'volatility_patterns': volatility_patterns
            }, f, indent=2)
        
        logger.info(f"Saved volatility patterns to {results_file}")
        
        return volatility_patterns
    
    def analyze_correlations(self, symbols: List[str], days: int = 90) -> Dict:
        """
        Analyze correlations between trading pairs.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with correlation analysis
        """
        logger.info(f"Analyzing correlations for {len(symbols)} symbols over {days} days")
        
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            df = self.load_historical_data(symbol, days)
            if not df.empty:
                data[symbol] = df
        
        if not data:
            logger.error("No historical data available for analysis")
            return {}
        
        # Create a combined dataframe with closing prices
        combined_data = pd.DataFrame()
        for symbol, df in data.items():
            combined_data[symbol] = df.set_index('timestamp')['close']
        
        # Calculate correlation matrix
        correlation_matrix = combined_data.corr()
        
        # Find pairs with low correlation (good for diversification)
        low_correlation_pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[symbol1, symbol2]
                    low_correlation_pairs.append((symbol1, symbol2, correlation))
        
        # Sort by correlation (lowest first)
        low_correlation_pairs.sort(key=lambda x: abs(x[2]))
        
        # Store results
        self.correlation_data = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'low_correlation_pairs': [
                {'pair1': pair1, 'pair2': pair2, 'correlation': corr}
                for pair1, pair2, corr in low_correlation_pairs[:10]
            ]
        }
        
        # Save results to file
        results_file = self.data_dir / f"correlation_analysis_{start_date}_{end_date}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'low_correlation_pairs': self.correlation_data['low_correlation_pairs']
            }, f, indent=2)
        
        logger.info(f"Saved correlation analysis to {results_file}")
        
        return self.correlation_data
    
    def identify_optimal_entry_points(self, symbols: List[str], days: int = 90) -> Dict:
        """
        Identify optimal entry points for trading based on multiple factors.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with optimal entry points
        """
        logger.info(f"Identifying optimal entry points for {len(symbols)} symbols")
        
        # Ensure we have all the necessary analysis
        if not self.time_patterns:
            self.analyze_time_patterns(symbols, days)
        
        if not self.volatility_patterns:
            self.analyze_volatility(symbols, days)
        
        # Combine analyses to find optimal entry points
        optimal_entry_points = {}
        
        for symbol in symbols:
            entry_points = []
            
            # Get optimal hours for this symbol
            optimal_hours = self.time_patterns.get('optimal_hours', [])
            
            # Get volatility-based entry points
            volatility_entries = self.volatility_patterns.get(symbol, {}).get('entry_points', [])
            
            # Filter entry points by optimal hours
            for entry in volatility_entries:
                entry_dt = datetime.strptime(entry, '%Y-%m-%d %H:%M:%S')
                if entry_dt.hour in optimal_hours:
                    entry_points.append(entry)
            
            optimal_entry_points[symbol] = entry_points
        
        # Store results
        self.optimal_entry_times = optimal_entry_points
        
        # Save results to file
        results_file = self.data_dir / f"optimal_entry_points_{start_date}_{end_date}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'optimal_entry_points': optimal_entry_points
            }, f, indent=2)
        
        logger.info(f"Saved optimal entry points to {results_file}")
        
        return optimal_entry_points
    
    def generate_trading_recommendations(self, symbols: List[str] = None) -> Dict:
        """
        Generate trading recommendations based on all analyses.
        
        Args:
            symbols: List of trading pair symbols (optional)
            
        Returns:
            Dictionary with trading recommendations
        """
        # If no symbols provided, try to use the ones from previous analyses
        if not symbols:
            symbols = []
            if self.time_patterns and 'hourly' in self.time_patterns:
                symbols.extend(list(self.time_patterns['hourly'].keys()))
            if self.volatility_patterns:
                symbols.extend(list(self.volatility_patterns.keys()))
            symbols = list(set(symbols))  # Remove duplicates
        
        if not symbols:
            logger.error("No symbols provided for generating recommendations")
            return {}
        
        logger.info(f"Generating trading recommendations for {len(symbols)} symbols")
        
        # Ensure we have all necessary analyses
        if not self.optimal_entry_times:
            self.identify_optimal_entry_points(symbols)
        
        if not self.correlation_data:
            self.analyze_correlations(symbols)
        
        # Generate recommendations
        recommendations = {
            'optimal_trading_hours': self.time_patterns.get('optimal_hours', []),
            'pair_recommendations': [],
            'triangle_recommendations': []
        }
        
        # Pair recommendations
        for symbol in symbols:
            volatility = self.volatility_patterns.get(symbol, {}).get('mean_volatility', 0)
            entry_points = self.optimal_entry_times.get(symbol, [])
            
            if volatility and entry_points:
                recommendations['pair_recommendations'].append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'recent_entry_points': entry_points[-5:] if len(entry_points) > 5 else entry_points,
                    'recommendation': 'Strong Buy' if volatility > 2 else 'Buy' if volatility > 1 else 'Hold'
                })
        
        # Sort by volatility (highest first)
        recommendations['pair_recommendations'].sort(key=lambda x: x['volatility'], reverse=True)
        
        # Triangle recommendations
        # Try to import the triangle optimizer to get optimal triangles
        try:
            from triangle_optimizer import TriangleOptimizer
            optimizer = TriangleOptimizer()
            optimal_triangles = optimizer.get_optimal_triangles(count=5)
            
            if optimal_triangles:
                # Add volatility information to triangles
                for triangle in optimal_triangles:
                    triangle_volatility = 0
                    for pair in triangle.get('pairs', []):
                        if pair in self.volatility_patterns:
                            triangle_volatility += self.volatility_patterns[pair].get('mean_volatility', 0)
                    
                    triangle['volatility'] = triangle_volatility / len(triangle.get('pairs', [1])) if triangle.get('pairs') else 0
                    triangle['recommendation'] = 'Strong Buy' if triangle['success_rate'] > 70 else 'Buy' if triangle['success_rate'] > 50 else 'Hold'
                
                recommendations['triangle_recommendations'] = optimal_triangles
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load optimal triangles: {e}")
        
        # Save recommendations to file
        results_file = self.data_dir / f"trading_recommendations_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'recommendations': recommendations
            }, f, indent=2)
        
        logger.info(f"Saved trading recommendations to {results_file}")
        
        return recommendations
    
    def plot_volatility_patterns(self, symbol: str, days: int = 90, save_path: Optional[str] = None):
        """
        Plot volatility patterns for a symbol.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
            save_path: Path to save the plot (optional)
        """
        df = self.load_historical_data(symbol, days)
        
        if df.empty:
            logger.error(f"No historical data available for {symbol}")
            return
        
        # Calculate necessary metrics if not already present
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change() * 100
        
        if 'volatility_20d' not in df.columns:
            df['volatility_20d'] = df['return'].rolling(window=20).std()
        
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['upper_band'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
            df['lower_band'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)
        
        if 'entry_signal' not in df.columns:
            df['band_distance'] = (df['close'] - df['sma_20']) / (df['upper_band'] - df['sma_20'])
            df['entry_signal'] = (df['band_distance'] < -0.8) & (df['band_distance'] > -1.2)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price and Bollinger Bands
        ax1.plot(df['timestamp'], df['close'], label='Close Price')
        ax1.plot(df['timestamp'], df['sma_20'], label='20-day SMA', linestyle='--')
        ax1.plot(df['timestamp'], df['upper_band'], label='Upper Band', linestyle=':')
        ax1.plot(df['timestamp'], df['lower_band'], label='Lower Band', linestyle=':')
        
        # Plot entry points
        entry_points = df[df['entry_signal']]
        ax1.scatter(entry_points['timestamp'], entry_points['close'], color='green', marker='^', s=100, label='Entry Points')
        
        ax1.set_title(f'Price and Bollinger Bands for {symbol}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot volatility
        ax2.plot(df['timestamp'], df['volatility_20d'], label='20-day Volatility', color='orange')
        ax2.set_title(f'Volatility for {symbol}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved volatility plot to {save_path}")
        else:
            plt.show()
    
    def print_trading_recommendations(self):
        """Print trading recommendations."""
        recommendations = self.generate_trading_recommendations()
        
        if not recommendations:
            print("No trading recommendations available. Run analyses first.")
            return
        
        print("\n" + "=" * 80)
        print("KrakenBot Trading Recommendations".center(80))
        print("=" * 80)
        
        # Print optimal trading hours
        optimal_hours = recommendations.get('optimal_trading_hours', [])
        if optimal_hours:
            print("\nOptimal Trading Hours (UTC):")
            for hour in optimal_hours:
                print(f"  - {hour}:00 - {hour+1}:00")
        
        # Print pair recommendations
        pair_recommendations = recommendations.get('pair_recommendations', [])
        if pair_recommendations:
            print("\nTop Trading Pair Recommendations:")
            for i, rec in enumerate(pair_recommendations[:5], 1):
                print(f"\n{i}. {rec['symbol']} - {rec['recommendation']}")
                print(f"   Volatility: {rec['volatility']:.4f}")
                if rec['recent_entry_points']:
                    print(f"   Recent Entry Points:")
                    for point in rec['recent_entry_points'][-3:]:
                        print(f"     - {point}")
        
        # Print triangle recommendations
        triangle_recommendations = recommendations.get('triangle_recommendations', [])
        if triangle_recommendations:
            print("\nTop Triangular Arbitrage Recommendations:")
            for i, rec in enumerate(triangle_recommendations[:5], 1):
                print(f"\n{i}. Path: {' → '.join(rec['path'])}")
                print(f"   Trading Pairs: {', '.join(rec['pairs'])}")
                print(f"   Success Rate: {rec['success_rate']:.2f}%")
                print(f"   Avg Profit: {rec['avg_profit_pct']:.4f}%")
                print(f"   Recommendation: {rec['recommendation']}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer()
    
    # Define symbols to analyze
    symbols = ['BTC/CAD', 'ETH/CAD', 'ETH/BTC']
    
    # Run analyses
    analyzer.analyze_time_patterns(symbols)
    analyzer.analyze_volatility(symbols)
    analyzer.analyze_correlations(symbols)
    analyzer.identify_optimal_entry_points(symbols)
    
    # Generate and print recommendations
    analyzer.print_trading_recommendations()
    
    # Plot volatility patterns for BTC/CAD
    analyzer.plot_volatility_patterns('BTC/CAD', save_path=str(analyzer.data_dir / 'BTC_CAD_volatility.png'))