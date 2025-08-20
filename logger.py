"""
Logging module for the KrakenBot triangular arbitrage system.
Handles logging to console, file, and CSV for data analysis.
"""
import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

import config
from arbitrage import ArbitrageOpportunity

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    config.LOG_DIR / "krakenbot.log",
    rotation="1 day",
    retention="30 days",
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(
    lambda msg: print(msg),  # Console output
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {message}"
)


class DataLogger:
    """Handles logging of arbitrage opportunities and trade data."""
    
    def __init__(self):
        """Initialize the data logger."""
        self.opportunities_file = config.DATA_DIR / "opportunities.csv"
        self.trades_file = config.DATA_DIR / "trades.csv"
        self.stats_file = config.DATA_DIR / "stats.json"
        
        self._initialize_files()
        self.stats = self._load_stats()
    
    def _initialize_files(self):
        """Initialize CSV files with headers if they don't exist."""
        # Opportunities CSV
        if not self.opportunities_file.exists():
            with open(self.opportunities_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'start_amount', 'end_amount', 'profit',
                    'profit_percentage', 'path', 'is_profitable',
                    'btc_cad_price', 'eth_cad_price', 'eth_btc_price'
                ])
        
        # Trades CSV
        if not self.trades_file.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'amount', 'price',
                    'cost', 'fee', 'order_id', 'status'
                ])
        
        # Stats JSON
        if not self.stats_file.exists():
            self._save_stats({
                'total_opportunities': 0,
                'profitable_opportunities': 0,
                'total_profit': 0.0,
                'average_profit': 0.0,
                'max_profit': 0.0,
                'last_updated': datetime.now().isoformat(),
                'hourly_stats': {},
                'daily_stats': {}
            })
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics from JSON file."""
        if not self.stats_file.exists():
            return {
                'total_opportunities': 0,
                'profitable_opportunities': 0,
                'total_profit': 0.0,
                'average_profit': 0.0,
                'max_profit': 0.0,
                'last_updated': datetime.now().isoformat(),
                'hourly_stats': {},
                'daily_stats': {}
            }
        
        with open(self.stats_file, 'r') as f:
            return json.load(f)
    
    def _save_stats(self, stats: Dict[str, Any]):
        """Save statistics to JSON file."""
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def log_opportunity(self, opportunity: ArbitrageOpportunity):
        """
        Log an arbitrage opportunity to CSV and update statistics.
        
        Args:
            opportunity: ArbitrageOpportunity object
        """
        # Log to CSV
        with open(self.opportunities_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                opportunity.timestamp.isoformat(),
                opportunity.start_amount,
                opportunity.end_amount,
                opportunity.profit,
                opportunity.profit_percentage,
                '->'.join(opportunity.path),
                opportunity.is_profitable,
                opportunity.prices.get('BTC/CAD', 0),
                opportunity.prices.get('ETH/CAD', 0),
                opportunity.prices.get('ETH/BTC', 0)
            ])
        
        # Update statistics
        self.stats['total_opportunities'] += 1
        if opportunity.is_profitable:
            self.stats['profitable_opportunities'] += 1
            self.stats['total_profit'] += opportunity.profit
            self.stats['average_profit'] = (
                self.stats['total_profit'] / self.stats['profitable_opportunities']
            )
            self.stats['max_profit'] = max(self.stats['max_profit'], opportunity.profit)
        
        # Update time-based stats
        hour_key = opportunity.timestamp.strftime('%Y-%m-%d %H:00')
        day_key = opportunity.timestamp.strftime('%Y-%m-%d')
        
        # Hourly stats
        if hour_key not in self.stats['hourly_stats']:
            self.stats['hourly_stats'][hour_key] = {
                'opportunities': 0,
                'profitable': 0,
                'profit': 0.0
            }
        self.stats['hourly_stats'][hour_key]['opportunities'] += 1
        if opportunity.is_profitable:
            self.stats['hourly_stats'][hour_key]['profitable'] += 1
            self.stats['hourly_stats'][hour_key]['profit'] += opportunity.profit
        
        # Daily stats
        if day_key not in self.stats['daily_stats']:
            self.stats['daily_stats'][day_key] = {
                'opportunities': 0,
                'profitable': 0,
                'profit': 0.0
            }
        self.stats['daily_stats'][day_key]['opportunities'] += 1
        if opportunity.is_profitable:
            self.stats['daily_stats'][day_key]['profitable'] += 1
            self.stats['daily_stats'][day_key]['profit'] += opportunity.profit
        
        # Save updated stats
        self.stats['last_updated'] = datetime.now().isoformat()
        self._save_stats(self.stats)
        
        # Log to console if profitable
        if opportunity.is_profitable:
            logger.success(f"Profitable opportunity found: {opportunity}")
        else:
            logger.debug(f"Opportunity not profitable: {opportunity}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log a trade to CSV.
        
        Args:
            trade_data: Trade data dictionary
        """
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('amount', 0),
                trade_data.get('price', 0),
                trade_data.get('cost', 0),
                trade_data.get('fee', 0),
                trade_data.get('id', ''),
                trade_data.get('status', '')
            ])
        
        logger.info(f"Trade logged: {trade_data.get('side')} {trade_data.get('amount')} "
                   f"{trade_data.get('symbol')} at {trade_data.get('price')}")
    
    def get_summary(self) -> str:
        """
        Get a summary of statistics.
        
        Returns:
            Summary string
        """
        return (
            f"Total opportunities: {self.stats['total_opportunities']}\n"
            f"Profitable opportunities: {self.stats['profitable_opportunities']}\n"
            f"Total profit: ${self.stats['total_profit']:.2f}\n"
            f"Average profit: ${self.stats['average_profit']:.2f}\n"
            f"Max profit: ${self.stats['max_profit']:.2f}\n"
            f"Last updated: {self.stats['last_updated']}"
        )