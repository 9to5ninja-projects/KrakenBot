"""
Main monitoring module for the KrakenBot triangular arbitrage system.
Continuously checks for arbitrage opportunities and logs them.
"""
import time
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger

import config
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator
from logger import DataLogger
from position_manager import get_position_manager

# Import notification manager if available
try:
    from notifications import get_notification_manager
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

class ArbitrageMonitor:
    """Main class for monitoring arbitrage opportunities."""
    
    def __init__(self, use_optimal_triangles: bool = True, use_position_manager: bool = True):
        """
        Initialize the arbitrage monitor.
        
        Args:
            use_optimal_triangles: Whether to use optimal triangular paths
            use_position_manager: Whether to use position manager for capital allocation
        """
        logger.info("Initializing KrakenBot Triangular Arbitrage Monitor")
        
        # Initialize components
        self.exchange = ExchangeManager(use_api_keys=False)  # No API keys needed for monitoring
        self.calculator = ArbitrageCalculator(self.exchange)
        self.data_logger = DataLogger()
        
        # Initialize notification manager if available
        self.notification_manager = None
        if NOTIFICATIONS_AVAILABLE and os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true':
            self.notification_manager = get_notification_manager()
            logger.info(f"Notifications enabled using {self.notification_manager.method}")
        
        # Initialize position manager if enabled
        self.use_position_manager = use_position_manager
        self.position_manager = get_position_manager() if use_position_manager else None
        
        # Runtime variables
        self.running = False
        self.check_interval = config.CHECK_INTERVAL
        self.start_time = None
        self.checks_performed = 0
        self.opportunities_found = 0
        self.profitable_opportunities = 0
        
        # Optimal triangular paths
        self.use_optimal_triangles = use_optimal_triangles
        self.optimal_triangles = []
        
        if self.use_optimal_triangles:
            self._load_optimal_triangles()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def _load_optimal_triangles(self):
        """Load optimal triangular paths."""
        try:
            # Try to import the triangle optimizer
            from triangle_optimizer import TriangleOptimizer
            
            # Load optimal triangles
            optimizer = TriangleOptimizer()
            self.optimal_triangles = optimizer.get_optimal_triangles(count=5)
            
            if not self.optimal_triangles:
                logger.warning("No optimal triangles found. Using default trading pairs.")
            else:
                logger.info(f"Loaded {len(self.optimal_triangles)} optimal triangular paths")
                
                # Extract trading pairs from optimal triangles
                trading_pairs = set()
                for triangle in self.optimal_triangles:
                    for pair in triangle.get('pairs', []):
                        trading_pairs.add(pair)
                
                # Update calculator's trading pairs if we have optimal triangles
                if trading_pairs:
                    self.calculator.trading_pairs = list(trading_pairs)
                    logger.info(f"Using optimal trading pairs: {', '.join(self.calculator.trading_pairs)}")
                
                # Try to load statistical analysis for optimal entry points
                try:
                    from statistical_analyzer import StatisticalAnalyzer
                    analyzer = StatisticalAnalyzer()
                    
                    # Get trading recommendations
                    recommendations = analyzer.generate_trading_recommendations(self.calculator.trading_pairs)
                    
                    if recommendations:
                        # Log optimal trading hours
                        optimal_hours = recommendations.get('optimal_trading_hours', [])
                        if optimal_hours:
                            logger.info(f"Optimal trading hours (UTC): {', '.join([f'{h}:00' for h in optimal_hours])}")
                        
                        # Log pair recommendations
                        pair_recommendations = recommendations.get('pair_recommendations', [])
                        if pair_recommendations:
                            logger.info("Top trading pair recommendations:")
                            for i, rec in enumerate(pair_recommendations[:3], 1):
                                logger.info(f"  {i}. {rec['symbol']} - {rec['recommendation']} (Volatility: {rec['volatility']:.4f})")
                        
                        # Log triangle recommendations
                        triangle_recommendations = recommendations.get('triangle_recommendations', [])
                        if triangle_recommendations:
                            logger.info("Top triangular arbitrage recommendations:")
                            for i, rec in enumerate(triangle_recommendations[:3], 1):
                                logger.info(f"  {i}. {' → '.join(rec['path'])} - {rec['recommendation']} (Success Rate: {rec['success_rate']:.2f}%)")
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not load statistical analysis: {e}")
        
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load optimal triangles: {e}")
            logger.info("Using default trading pairs")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping monitor...")
        self.running = False
    
    def start(self):
        """Start the monitoring loop."""
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting arbitrage monitor in {config.TRADING_MODE} mode")
        logger.info(f"Monitoring pairs: {', '.join(self.calculator.trading_pairs)}")
        logger.info(f"Starting amount: ${config.START_AMOUNT:.2f} CAD")
        logger.info(f"Minimum profit threshold: {config.MIN_PROFIT_PCT:.2f}%")
        logger.info(f"Breakeven threshold: ~{config.BREAKEVEN_PCT:.2f}%")
        logger.info(f"Check interval: {config.CHECK_INTERVAL} seconds")
        logger.info(f"Notifications: {'Enabled' if self.notification_manager else 'Disabled'}")
        
        # Log position manager information if enabled
        if self.use_position_manager and self.position_manager:
            logger.info(f"Position Manager: Enabled")
            logger.info(f"Max capital: ${self.position_manager.max_capital:.2f}")
            logger.info(f"Position size: ${self.position_manager.position_size:.2f} ({self.position_manager.position_size_pct:.1f}%)")
            logger.info(f"Max positions: {self.position_manager.max_positions}")
            logger.info(f"Open positions: {len(self.position_manager.open_positions)}")
            logger.info(f"Available capital: ${self.position_manager.available_capital:.2f}")
        
        try:
            self._run_monitor_loop()
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")
            raise
        finally:
            self._print_summary()
    
    def _run_monitor_loop(self):
        """Run the main monitoring loop."""
        while self.running:
            loop_start = time.time()
            
            try:
                # Find opportunities
                opportunities = self.calculator.find_opportunities()
                
                # Log all opportunities
                for opportunity in opportunities:
                    self.data_logger.log_opportunity(opportunity)
                    self.opportunities_found += 1
                    
                    if opportunity.is_profitable:
                        self.profitable_opportunities += 1
                        
                        # Open position if position manager is enabled and in simulate/live mode
                        if (self.use_position_manager and self.position_manager and 
                            config.TRADING_MODE in ['simulate', 'live']):
                            try:
                                # Check if we can open a position
                                if self.position_manager.can_open_position():
                                    # Open position
                                    position = self.position_manager.open_position(opportunity)
                                    
                                    if position:
                                        logger.info(f"Opened position {position.id}")
                                        logger.info(f"  Path: {' → '.join(opportunity.path)}")
                                        logger.info(f"  Position size: ${position.position_size:.2f}")
                                        logger.info(f"  Expected profit: ${opportunity.profit:.2f} ({opportunity.profit_percentage:.2f}%)")
                                        
                                        # In simulate mode, we can immediately close the position with the expected profit
                                        if config.TRADING_MODE == 'simulate':
                                            # Calculate exit amount (position size + profit)
                                            exit_amount = position.position_size * (1 + opportunity.profit_percentage / 100)
                                            
                                            # Close position
                                            closed_position = self.position_manager.close_position(position.id, exit_amount)
                                            
                                            if closed_position:
                                                logger.info(f"Closed position {closed_position.id}")
                                                logger.info(f"  Profit: ${closed_position.profit:.2f} ({closed_position.profit_pct:.2f}%)")
                                else:
                                    logger.debug("Cannot open position: insufficient capital or position slots")
                            except Exception as e:
                                logger.error(f"Error managing position: {e}")
                        
                        # Send notification if enabled
                        if self.notification_manager:
                            try:
                                self.notification_manager.send_notification(opportunity)
                            except Exception as e:
                                logger.error(f"Error sending notification: {e}")
                
                # Print status update every 10 checks
                self.checks_performed += 1
                if self.checks_performed % 10 == 0:
                    self._print_status_update()
                
            except Exception as e:
                logger.error(f"Error checking for arbitrage: {e}")
                time.sleep(5)  # Wait a bit longer on error
            
            # Calculate sleep time to maintain consistent interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, self.check_interval - elapsed)
            time.sleep(sleep_time)
    
    def _print_status_update(self):
        """Print a status update to the console."""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(runtime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        status_msg = (
            f"Status: Running for {int(hours)}h {int(minutes)}m {int(seconds)}s | "
            f"Checks: {self.checks_performed} | "
            f"Opportunities: {self.opportunities_found} | "
            f"Profitable: {self.profitable_opportunities}"
        )
        
        # Add position information if position manager is enabled
        if self.use_position_manager and self.position_manager:
            position_msg = (
                f" | Positions: {len(self.position_manager.open_positions)}/{self.position_manager.max_positions} | "
                f"Capital: ${self.position_manager.allocated_capital:.2f}/${self.position_manager.max_capital:.2f}"
            )
            status_msg += position_msg
        
        logger.info(status_msg)
    
    def _print_summary(self):
        """Print a summary of the monitoring session."""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(runtime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("=" * 50)
        logger.info("Monitoring session ended")
        logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Checks performed: {self.checks_performed}")
        logger.info(f"Opportunities found: {self.opportunities_found}")
        logger.info(f"Profitable opportunities: {self.profitable_opportunities}")
        
        # Add notification stats if available
        if self.notification_manager:
            notification_stats = self.notification_manager.get_notification_stats()
            logger.info(f"Notifications sent: {notification_stats['count']}")
        
        # Add position stats if available
        if self.use_position_manager and self.position_manager:
            position_summary = self.position_manager.get_position_summary()
            logger.info("Position Manager Summary:")
            logger.info(f"  Open positions: {position_summary['open_positions']}/{position_summary['max_positions']}")
            logger.info(f"  Closed positions: {position_summary['closed_positions']}")
            logger.info(f"  Allocated capital: ${position_summary['allocated_capital']:.2f}")
            logger.info(f"  Available capital: ${position_summary['available_capital']:.2f}")
            logger.info(f"  Total profit: ${position_summary['total_profit']:.2f}")
            logger.info(f"  Average profit: {position_summary['avg_profit_pct']:.2f}%")
        
        logger.info("=" * 50)
        logger.info(self.data_logger.get_summary())
        logger.info("=" * 50)


if __name__ == "__main__":
    # Create data and log directories if they don't exist
    config.DATA_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    
    # Start the monitor
    monitor = ArbitrageMonitor()
    monitor.start()