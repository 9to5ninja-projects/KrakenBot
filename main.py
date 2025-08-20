"""
Main entry point for the KrakenBot triangular arbitrage system.
Provides command-line interface to start the monitor, dashboard, or other tools.
"""
import argparse
import sys
import os
import subprocess
import datetime
import threading
from pathlib import Path
from loguru import logger

import config
from monitor import ArbitrageMonitor
from health_monitor import get_monitor as get_health_monitor
from notifications import get_notification_manager
from reports import ReportGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KrakenBot Triangular Arbitrage System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the KrakenBot system")
    run_parser.add_argument(
        "--mode",
        choices=["monitor", "simulate", "live", "dashboard"],
        default="monitor",
        help="Operating mode: monitor (default), simulate, live, or dashboard"
    )
    
    run_parser.add_argument(
        "--amount",
        type=float,
        default=config.START_AMOUNT,
        help=f"Starting amount in CAD (default: {config.START_AMOUNT})"
    )
    
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=config.PROFIT_THRESHOLD,
        help=f"Profit threshold in CAD (legacy, default: {config.PROFIT_THRESHOLD})"
    )
    
    run_parser.add_argument(
        "--min-profit-pct",
        type=float,
        default=config.MIN_PROFIT_PCT,
        help=f"Minimum profit percentage (default: {config.MIN_PROFIT_PCT}%)"
    )
    
    run_parser.add_argument(
        "--max-capital",
        type=float,
        default=config.MAX_CAPITAL,
        help=f"Maximum capital to use for trading (default: ${config.MAX_CAPITAL})"
    )
    
    run_parser.add_argument(
        "--position-size-pct",
        type=float,
        default=config.POSITION_SIZE_PCT,
        help=f"Position size as percentage of max capital (default: {config.POSITION_SIZE_PCT}%)"
    )
    
    run_parser.add_argument(
        "--max-positions",
        type=int,
        default=config.MAX_POSITIONS,
        help=f"Maximum number of concurrent positions (default: {config.MAX_POSITIONS})"
    )
    
    run_parser.add_argument(
        "--interval",
        type=int,
        default=config.CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {config.CHECK_INTERVAL})"
    )
    
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=config.LOG_LEVEL,
        help=f"Logging level (default: {config.LOG_LEVEL})"
    )
    
    run_parser.add_argument(
        "--health-monitor",
        action="store_true",
        help="Enable health monitoring"
    )
    
    run_parser.add_argument(
        "--notifications",
        action="store_true",
        help="Enable notifications for profitable opportunities"
    )
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run the setup wizard")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
        help="Start date in YYYY-MM-DD format"
    )
    
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.datetime.now().strftime('%Y-%m-%d'),
        help="End date in YYYY-MM-DD format"
    )
    
    backtest_parser.add_argument(
        "--amount",
        type=float,
        default=config.START_AMOUNT,
        help="Starting amount in CAD"
    )
    
    backtest_parser.add_argument(
        "--threshold",
        type=float,
        default=config.PROFIT_THRESHOLD,
        help="Profit threshold in CAD (legacy)"
    )
    
    backtest_parser.add_argument(
        "--min-profit-pct",
        type=float,
        default=config.MIN_PROFIT_PCT,
        help=f"Minimum profit percentage (default: {config.MIN_PROFIT_PCT}%)"
    )
    
    backtest_parser.add_argument(
        "--download",
        action="store_true",
        help="Download new historical data"
    )
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a report")
    report_parser.add_argument(
        "--type",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Type of report to generate"
    )
    
    report_parser.add_argument(
        "--date",
        type=str,
        help="Date for daily report (YYYY-MM-DD), end date for weekly report, or ignored for monthly report"
    )
    
    report_parser.add_argument(
        "--month",
        type=int,
        help="Month for monthly report (1-12)"
    )
    
    report_parser.add_argument(
        "--year",
        type=int,
        help="Year for monthly report"
    )
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Find optimal triangular arbitrage paths")
    optimize_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to analyze"
    )
    optimize_parser.add_argument(
        "--base",
        type=str,
        default="CAD",
        help="Base currency to start and end with"
    )
    optimize_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of optimal paths to return"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Perform statistical analysis on trading pairs")
    analyze_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC/CAD", "ETH/CAD", "ETH/BTC"],
        help="Trading pairs to analyze"
    )
    analyze_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to analyze"
    )
    analyze_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate volatility plots for each symbol"
    )
    
    # If no arguments are provided, default to run in monitor mode
    if len(sys.argv) == 1:
        return parser.parse_args(["run"])
    
    return parser.parse_args()

def start_dashboard():
    """Start the Streamlit dashboard."""
    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    
    try:
        # Check if Streamlit is installed
        subprocess.run(
            ["streamlit", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Start the dashboard
        logger.info("Starting Streamlit dashboard...")
        subprocess.run(
            ["streamlit", "run", str(dashboard_path)],
            check=True
        )
    except FileNotFoundError:
        logger.error("Streamlit is not installed. Please install it with 'pip install streamlit'")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)

def run_setup_wizard():
    """Run the setup wizard."""
    from setup_wizard import run_wizard
    run_wizard()

def run_backtest(args):
    """Run a backtest."""
    from backtest import BacktestEngine
    
    # Print banner
    print("=" * 80)
    print("KrakenBot Backtesting Tool".center(80))
    print("=" * 80)
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Starting Amount: ${args.amount}")
    print(f"Minimum Profit Threshold: {args.min_profit_pct}%")
    print(f"Breakeven Threshold: ~{config.BREAKEVEN_PCT:.2f}%")
    print(f"Download New Data: {args.download}")
    print("=" * 80)
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        start_amount=args.amount,
        profit_threshold=args.threshold,  # Legacy parameter
        min_profit_pct=args.min_profit_pct,  # New percentage-based parameter
        maker_fee=config.MAKER_FEE,
        taker_fee=config.TAKER_FEE,
        slippage_buffer=config.SLIPPAGE_BUFFER,
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

def generate_report(args):
    """Generate a report."""
    report_generator = ReportGenerator()
    
    if args.type == "daily":
        report_file = report_generator.generate_daily_report(args.date)
        print(f"Daily report generated: {report_file}")
    
    elif args.type == "weekly":
        report_file = report_generator.generate_weekly_report(args.date)
        print(f"Weekly report generated: {report_file}")
    
    elif args.type == "monthly":
        report_file = report_generator.generate_monthly_report(args.month, args.year)
        print(f"Monthly report generated: {report_file}")

def check_health():
    """Check system health."""
    health_monitor = get_health_monitor()
    health_monitor.check_health()
    health_monitor.save_health_data()
    
    print("\nHealth Summary:")
    print("=" * 80)
    print(health_monitor.get_health_summary())
    print("=" * 80)
    print(f"\nDetailed health data saved to {health_monitor.health_file}")

def optimize_triangles(args):
    """Find optimal triangular arbitrage paths."""
    from triangle_optimizer import TriangleOptimizer
    
    print("=" * 80)
    print("KrakenBot Triangle Optimizer".center(80))
    print("=" * 80)
    print(f"Analyzing {args.days} days of historical data")
    print(f"Base currency: {args.base}")
    print(f"Finding top {args.count} optimal paths")
    print("=" * 80)
    print("\nThis may take some time depending on the number of trading pairs...")
    
    optimizer = TriangleOptimizer()
    
    # Check if we already have optimal triangles
    existing_triangles = optimizer.get_optimal_triangles(args.count)
    
    if not existing_triangles:
        # Analyze historical data
        optimizer.analyze_historical_data(days=args.days, base_currency=args.base)
    
    # Print the optimal triangles
    optimizer.print_optimal_triangles(count=args.count)
    
    # Return the optimal triangles for use in monitor
    return optimizer.get_optimal_triangles(args.count)

def analyze_trading_pairs(args):
    """Perform statistical analysis on trading pairs."""
    from statistical_analyzer import StatisticalAnalyzer
    
    print("=" * 80)
    print("KrakenBot Statistical Analyzer".center(80))
    print("=" * 80)
    print(f"Analyzing {len(args.symbols)} trading pairs over {args.days} days")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Generate plots: {args.plot}")
    print("=" * 80)
    print("\nThis may take some time depending on the number of trading pairs...")
    
    analyzer = StatisticalAnalyzer()
    
    # Run analyses
    analyzer.analyze_time_patterns(args.symbols, args.days)
    analyzer.analyze_volatility(args.symbols, args.days)
    analyzer.analyze_correlations(args.symbols, args.days)
    analyzer.identify_optimal_entry_points(args.symbols, args.days)
    
    # Generate and print recommendations
    analyzer.print_trading_recommendations()
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating volatility plots...")
        for symbol in args.symbols:
            plot_path = str(config.DATA_DIR / 'statistical' / f"{symbol.replace('/', '_')}_volatility.png")
            analyzer.plot_volatility_patterns(symbol, args.days, save_path=plot_path)
            print(f"  - Plot saved for {symbol}")
    
    return analyzer.generate_trading_recommendations()

def run_monitor(args):
    """Run the monitor in the specified mode."""
    # Update config based on command line arguments
    config.START_AMOUNT = args.amount
    config.PROFIT_THRESHOLD = args.threshold
    config.MIN_PROFIT_PCT = args.min_profit_pct
    config.CHECK_INTERVAL = args.interval
    config.LOG_LEVEL = args.log_level
    config.TRADING_MODE = args.mode if args.mode != "dashboard" else "monitor"
    
    # Update position sizing parameters
    config.MAX_CAPITAL = args.max_capital
    config.POSITION_SIZE_PCT = args.position_size_pct
    config.MAX_POSITIONS = args.max_positions
    
    # Recalculate breakeven threshold
    config.BREAKEVEN_PCT = (config.TAKER_FEE + config.SLIPPAGE_BUFFER) * 3 * 100
    
    # Calculate position size
    position_size = config.MAX_CAPITAL * (config.POSITION_SIZE_PCT / 100)
    
    # Print banner
    print("=" * 80)
    print("KrakenBot Triangular Arbitrage System")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Starting amount: ${args.amount:.2f} CAD")
    print(f"Minimum profit threshold: {args.min_profit_pct:.2f}%")
    print(f"Breakeven threshold: ~{config.BREAKEVEN_PCT:.2f}%")
    print(f"Max capital: ${args.max_capital:.2f}")
    print(f"Position size: ${position_size:.2f} ({args.position_size_pct:.1f}%)")
    print(f"Max positions: {args.max_positions}")
    print(f"Check interval: {args.interval} seconds")
    print(f"Log level: {args.log_level}")
    print(f"Health monitoring: {'Enabled' if args.health_monitor else 'Disabled'}")
    print(f"Notifications: {'Enabled' if args.notifications else 'Disabled'}")
    print("=" * 80)
    
    # Start health monitor if enabled
    if args.health_monitor:
        health_monitor = get_health_monitor()
        health_monitor.start()
        logger.info("Health monitor started")
    
    # Initialize notification manager if enabled
    if args.notifications:
        notification_manager = get_notification_manager()
        logger.info(f"Notifications enabled using {notification_manager.method}")
    
    # Start the appropriate mode
    if args.mode == "dashboard":
        start_dashboard()
    else:
        # Start the monitor
        monitor = ArbitrageMonitor()
        
        # Start daily report generation thread if notifications are enabled
        if args.notifications:
            def generate_daily_reports():
                """Generate daily reports at midnight."""
                while True:
                    # Calculate time until next midnight
                    now = datetime.datetime.now()
                    tomorrow = now + datetime.timedelta(days=1)
                    midnight = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
                    seconds_until_midnight = (midnight - now).total_seconds()
                    
                    # Sleep until midnight
                    logger.info(f"Next daily report scheduled in {seconds_until_midnight:.0f} seconds")
                    threading.Event().wait(seconds_until_midnight)
                    
                    # Generate report for yesterday
                    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    try:
                        logger.info(f"Generating daily report for {yesterday}")
                        report_generator = ReportGenerator()
                        report_file = report_generator.generate_daily_report(yesterday)
                        logger.info(f"Daily report generated: {report_file}")
                    except Exception as e:
                        logger.error(f"Error generating daily report: {e}")
            
            # Start report generation thread
            report_thread = threading.Thread(target=generate_daily_reports, daemon=True)
            report_thread.start()
            logger.info("Daily report generation scheduled")
        
        # Start the monitor
        monitor.start()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create necessary directories
    config.DATA_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    
    # Configure logger
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
    
    # Execute the appropriate command
    if args.command == "run":
        run_monitor(args)
    elif args.command == "setup":
        run_setup_wizard()
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "report":
        generate_report(args)
    elif args.command == "health":
        check_health()
    elif args.command == "optimize":
        optimize_triangles(args)
    elif args.command == "analyze":
        analyze_trading_pairs(args)
    else:
        # Default to run in monitor mode
        run_monitor(args)

if __name__ == "__main__":
    main()