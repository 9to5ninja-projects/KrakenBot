"""
Launch script for Enhanced KrakenBot Simulation
Runs the complete AI-enhanced trading system with $1000 capital
"""

import asyncio
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def install_requirements():
    """Install additional requirements for enhanced features."""
    additional_packages = [
        'scikit-learn>=1.0.0',
        'plotly>=5.0.0',
        'optuna>=3.0.0'
    ]
    
    print("Installing additional packages for AI features...")
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package} - continuing anyway")

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'sklearn',
        'plotly',
        'pandas',
        'numpy',
        'streamlit'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Installing missing dependencies...")
        install_requirements()
    else:
        print("All dependencies available")

async def run_enhanced_simulation(args):
    """Run the enhanced simulation system."""
    from enhanced_simulation_system import EnhancedSimulationSystem
    
    print("Starting Enhanced KrakenBot Simulation")
    print("=" * 60)
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Duration: {args.duration} hours")
    print(f"Interval: {args.interval} seconds")
    print(f"Pairs: ETH/CAD, BTC/CAD, SOL/CAD, XRP/CAD")
    print("=" * 60)
    
    # Create simulation system
    simulation = EnhancedSimulationSystem(
        initial_capital=args.capital,
        session_name=args.session_name
    )
    
    # Run simulation
    await simulation.run_simulation(
        duration_hours=args.duration,
        check_interval=args.interval
    )

def launch_dashboard(session_name=None):
    """Launch the AI-enhanced dashboard."""
    print("Launching AI-Enhanced Dashboard...")
    
    dashboard_cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'ai_enhanced_dashboard.py',
        '--server.port', '8506',
        '--server.headless', 'true'
    ]
    
    try:
        subprocess.Popen(dashboard_cmd)
        print("Dashboard launched at http://localhost:8506")
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")

def run_nlp_analysis(session_name):
    """Run NLP analysis on completed session."""
    from enhanced_nlp_analyzer import EnhancedNLPAnalyzer
    
    print(f"Running NLP Analysis on {session_name}...")
    
    analyzer = EnhancedNLPAnalyzer()
    session_path = Path("data") / session_name
    
    if session_path.exists():
        analysis = analyzer.analyze_trading_session(str(session_path))
        
        if 'error' not in analysis:
            print("NLP Analysis Complete")
            print("\nSession Summary:")
            print(analysis['session_summary'])
            
            print("\nKey Insights:")
            for insight in analysis['key_insights'][:3]:
                print(f"- {insight}")
            
            print("\nRecommendations:")
            for rec in analysis['strategic_recommendations'][:3]:
                print(f"- {rec}")
        else:
            print(f"NLP Analysis failed: {analysis['error']}")
    else:
        print(f"Session not found: {session_name}")

def test_ai_components():
    """Test all AI components."""
    print("Testing AI Components...")
    
    # Test technical indicators
    try:
        from advanced_technical_indicators import AdvancedTechnicalAnalyzer, create_sample_price_data
        analyzer = AdvancedTechnicalAnalyzer()
        sample_data = create_sample_price_data("ETH/CAD")
        analysis = analyzer.analyze_pair(sample_data, "ETH/CAD")
        print("Technical Analysis: Working")
    except Exception as e:
        print(f"Technical Analysis: {e}")
    
    # Test AI optimizer
    try:
        from ai_strategy_optimizer import AIStrategyOptimizer
        optimizer = AIStrategyOptimizer()
        print("AI Strategy Optimizer: Working")
    except Exception as e:
        print(f"AI Strategy Optimizer: {e}")
    
    # Test multi-coin analyzer
    try:
        from multi_coin_analyzer import MultiCoinAnalyzer
        multi_analyzer = MultiCoinAnalyzer()
        print("Multi-Coin Analyzer: Working")
    except Exception as e:
        print(f"Multi-Coin Analyzer: {e}")
    
    # Test NLP analyzer
    try:
        from enhanced_nlp_analyzer import EnhancedNLPAnalyzer
        nlp_analyzer = EnhancedNLPAnalyzer()
        print("NLP Analyzer: Working")
    except Exception as e:
        print(f"NLP Analyzer: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Enhanced KrakenBot Simulation System')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run enhanced simulation')
    sim_parser.add_argument('--capital', type=float, default=1000.0, 
                           help='Initial capital (default: 1000)')
    sim_parser.add_argument('--duration', type=float, default=4.0, 
                           help='Duration in hours (default: 4)')
    sim_parser.add_argument('--interval', type=int, default=60, 
                           help='Check interval in seconds (default: 60)')
    sim_parser.add_argument('--session-name', type=str, 
                           help='Custom session name')
    sim_parser.add_argument('--launch-dashboard', action='store_true',
                           help='Launch dashboard after simulation')
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch AI dashboard')
    dash_parser.add_argument('--session', type=str, help='Specific session to display')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Run NLP analysis')
    analysis_parser.add_argument('session_name', help='Session name to analyze')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test AI components')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install dependencies')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check dependencies first
    if args.command != 'install':
        check_dependencies()
    
    if args.command == 'simulate':
        # Run simulation
        asyncio.run(run_enhanced_simulation(args))
        
        # Launch dashboard if requested
        if args.launch_dashboard:
            launch_dashboard(args.session_name)
    
    elif args.command == 'dashboard':
        launch_dashboard(args.session)
    
    elif args.command == 'analyze':
        run_nlp_analysis(args.session_name)
    
    elif args.command == 'test':
        test_ai_components()
    
    elif args.command == 'install':
        install_requirements()
        print("Installation complete!")

if __name__ == "__main__":
    print("KrakenBot Enhanced Simulation System")
    print("AI-Powered Trading with Advanced Analytics")
    print("=" * 50)
    
    main()