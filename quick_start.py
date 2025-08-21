"""
Quick Start Script for Enhanced KrakenBot
Makes it easy to run your first AI trading session
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("    KRAKENBOT ENHANCED AI TRADING SYSTEM")
    print("         Quick Start Assistant")
    print("=" * 60)
    print()

def check_setup():
    """Check if system is ready."""
    print("Checking system setup...")
    
    # Check if enhanced files exist
    required_files = [
        'run_enhanced_simulation.py',
        'enhanced_simulation_system.py',
        'ai_enhanced_dashboard.py',
        'advanced_technical_indicators.py',
        'ai_strategy_optimizer.py',
        'multi_coin_analyzer.py',
        'enhanced_nlp_analyzer.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False
    
    print("All enhanced system files found")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling AI dependencies...")
    try:
        result = subprocess.run([
            sys.executable, 'run_enhanced_simulation.py', 'install'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Dependencies installed successfully")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Installation error: {e}")
        return False

def test_components():
    """Test AI components."""
    print("\nTesting AI components...")
    try:
        result = subprocess.run([
            sys.executable, 'run_enhanced_simulation.py', 'test'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("All AI components working")
            return True
        else:
            print(f"Component test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Test error: {e}")
        return False

def get_user_preferences():
    """Get user preferences for first session."""
    print("\nLet's configure your first AI trading session!")
    print()
    
    # Session duration
    print("How long should your first learning session be?")
    print("1. Short (1 hour) - Quick test")
    print("2. Medium (2 hours) - Recommended for first time")
    print("3. Long (4 hours) - More comprehensive learning")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == '1':
            duration = 1
            break
        elif choice == '2':
            duration = 2
            break
        elif choice == '3':
            duration = 4
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Check interval
    print(f"\nSelected: {duration} hour session")
    print("\nHow often should the AI check for trading opportunities?")
    print("1. Conservative (2 minutes) - Fewer trades, more careful")
    print("2. Balanced (1 minute) - Recommended")
    print("3. Active (30 seconds) - More frequent analysis")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == '1':
            interval = 120
            break
        elif choice == '2':
            interval = 60
            break
        elif choice == '3':
            interval = 30
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Dashboard
    print(f"\nSelected: Check every {interval} seconds")
    dashboard = input("\nLaunch AI dashboard automatically? (y/n): ").strip().lower() == 'y'
    
    return duration, interval, dashboard

def run_first_session(duration, interval, dashboard):
    """Run the first AI trading session."""
    session_name = f"first_ai_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"\nStarting your first AI trading session!")
    print(f"Session: {session_name}")
    print(f"Capital: $1,000 (simulated)")
    print(f"Duration: {duration} hours")
    print(f"Check interval: {interval} seconds")
    print(f"Dashboard: {'Yes' if dashboard else 'No'}")
    print()
    
    # Build command
    cmd = [
        sys.executable, 'run_enhanced_simulation.py', 'simulate',
        '--capital', '1000',
        '--duration', str(duration),
        '--interval', str(interval),
        '--session-name', session_name
    ]
    
    # Don't launch dashboard during simulation - launch separately
    
    print("AI is now learning and trading...")
    print("Watch the progress below:")
    print("-" * 60)
    
    try:
        # Run the simulation
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("\nSession completed successfully!")
            
            # Launch dashboard if requested
            if dashboard:
                print("\nLaunching AI Dashboard...")
                dashboard_cmd = [
                    sys.executable, 'run_enhanced_simulation.py', 'dashboard'
                ]
                try:
                    subprocess.Popen(dashboard_cmd)
                    print("Dashboard launched at http://localhost:8506")
                    print("You can view your trading results in the browser!")
                except Exception as e:
                    print(f"Failed to launch dashboard: {e}")
            
            return session_name
        else:
            print(f"\nSession failed with return code: {process.returncode}")
            return None
            
    except KeyboardInterrupt:
        print("\nSession stopped by user")
        process.terminate()
        return session_name
    except Exception as e:
        print(f"\nSession error: {e}")
        return None

def analyze_results(session_name):
    """Analyze session results."""
    if not session_name:
        return
    
    print(f"\nAnalyzing AI performance for session: {session_name}")
    print("-" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, 'run_enhanced_simulation.py', 'analyze', session_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Analysis failed: {result.stderr}")
    except Exception as e:
        print(f"Analysis error: {e}")

def launch_dashboard():
    """Launch the dashboard."""
    print("\nLaunching AI Dashboard...")
    try:
        subprocess.Popen([
            sys.executable, 'run_enhanced_simulation.py', 'dashboard'
        ])
        print("Dashboard launched at http://localhost:8506")
        print("   Open this URL in your browser to see detailed results")
    except Exception as e:
        print(f"Dashboard launch error: {e}")

def show_next_steps():
    """Show recommended next steps."""
    print("\nNEXT STEPS:")
    print("=" * 50)
    print("1. Review the AI analysis above")
    print("2. Check the dashboard for detailed charts")
    print("3. Run more sessions to improve AI learning")
    print("4. Watch performance improve over time")
    print()
    print("RECOMMENDED FOLLOW-UP SESSIONS:")
    print("   python run_enhanced_simulation.py simulate --capital 1000 --duration 3")
    print("   python run_enhanced_simulation.py simulate --capital 1000 --duration 4")
    print()
    print("READ THE GUIDE:")
    print("   Check GETTING_STARTED_GUIDE.md for detailed instructions")
    print()
    print("The AI gets smarter with each session!")

def main():
    """Main quick start function."""
    print_banner()
    
    # Check setup
    if not check_setup():
        print("Setup incomplete. Please ensure all enhanced system files are present.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check your Python environment.")
        return
    
    # Test components
    if not test_components():
        print("Component tests failed. Please check the error messages above.")
        return
    
    # Get user preferences
    duration, interval, dashboard = get_user_preferences()
    
    # Run first session
    session_name = run_first_session(duration, interval, dashboard)
    
    # Analyze results
    analyze_results(session_name)
    
    # Launch dashboard if not already launched
    if not dashboard and session_name:
        launch_dash = input("\nLaunch dashboard to see detailed results? (y/n): ").strip().lower() == 'y'
        if launch_dash:
            launch_dashboard()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()