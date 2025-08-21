"""
Simple Windows-Compatible Launcher for Enhanced KrakenBot
No emojis, just functionality
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required AI packages."""
    print("Installing AI dependencies...")
    
    packages = [
        'scikit-learn>=1.0.0',
        'plotly>=5.0.0', 
        'optuna>=3.0.0'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"SUCCESS: {package}")
        except subprocess.CalledProcessError:
            print(f"WARNING: Failed to install {package}")
    
    print("Installation complete!")

def test_system():
    """Test if all components work."""
    print("Testing system components...")
    
    # Test imports
    try:
        import sklearn
        print("SUCCESS: scikit-learn")
    except ImportError:
        print("ERROR: scikit-learn not available")
    
    try:
        import plotly
        print("SUCCESS: plotly")
    except ImportError:
        print("ERROR: plotly not available")
    
    try:
        import pandas
        print("SUCCESS: pandas")
    except ImportError:
        print("ERROR: pandas not available")
    
    try:
        import numpy
        print("SUCCESS: numpy")
    except ImportError:
        print("ERROR: numpy not available")
    
    print("Component test complete!")

def run_first_simulation():
    """Run your first AI simulation."""
    print("Starting your first AI trading simulation...")
    print("Capital: $1000 (simulated)")
    print("Duration: 2 hours")
    print("Pairs: ETH/CAD, BTC/CAD, ADA/CAD, DOT/CAD")
    print("-" * 50)
    
    cmd = [
        sys.executable, 'run_enhanced_simulation.py', 'simulate',
        '--capital', '1000',
        '--duration', '2', 
        '--interval', '60',
        '--session-name', 'first_ai_session'
    ]
    
    try:
        subprocess.run(cmd)
        print("Simulation complete!")
        return True
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False

def launch_dashboard():
    """Launch the dashboard."""
    print("Launching dashboard...")
    try:
        subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run',
            'ai_enhanced_dashboard.py',
            '--server.port', '8506'
        ])
        print("Dashboard launched at: http://localhost:8506")
    except Exception as e:
        print(f"Dashboard launch failed: {e}")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("    KRAKENBOT ENHANCED AI TRADING SYSTEM")
    print("         Simple Launcher")
    print("=" * 60)
    print()
    
    print("What would you like to do?")
    print("1. Install AI dependencies")
    print("2. Test system components")
    print("3. Run first AI simulation (2 hours)")
    print("4. Launch dashboard")
    print("5. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            install_dependencies()
            break
        elif choice == '2':
            test_system()
            break
        elif choice == '3':
            success = run_first_simulation()
            if success:
                launch_dash = input("Launch dashboard to see results? (y/n): ").strip().lower()
                if launch_dash == 'y':
                    launch_dashboard()
            break
        elif choice == '4':
            launch_dashboard()
            break
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, 3, 4, or 5")

if __name__ == "__main__":
    main()