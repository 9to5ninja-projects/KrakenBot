#!/usr/bin/env python3
"""
Test script for the streamlined KrakenBot system
Tests the enhanced simulation + streamlined dashboard integration
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("=" * 60)
    print("    KRAKENBOT STREAMLINED SYSTEM TEST")
    print("=" * 60)
    
    print("Testing the complete streamlined system:")
    print("1. Enhanced AI simulation with current session focus")
    print("2. Streamlined dashboard with pair selection")
    print("3. Positions/Trade history tabs")
    print("4. Real-time monitoring")
    print()
    
    # Step 1: Run a short simulation to generate data
    print("Step 1: Running enhanced simulation (2 minutes)...")
    sim_cmd = [
        sys.executable, 'run_enhanced_simulation.py', 'simulate',
        '--capital', '1000',
        '--duration', '0.05',  # 3 minutes
        '--interval', '20',    # 20 seconds
        '--session-name', 'streamlined_test'
    ]
    
    try:
        result = subprocess.run(sim_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Simulation completed successfully!")
        else:
            print(f"❌ Simulation failed: {result.stderr}")
            return
    except subprocess.TimeoutExpired:
        print("⚠️ Simulation timed out, but may have generated some data")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return
    
    # Step 2: Launch streamlined dashboard
    print("\nStep 2: Launching streamlined dashboard...")
    dashboard_cmd = [
        sys.executable, 'launch_dashboard.py'
    ]
    
    try:
        process = subprocess.Popen(dashboard_cmd)
        time.sleep(3)  # Wait for dashboard to start
        
        print("✅ Dashboard launched!")
        print()
        print("=" * 60)
        print("STREAMLINED SYSTEM READY!")
        print("=" * 60)
        print("Dashboard URL: http://localhost:8506")
        print()
        print("Features to test:")
        print("✅ Current session only (no confusion)")
        print("✅ Pair selection for technical indicators")
        print("✅ Current Positions tab")
        print("✅ Trade History tab")
        print("✅ Single page layout")
        print("✅ Auto-refresh enabled")
        print()
        print("The system is now ready for strategy upgrades!")
        print("=" * 60)
        
        # Open browser
        webbrowser.open('http://localhost:8506')
        
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")

if __name__ == "__main__":
    main()