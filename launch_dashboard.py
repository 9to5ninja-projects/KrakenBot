#!/usr/bin/env python3
"""
Simple Dashboard Launcher for KrakenBot
Launches the AI-enhanced dashboard for monitoring trading sessions
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("    KRAKENBOT AI DASHBOARD LAUNCHER")
    print("=" * 60)
    
    # Check if dashboard file exists
    dashboard_file = Path("ai_enhanced_dashboard.py")
    if not dashboard_file.exists():
        print("ERROR: Dashboard file not found!")
        print("Make sure you're running this from the KrakenBot directory.")
        return
    
    print("Launching KrakenBot AI Dashboard...")
    print("This will open in your web browser at http://localhost:8506")
    print()
    print("Features available:")
    print("- Real-time portfolio monitoring")
    print("- AI trading insights and predictions") 
    print("- Technical analysis charts")
    print("- Performance analytics")
    print("- Trading opportunity tracking")
    print("- NLP-generated insights")
    print()
    
    # Launch dashboard
    try:
        dashboard_cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'ai_enhanced_dashboard.py',
            '--server.port', '8506',
            '--server.headless', 'true'
        ]
        
        print("Starting dashboard server...")
        process = subprocess.Popen(dashboard_cmd)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        print("Opening dashboard in browser...")
        webbrowser.open('http://localhost:8506')
        
        print()
        print("=" * 60)
        print("DASHBOARD RUNNING!")
        print("=" * 60)
        print("URL: http://localhost:8506")
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 60)
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")
            process.terminate()
            
    except Exception as e:
        print(f"ERROR: Failed to launch dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()