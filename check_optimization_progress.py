"""
Quick status check for the running AI optimization session
"""

import os
import json
from pathlib import Path
from datetime import datetime
import psutil

def check_optimization_status():
    """Check the status of the running optimization session."""
    print("🔍 AI Optimization Session Status Check")
    print("=" * 50)
    
    # Check if the optimized session directory exists
    data_dir = Path("data")
    optimized_sessions = list(data_dir.glob("optimized_session_*"))
    
    if optimized_sessions:
        latest_session = max(optimized_sessions, key=lambda x: x.stat().st_mtime)
        print(f"✅ Found optimized session: {latest_session.name}")
        
        # Check for data files
        files_to_check = [
            "portfolio_history.json",
            "trades.json", 
            "price_history.json"
        ]
        
        for file_name in files_to_check:
            file_path = latest_session / file_name
            if file_path.exists():
                # Get file size and modification time
                size = file_path.stat().st_size
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  📄 {file_name}: {size} bytes (updated: {mod_time.strftime('%H:%M:%S')})")
            else:
                print(f"  ❌ {file_name}: Not found")
        
        # Check portfolio history for progress
        portfolio_file = latest_session / "portfolio_history.json"
        if portfolio_file.exists():
            try:
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                if portfolio_data:
                    print(f"\n📊 Session Progress:")
                    print(f"  Data points collected: {len(portfolio_data)}")
                    
                    latest_data = portfolio_data[-1]
                    print(f"  Current portfolio value: ${latest_data['portfolio_value']:.2f}")
                    print(f"  Total return: {latest_data['total_return']:.4f}%")
                    print(f"  Last update: {latest_data['timestamp']}")
                    
                    # Check for trades
                    trades_file = latest_session / "trades.json"
                    if trades_file.exists():
                        with open(trades_file, 'r') as f:
                            trades_data = json.load(f)
                        print(f"  Trades executed: {len(trades_data)}")
                    
            except Exception as e:
                print(f"  ❌ Error reading portfolio data: {e}")
    else:
        print("❌ No optimized sessions found")
    
    # Check process status
    print(f"\n🔄 Process Status:")
    try:
        # Look for python processes running the monitor
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('run_simple_pair_monitor.py' in str(cmd) for cmd in cmdline):
                        python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if python_processes:
            for proc in python_processes:
                print(f"  ✅ Monitor process running (PID: {proc['pid']})")
        else:
            print("  ⚠️ No monitor processes found")
            
    except Exception as e:
        print(f"  ❌ Error checking processes: {e}")
    
    print(f"\n🌐 Dashboard: http://localhost:8503")
    print("💡 Check the Simple Pair Trading tab for real-time updates")

if __name__ == "__main__":
    check_optimization_status()