"""
Continuous monitoring of the AI optimization session
Automatically checks progress and alerts when complete
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import psutil

def monitor_session_continuously():
    """Monitor the optimization session and provide updates."""
    print("🔄 Starting Continuous AI Optimization Session Monitor")
    print("=" * 60)
    
    session_name = "optimized_session_2025-08-20_14-33"
    session_dir = Path("data") / session_name
    start_time = datetime.now()
    expected_end_time = start_time + timedelta(minutes=60)
    
    print(f"📊 Monitoring Session: {session_name}")
    print(f"⏰ Started: {start_time.strftime('%H:%M:%S')}")
    print(f"🎯 Expected End: {expected_end_time.strftime('%H:%M:%S')}")
    print(f"⏱️ Duration: 60 minutes")
    print("\n" + "="*60)
    
    last_data_points = 0
    last_trades = 0
    
    try:
        while True:
            current_time = datetime.now()
            elapsed = current_time - start_time
            remaining = expected_end_time - current_time
            
            print(f"\n🕐 {current_time.strftime('%H:%M:%S')} - Progress Check")
            print(f"⏱️ Elapsed: {elapsed.seconds//60}m {elapsed.seconds%60}s")
            
            if remaining.total_seconds() > 0:
                print(f"⏳ Remaining: {int(remaining.total_seconds()//60)}m {int(remaining.total_seconds()%60)}s")
            else:
                print("🎉 Session should be complete!")
            
            # Check session data
            portfolio_file = session_dir / "portfolio_history.json"
            trades_file = session_dir / "trades.json"
            
            data_points = 0
            trades_count = 0
            current_value = 100.0
            total_return = 0.0
            
            if portfolio_file.exists():
                try:
                    with open(portfolio_file, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    data_points = len(portfolio_data)
                    if portfolio_data:
                        latest = portfolio_data[-1]
                        current_value = latest['portfolio_value']
                        total_return = latest['total_return']
                        
                        print(f"📊 Data Points: {data_points} (+{data_points - last_data_points})")
                        print(f"💰 Portfolio Value: ${current_value:.2f}")
                        print(f"📈 Total Return: {total_return:.4f}%")
                        
                        last_data_points = data_points
                    
                except Exception as e:
                    print(f"❌ Error reading portfolio data: {e}")
            else:
                print("⏳ Portfolio data not yet available")
            
            if trades_file.exists():
                try:
                    with open(trades_file, 'r') as f:
                        trades_data = json.load(f)
                    
                    trades_count = len(trades_data)
                    new_trades = trades_count - last_trades
                    
                    print(f"🔄 Trades Executed: {trades_count} (+{new_trades})")
                    
                    if new_trades > 0:
                        print("🎯 NEW TRADES DETECTED!")
                        # Show latest trades
                        for trade in trades_data[-new_trades:]:
                            action = trade.get('action', 'unknown')
                            pair = trade.get('pair', 'unknown')
                            amount = trade.get('amount', 0)
                            price = trade.get('price', 0)
                            profit = trade.get('profit', 0)
                            print(f"   • {action} {pair}: {amount:.4f} @ ${price:.2f} (Profit: {profit:+.4f}%)")
                    
                    last_trades = trades_count
                    
                except Exception as e:
                    print(f"❌ Error reading trades data: {e}")
            else:
                print("⏳ No trades executed yet")
            
            # Check if process is still running
            process_running = False
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('run_simple_pair_monitor.py' in str(cmd) for cmd in cmdline):
                                process_running = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception:
                pass
            
            if process_running:
                print("✅ Session process: RUNNING")
            else:
                print("❌ Session process: NOT FOUND")
                print("🎉 SESSION APPEARS TO BE COMPLETE!")
                break
            
            # Check if we're past expected end time
            if remaining.total_seconds() <= 0:
                print("\n🎉 EXPECTED SESSION COMPLETION TIME REACHED!")
                print("Checking for final results...")
                
                # Wait a bit more for final data to be written
                time.sleep(30)
                
                # Final check
                if portfolio_file.exists():
                    try:
                        with open(portfolio_file, 'r') as f:
                            final_portfolio = json.load(f)
                        
                        if final_portfolio:
                            final_data = final_portfolio[-1]
                            print(f"\n📊 FINAL RESULTS:")
                            print(f"💰 Final Portfolio Value: ${final_data['portfolio_value']:.2f}")
                            print(f"📈 Total Return: {final_data['total_return']:.4f}%")
                            print(f"📊 Total Data Points: {len(final_portfolio)}")
                            
                            if trades_file.exists():
                                with open(trades_file, 'r') as f:
                                    final_trades = json.load(f)
                                print(f"🔄 Total Trades: {len(final_trades)}")
                    
                    except Exception as e:
                        print(f"❌ Error reading final data: {e}")
                
                print(f"\n🎯 NEXT STEPS:")
                print("1. Open dashboard: http://localhost:8503")
                print("2. Go to Strategy Optimization tab")
                print("3. Load the optimized session data")
                print("4. Compare with previous performance")
                print("5. Apply any new AI recommendations")
                
                break
            
            print("-" * 60)
            
            # Wait 5 minutes before next check
            print("⏳ Next check in 5 minutes... (Ctrl+C to stop monitoring)")
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Monitoring stopped by user")
        print(f"🔄 Session is still running in background")
        print(f"💡 Use 'python check_optimization_progress.py' for manual checks")
    
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        print(f"🔄 Session may still be running")

if __name__ == "__main__":
    monitor_session_continuously()