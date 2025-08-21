"""
Real-time monitoring of the aggressive test session
Tracks trade execution and performance in real-time
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def monitor_aggressive_session():
    """Monitor the aggressive test session in real-time."""
    print("⚡ Aggressive Test Session Monitor")
    print("=" * 50)
    
    session_name = "aggressive_test_2025-08-20_15-40"
    session_dir = Path("data") / session_name
    start_time = datetime.now()
    expected_end_time = start_time + timedelta(minutes=30)
    
    print(f"📊 Session: {session_name}")
    print(f"⏰ Started: {start_time.strftime('%H:%M:%S')}")
    print(f"🎯 Expected End: {expected_end_time.strftime('%H:%M:%S')}")
    print(f"⏱️ Duration: 30 minutes")
    print(f"🎯 Goal: Execute actual trades with aggressive thresholds!")
    print("\n" + "="*50)
    
    last_data_points = 0
    last_trades = 0
    trades_detected = []
    
    try:
        while True:
            current_time = datetime.now()
            elapsed = current_time - start_time
            remaining = expected_end_time - current_time
            
            print(f"\n🕐 {current_time.strftime('%H:%M:%S')} - Aggressive Test Check")
            print(f"⏱️ Elapsed: {elapsed.seconds//60}m {elapsed.seconds%60}s")
            
            if remaining.total_seconds() > 0:
                print(f"⏳ Remaining: {int(remaining.total_seconds()//60)}m {int(remaining.total_seconds()%60)}s")
            else:
                print("🎉 Test session should be complete!")
            
            # Check session data
            portfolio_file = session_dir / "portfolio_history.json"
            trades_file = session_dir / "trades.json"
            
            if portfolio_file.exists():
                try:
                    with open(portfolio_file, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    if portfolio_data:
                        data_points = len(portfolio_data)
                        latest = portfolio_data[-1]
                        current_value = latest['portfolio_value']
                        total_return = latest['total_return']
                        
                        print(f"📊 Data Points: {data_points} (+{data_points - last_data_points})")
                        print(f"💰 Portfolio Value: ${current_value:.2f}")
                        print(f"📈 Total Return: {total_return:.4f}%")
                        
                        # Check for significant price movements
                        if data_points >= 2:
                            prev_value = portfolio_data[-2]['portfolio_value']
                            value_change = current_value - prev_value
                            if abs(value_change) > 0.01:  # More than 1 cent change
                                change_pct = (value_change / prev_value) * 100
                                print(f"📊 Value Change: ${value_change:+.2f} ({change_pct:+.4f}%)")
                        
                        last_data_points = data_points
                    
                except Exception as e:
                    print(f"❌ Error reading portfolio data: {e}")
            else:
                print("⏳ Portfolio data not yet available")
            
            # Check for trades
            if trades_file.exists():
                try:
                    with open(trades_file, 'r') as f:
                        trades_data = json.load(f)
                    
                    trades_count = len(trades_data)
                    new_trades = trades_count - last_trades
                    
                    if new_trades > 0:
                        print(f"🎉 NEW TRADES DETECTED! Total: {trades_count} (+{new_trades})")
                        
                        # Show new trades
                        for trade in trades_data[-new_trades:]:
                            action = trade.get('action', 'unknown')
                            pair = trade.get('pair', 'unknown')
                            price = trade.get('price', 0)
                            amount = trade.get('amount', 0)
                            value = trade.get('value', 0)
                            
                            if action == 'buy':
                                print(f"   🟢 BUY {pair}: {amount:.6f} @ ${price:.2f} (${value:.2f})")
                                stop_loss = trade.get('stop_loss', 0)
                                take_profit = trade.get('take_profit', 0)
                                print(f"      Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")
                            elif action == 'sell':
                                profit = trade.get('profit', 0)
                                profit_pct = trade.get('profit_pct', 0)
                                hold_time = trade.get('hold_time', 0)
                                emoji = "📈" if profit > 0 else "📉"
                                print(f"   🔴 SELL {pair}: {amount:.6f} @ ${price:.2f}")
                                print(f"      {emoji} Profit: ${profit:.2f} ({profit_pct:+.3f}%) | Hold: {hold_time:.0f}s")
                            
                            trades_detected.append(trade)
                    else:
                        print(f"🔄 Trades: {trades_count} (no new trades)")
                    
                    last_trades = trades_count
                    
                except Exception as e:
                    print(f"❌ Error reading trades data: {e}")
            else:
                print("⏳ No trades file yet - waiting for first trade...")
            
            # Show current thresholds
            print(f"⚡ Aggressive Thresholds:")
            print(f"   Buy: -0.3% | Sell: +0.5% | Lookback: 5 periods")
            
            # Check if we're past expected end time
            if remaining.total_seconds() <= 0:
                print(f"\n🎉 AGGRESSIVE TEST COMPLETED!")
                
                # Final summary
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
                            print(f"🔄 Total Trades Executed: {len(trades_detected)}")
                            
                            if trades_detected:
                                print(f"\n🎯 TRADE ANALYSIS:")
                                buy_trades = [t for t in trades_detected if t.get('action') == 'buy']
                                sell_trades = [t for t in trades_detected if t.get('action') == 'sell']
                                
                                print(f"   Buy Orders: {len(buy_trades)}")
                                print(f"   Sell Orders: {len(sell_trades)}")
                                
                                if sell_trades:
                                    total_profit = sum(t.get('profit', 0) for t in sell_trades)
                                    avg_profit = total_profit / len(sell_trades)
                                    profitable = len([t for t in sell_trades if t.get('profit', 0) > 0])
                                    win_rate = (profitable / len(sell_trades)) * 100
                                    
                                    print(f"   Total Profit: ${total_profit:.2f}")
                                    print(f"   Average Profit: ${avg_profit:.2f}")
                                    print(f"   Win Rate: {win_rate:.1f}%")
                                    
                                    print(f"\n✅ SUCCESS: Aggressive thresholds executed trades!")
                                else:
                                    print(f"\n⚠️ Trades opened but none closed yet")
                            else:
                                print(f"\n❌ NO TRADES EXECUTED")
                                print("Need even more aggressive thresholds!")
                    
                    except Exception as e:
                        print(f"❌ Error reading final data: {e}")
                
                break
            
            print("-" * 50)
            print("⏳ Next check in 2 minutes... (Ctrl+C to stop)")
            time.sleep(120)  # Check every 2 minutes for aggressive monitoring
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Monitoring stopped by user")
        print(f"🔄 Aggressive test session is still running")
        print(f"💡 Check data directory: {session_dir}")

if __name__ == "__main__":
    monitor_aggressive_session()