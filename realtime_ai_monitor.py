#!/usr/bin/env python3
"""
Real-Time AI Monitor
Shows live AI decision-making as it happens
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import os

class RealTimeAIMonitor:
    """Monitor AI decisions in real-time"""
    
    def __init__(self):
        self.running = True
        self.last_step = 0
        self.session_dir = None
    
    def clear_screen(self):
        """Clear screen"""
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    def find_active_session(self):
        """Find the most recent active session"""
        data_dir = Path("e:/KrakenBot/data")
        
        # Look for active_trading sessions first
        active_sessions = list(data_dir.glob("active_trading_*"))
        if active_sessions:
            latest = max(active_sessions, key=lambda x: x.stat().st_mtime)
            return latest
        
        # Fallback to enhanced_sim sessions
        enhanced_sessions = list(data_dir.glob("enhanced_sim_*"))
        if enhanced_sessions:
            latest = max(enhanced_sessions, key=lambda x: x.stat().st_mtime)
            return latest
        
        return None
    
    def load_latest_data(self, session_dir):
        """Load the latest portfolio and opportunities data"""
        try:
            # Portfolio history
            portfolio_file = session_dir / "portfolio_history.json"
            portfolio_data = []
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
            
            # Opportunities
            opportunities_file = session_dir / "opportunities.json"
            opportunities_data = []
            if opportunities_file.exists():
                with open(opportunities_file, 'r') as f:
                    opportunities_data = json.load(f)
            
            # Trades
            trades_file = session_dir / "trades.json"
            trades_data = []
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
            
            return portfolio_data, opportunities_data, trades_data
        except Exception as e:
            return [], [], []
    
    def display_live_status(self, session_dir, portfolio_data, opportunities_data, trades_data):
        """Display live AI status"""
        self.clear_screen()
        
        session_name = session_dir.name
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print("╔" + "═" * 78 + "╗")
        print(f"║{'KRAKENBOT LIVE AI MONITOR - REAL-TIME DECISIONS':^78}║")
        print("╠" + "═" * 78 + "╣")
        print(f"║ 🔴 LIVE │ Session: {session_name:<25} │ Time: {current_time} ║")
        
        if portfolio_data:
            latest = portfolio_data[-1]
            portfolio_value = latest.get('portfolio_value', 1000.0)
            total_return = latest.get('total_return', 0.0)
            step = latest.get('step', 0)
            
            print(f"║ 💰 Portfolio: ${portfolio_value:,.2f} ({total_return:+.3f}%) │ Step: {step:<4} │ Trades: {len(trades_data):<3} ║")
        else:
            print(f"║ 💰 Portfolio: $1,000.00 (+0.000%) │ Step: 0    │ Trades: 0   ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Recent opportunities
        print(f"║ 🎯 RECENT AI DECISIONS (LAST 5)                                     ║")
        if opportunities_data:
            recent_opps = opportunities_data[-5:]
            for i, opp in enumerate(recent_opps, 1):
                timestamp = opp.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime('%H:%M:%S')
                    except:
                        timestamp = 'Unknown'
                
                pair = opp.get('pair', 'Unknown')[:8]
                signal = opp.get('signal', 'HOLD')
                confidence = opp.get('confidence', 0)
                executed = "✅" if opp.get('executed', False) else "❌"
                
                print(f"║ {i}. {timestamp} │ {pair:<8} │ {signal:<4} │ {confidence:5.1f}% │ {executed}                    ║")
        else:
            print(f"║   No opportunities found yet - AI is analyzing...                    ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Recent trades
        print(f"║ 💸 RECENT TRADES (LAST 3)                                           ║")
        if trades_data:
            recent_trades = trades_data[-3:]
            for i, trade in enumerate(recent_trades, 1):
                timestamp = trade.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime('%H:%M:%S')
                    except:
                        timestamp = 'Unknown'
                
                pair = trade.get('pair', 'Unknown')[:8]
                side = trade.get('side', 'unknown').upper()
                amount = trade.get('amount', 0)
                price = trade.get('price', 0)
                
                print(f"║ {i}. {timestamp} │ {pair:<8} │ {side:<4} │ ${amount:6.2f} @ ${price:8.2f}        ║")
        else:
            print(f"║   No trades executed yet                                             ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Live insights
        if portfolio_data:
            latest = portfolio_data[-1]
            insights = latest.get('insights', [])
            print(f"║ 🧠 LIVE AI INSIGHTS                                                 ║")
            for insight in insights[-3:]:  # Show last 3 insights
                insight_text = insight[:65] if len(insight) > 65 else insight
                print(f"║   • {insight_text:<65} ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Status
        if "active_trading" in session_name:
            print(f"║ 🚀 STATUS: ACTIVE TRADING MODE - Lower thresholds for more trades   ║")
        else:
            print(f"║ 🐌 STATUS: CONSERVATIVE MODE - High thresholds, fewer trades        ║")
        
        print(f"║ 🎮 Press Ctrl+C to exit │ Auto-refresh every 5 seconds             ║")
        print("╚" + "═" * 78 + "╝")
        print()
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Find active session
                session_dir = self.find_active_session()
                if not session_dir:
                    self.clear_screen()
                    print("❌ No active simulation session found!")
                    print("   Start one with: python active_trading_simulation.py")
                    await asyncio.sleep(5)
                    continue
                
                # Load data
                portfolio_data, opportunities_data, trades_data = self.load_latest_data(session_dir)
                
                # Display status
                self.display_live_status(session_dir, portfolio_data, opportunities_data, trades_data)
                
                # Check if new activity
                current_step = len(portfolio_data)
                if current_step > self.last_step:
                    self.last_step = current_step
                    # Flash indicator for new activity
                    print("🔥 NEW ACTIVITY DETECTED!")
                
                # Wait before refresh
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(5)

async def main():
    """Main entry point"""
    print("🔴 Starting Real-Time AI Monitor...")
    
    monitor = RealTimeAIMonitor()
    
    try:
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())