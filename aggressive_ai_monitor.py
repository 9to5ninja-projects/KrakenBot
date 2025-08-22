#!/usr/bin/env python3
"""
Real-Time Aggressive AI Monitor
Monitors the aggressive AI trading session with live updates
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

class AggressiveAIMonitor:
    """Monitor for aggressive AI trading sessions"""
    
    def __init__(self):
        self.data_dir = Path("e:/KrakenBot/data")
        self.current_session = None
        self.last_update = None
        
    def find_latest_aggressive_session(self):
        """Find the latest aggressive AI session"""
        try:
            sessions = [d for d in self.data_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('aggressive_ai_')]
            
            if sessions:
                latest = max(sessions, key=lambda x: x.stat().st_mtime)
                return latest.name
            return None
            
        except Exception as e:
            print(f"Error finding session: {e}")
            return None
    
    def load_session_data(self, session_name):
        """Load session data"""
        try:
            session_dir = self.data_dir / session_name
            
            data = {
                'trades': [],
                'opportunities': [],
                'portfolio_state': {}
            }
            
            # Load trades
            trades_file = session_dir / "trades.json"
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    data['trades'] = json.load(f)
            
            # Load opportunities
            opportunities_file = session_dir / "opportunities.json"
            if opportunities_file.exists():
                with open(opportunities_file, 'r') as f:
                    data['opportunities'] = json.load(f)
            
            # Load portfolio state
            portfolio_file = session_dir / "portfolio_state.json"
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    data['portfolio_state'] = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error loading session data: {e}")
            return None
    
    def format_trade_summary(self, trades):
        """Format recent trades for display"""
        if not trades:
            return "   No trades executed yet"
        
        recent_trades = trades[-5:]  # Last 5 trades
        lines = []
        
        for trade in recent_trades:
            timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
            pair = trade['pair']
            signal = trade['signal']
            quantity = trade['quantity']
            price = trade['price']
            confidence = trade['confidence']
            
            # Color coding
            signal_color = "🟢" if signal == "BUY" else "🔴"
            
            line = f"   {signal_color} {timestamp} │ {signal} {quantity:.6f} {pair} @ ${price:.2f} (Conf: {confidence:.1f}%)"
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_portfolio_summary(self, portfolio_state, trades):
        """Format portfolio summary"""
        if not portfolio_state:
            return ["Portfolio data loading..."]
        
        lines = []
        
        # Basic stats
        initial = portfolio_state.get('initial_capital', 1000)
        current = portfolio_state.get('current_capital', 1000)
        total_return = portfolio_state.get('total_return', 0) * 100
        
        lines.append(f"💰 Capital: ${current:.2f} ({total_return:+.2f}%)")
        lines.append(f"📊 Total Trades: {len(trades)}")
        
        # Position summary
        positions = portfolio_state.get('positions', {})
        active_positions = {k: v for k, v in positions.items() if v.get('quantity', 0) > 0}
        
        if active_positions:
            lines.append(f"🎯 Active Positions: {len(active_positions)}")
            for pair, pos in active_positions.items():
                qty = pos.get('quantity', 0)
                avg_price = pos.get('avg_price', 0)
                lines.append(f"   • {pair}: {qty:.6f} @ ${avg_price:.2f}")
        else:
            lines.append("🎯 No active positions")
        
        return lines
    
    def display_monitor(self, session_name, data):
        """Display the monitoring interface"""
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        trades = data.get('trades', [])
        opportunities = data.get('opportunities', [])
        portfolio_state = data.get('portfolio_state', {})
        
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "AGGRESSIVE AI TRADING MONITOR - LIVE" + " " * 21 + "║")
        print("╠" + "═" * 78 + "╣")
        print(f"║ 🔥 LIVE │ Session: {session_name:<25} │ Time: {current_time} ║")
        
        # Portfolio summary
        portfolio_lines = self.format_portfolio_summary(portfolio_state, trades)
        for line in portfolio_lines:
            print(f"║ {line:<76} ║")
        
        print("╠" + "═" * 78 + "╣")
        print("║ 🔥 RECENT TRADES (LAST 5)" + " " * 49 + "║")
        
        trade_summary = self.format_trade_summary(trades)
        for line in trade_summary.split('\n'):
            print(f"║{line:<78}║")
        
        print("╠" + "═" * 78 + "╣")
        print("║ 🎯 RECENT OPPORTUNITIES (LAST 3)" + " " * 44 + "║")
        
        if opportunities:
            recent_opps = opportunities[-3:]
            for opp in recent_opps:
                timestamp = datetime.fromisoformat(opp['timestamp']).strftime('%H:%M:%S')
                pair = opp['pair']
                signal = opp['signal']
                confidence = opp['confidence']
                expected = opp['expected_return']
                
                signal_color = "🟢" if signal == "BUY" else "🔴"
                line = f"   {signal_color} {timestamp} │ {pair} {signal} (Conf: {confidence:.1f}%, Exp: {expected:+.4f})"
                print(f"║{line:<78}║")
        else:
            print("║   No opportunities found yet" + " " * 48 + "║")
        
        print("╠" + "═" * 78 + "╣")
        print("║ 🚀 STATUS: AGGRESSIVE MODE - Low thresholds, high activity" + " " * 18 + "║")
        print("║ 🎮 Press Ctrl+C to exit │ Auto-refresh every 3 seconds" + " " * 19 + "║")
        print("╚" + "═" * 78 + "╝")
    
    async def run_monitor(self):
        """Run the monitoring loop"""
        
        print("🔥 Starting Aggressive AI Monitor...")
        print("   Looking for active aggressive trading sessions...")
        
        while True:
            try:
                # Find latest session
                session_name = self.find_latest_aggressive_session()
                
                if not session_name:
                    print("❌ No aggressive AI sessions found!")
                    await asyncio.sleep(5)
                    continue
                
                if session_name != self.current_session:
                    self.current_session = session_name
                    print(f"📡 Monitoring session: {session_name}")
                
                # Load session data
                data = self.load_session_data(session_name)
                
                if data:
                    self.display_monitor(session_name, data)
                else:
                    print("❌ Could not load session data")
                
                # Wait before next update
                await asyncio.sleep(3)
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitor stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(5)

async def main():
    """Run the aggressive AI monitor"""
    monitor = AggressiveAIMonitor()
    await monitor.run_monitor()

if __name__ == "__main__":
    asyncio.run(main())