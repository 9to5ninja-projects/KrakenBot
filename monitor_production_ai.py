#!/usr/bin/env python3
"""
Production AI Trading Monitor
Real-time monitoring with alerts and status dashboard
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time

class ProductionAIMonitor:
    """Monitor for production AI trading sessions"""
    
    def __init__(self):
        self.data_dir = Path("e:/KrakenBot/data")
        self.current_session = None
        self.last_update = None
        self.alert_thresholds = {
            'max_loss_percent': -10.0,  # Alert if loss > 10%
            'no_trades_minutes': 60,    # Alert if no trades for 1 hour
            'consecutive_errors': 5,    # Alert if too many errors
        }
        
    def find_latest_production_session(self):
        """Find the latest production AI session"""
        try:
            sessions = [d for d in self.data_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('production_ai_')]
            
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
                'portfolio_state': {},
                'config': {}
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
            
            # Load config
            config_file = session_dir / "session_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data['config'] = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error loading session data: {e}")
            return None
    
    def check_alerts(self, data):
        """Check for alert conditions"""
        alerts = []
        
        portfolio_state = data.get('portfolio_state', {})
        trades = data.get('trades', [])
        
        # Check loss threshold
        total_return = portfolio_state.get('total_return', 0) * 100
        if total_return < self.alert_thresholds['max_loss_percent']:
            alerts.append(f"🚨 HIGH LOSS: {total_return:.2f}% (threshold: {self.alert_thresholds['max_loss_percent']}%)")
        
        # Check for recent trades
        if trades:
            last_trade_time = datetime.fromisoformat(trades[-1]['timestamp'])
            minutes_since_trade = (datetime.now() - last_trade_time).total_seconds() / 60
            if minutes_since_trade > self.alert_thresholds['no_trades_minutes']:
                alerts.append(f"⚠️ NO TRADES: {minutes_since_trade:.0f} minutes since last trade")
        
        # Check consecutive errors
        consecutive_errors = portfolio_state.get('consecutive_errors', 0)
        if consecutive_errors >= self.alert_thresholds['consecutive_errors']:
            alerts.append(f"🔥 ERRORS: {consecutive_errors} consecutive errors")
        
        return alerts
    
    def format_performance_summary(self, data):
        """Format performance summary"""
        portfolio_state = data.get('portfolio_state', {})
        trades = data.get('trades', [])
        config = data.get('config', {})
        
        lines = []
        
        # Basic performance
        initial = portfolio_state.get('initial_capital', 1000)
        current = portfolio_state.get('current_capital', 1000)
        total_return = portfolio_state.get('total_return', 0) * 100
        runtime_hours = portfolio_state.get('session_runtime_hours', 0)
        
        lines.append(f"💰 Capital: ${current:.2f} ({total_return:+.2f}%) | Runtime: {runtime_hours:.1f}h")
        
        # Trading stats
        total_trades = len(trades)
        daily_trades = portfolio_state.get('daily_trades', 0)
        daily_limit = config.get('daily_trade_limit', 50)
        
        lines.append(f"📊 Trades: {total_trades} total, {daily_trades}/{daily_limit} today")
        
        # Risk metrics
        max_daily_loss = config.get('max_daily_loss', 0.05) * 100
        max_position_risk = config.get('max_position_risk', 0.15) * 100
        
        lines.append(f"🛡️ Limits: {max_daily_loss:.1f}% daily loss, {max_position_risk:.1f}% position risk")
        
        # Recent performance
        if trades:
            recent_trades = trades[-5:]
            profitable = sum(1 for t in recent_trades if t.get('profit_loss', 0) > 0)
            lines.append(f"📈 Recent: {profitable}/{len(recent_trades)} profitable trades")
        
        return lines
    
    def format_positions_summary(self, portfolio_state):
        """Format active positions"""
        positions = portfolio_state.get('positions', {})
        active_positions = {k: v for k, v in positions.items() if v.get('quantity', 0) > 0}
        
        if not active_positions:
            return ["🎯 No active positions"]
        
        lines = [f"🎯 Active Positions ({len(active_positions)}):"]
        for pair, pos in active_positions.items():
            qty = pos.get('quantity', 0)
            avg_price = pos.get('avg_price', 0)
            total_cost = pos.get('total_cost', 0)
            lines.append(f"   • {pair}: {qty:.6f} @ ${avg_price:.2f} (${total_cost:.2f})")
        
        return lines
    
    def format_recent_activity(self, trades, opportunities):
        """Format recent trading activity"""
        lines = []
        
        # Recent trades
        if trades:
            recent_trades = trades[-3:]
            lines.append("🔥 Recent Trades:")
            for trade in recent_trades:
                timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
                pair = trade['pair']
                signal = trade['signal']
                confidence = trade['confidence']
                profit_loss = trade.get('profit_loss', 0)
                
                signal_color = "🟢" if signal == "BUY" else "🔴"
                pl_text = f" (P/L: {profit_loss:+.2f})" if profit_loss != 0 else ""
                
                lines.append(f"   {signal_color} {timestamp} │ {pair} {signal} ({confidence:.1f}%){pl_text}")
        else:
            lines.append("🔥 No trades executed yet")
        
        # Recent opportunities
        if opportunities:
            recent_opps = opportunities[-3:]
            lines.append("🎯 Recent Opportunities:")
            for opp in recent_opps:
                timestamp = datetime.fromisoformat(opp['timestamp']).strftime('%H:%M:%S')
                pair = opp['pair']
                signal = opp['signal']
                confidence = opp['confidence']
                executed = "✅" if opp.get('executed', False) else "❌"
                
                lines.append(f"   {executed} {timestamp} │ {pair} {signal} ({confidence:.1f}%)")
        
        return lines
    
    def display_monitor(self, session_name, data):
        """Display the monitoring interface"""
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("╔" + "═" * 88 + "╗")
        print("║" + " " * 25 + "PRODUCTION AI TRADING MONITOR" + " " * 33 + "║")
        print("╠" + "═" * 88 + "╣")
        print(f"║ 🚀 LIVE │ Session: {session_name:<35} │ {current_time} ║")
        
        # Check for alerts
        alerts = self.check_alerts(data)
        if alerts:
            print("╠" + "═" * 88 + "╣")
            print("║ 🚨 ALERTS:" + " " * 77 + "║")
            for alert in alerts:
                print(f"║ {alert:<86} ║")
        
        # Performance summary
        print("╠" + "═" * 88 + "╣")
        performance_lines = self.format_performance_summary(data)
        for line in performance_lines:
            print(f"║ {line:<86} ║")
        
        # Positions
        print("╠" + "═" * 88 + "╣")
        position_lines = self.format_positions_summary(data.get('portfolio_state', {}))
        for line in position_lines:
            print(f"║ {line:<86} ║")
        
        # Recent activity
        print("╠" + "═" * 88 + "╣")
        activity_lines = self.format_recent_activity(data.get('trades', []), data.get('opportunities', []))
        for line in activity_lines:
            print(f"║ {line:<86} ║")
        
        print("╠" + "═" * 88 + "╣")
        print("║ 🔄 STATUS: Production mode with safety limits and health monitoring" + " " * 18 + "║")
        print("║ 🎮 Press Ctrl+C to exit │ Auto-refresh every 10 seconds" + " " * 25 + "║")
        print("╚" + "═" * 88 + "╝")
    
    async def run_monitor(self):
        """Run the monitoring loop"""
        
        print("🚀 Starting Production AI Monitor...")
        print("   Looking for active production trading sessions...")
        
        while True:
            try:
                # Find latest session
                session_name = self.find_latest_production_session()
                
                if not session_name:
                    print("❌ No production AI sessions found!")
                    print("   Start a session with: python production_ai_trader.py")
                    await asyncio.sleep(10)
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
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitor stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(10)

async def main():
    """Run the production AI monitor"""
    monitor = ProductionAIMonitor()
    await monitor.run_monitor()

if __name__ == "__main__":
    asyncio.run(main())