#!/usr/bin/env python3
"""
Quick AI Trading Status Checker
Provides instant status without full monitoring interface
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

def find_latest_ai_session():
    """Find the latest AI trading session"""
    data_dir = Path("e:/KrakenBot/data")
    
    # Look for both production and aggressive sessions
    sessions = []
    for pattern in ['production_ai_', 'aggressive_ai_']:
        sessions.extend([d for d in data_dir.iterdir() 
                        if d.is_dir() and d.name.startswith(pattern)])
    
    if sessions:
        latest = max(sessions, key=lambda x: x.stat().st_mtime)
        return latest.name, latest
    return None, None

def load_session_summary(session_dir):
    """Load key session data"""
    try:
        data = {}
        
        # Load portfolio state
        portfolio_file = session_dir / "portfolio_state.json"
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                data['portfolio'] = json.load(f)
        
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
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def format_status_report(session_name, data):
    """Format a concise status report"""
    
    print("=" * 60)
    print("🤖 AI TRADING STATUS REPORT")
    print("=" * 60)
    
    if not data:
        print("❌ No data available")
        return
    
    portfolio = data.get('portfolio', {})
    trades = data.get('trades', [])
    opportunities = data.get('opportunities', [])
    
    # Basic info
    print(f"📊 Session: {session_name}")
    print(f"🕐 Last Update: {portfolio.get('timestamp', 'Unknown')}")
    
    # Performance
    initial = portfolio.get('initial_capital', 1000)
    current = portfolio.get('current_capital', 1000)
    total_return = portfolio.get('total_return', 0) * 100
    runtime_hours = portfolio.get('session_runtime_hours', 0)
    
    print(f"💰 Capital: ${initial:.2f} → ${current:.2f} ({total_return:+.2f}%)")
    print(f"⏱️ Runtime: {runtime_hours:.1f} hours")
    
    # Trading activity
    print(f"🔥 Total Trades: {len(trades)}")
    print(f"🎯 Opportunities: {len(opportunities)}")
    
    # Recent activity
    if trades:
        last_trade = trades[-1]
        last_trade_time = datetime.fromisoformat(last_trade['timestamp'])
        minutes_ago = (datetime.now() - last_trade_time).total_seconds() / 60
        
        print(f"🕐 Last Trade: {minutes_ago:.0f} minutes ago")
        print(f"   └─ {last_trade['signal']} {last_trade['pair']} (Conf: {last_trade['confidence']:.1f}%)")
    else:
        print("🕐 Last Trade: None")
    
    # Positions
    positions = portfolio.get('positions', {})
    active_positions = {k: v for k, v in positions.items() if v.get('quantity', 0) > 0}
    
    print(f"🎯 Active Positions: {len(active_positions)}")
    for pair, pos in active_positions.items():
        qty = pos.get('quantity', 0)
        avg_price = pos.get('avg_price', 0)
        print(f"   └─ {pair}: {qty:.6f} @ ${avg_price:.2f}")
    
    # Health indicators
    consecutive_errors = portfolio.get('consecutive_errors', 0)
    daily_trades = portfolio.get('daily_trades', 0)
    
    if consecutive_errors > 0:
        print(f"⚠️ Consecutive Errors: {consecutive_errors}")
    
    print(f"📈 Daily Trades: {daily_trades}")
    
    # Status assessment
    print("\n🔍 STATUS ASSESSMENT:")
    
    if runtime_hours < 0.1:
        print("   🟡 Just started - monitoring...")
    elif len(trades) == 0 and runtime_hours > 1:
        print("   🔴 No trades executed - check thresholds")
    elif consecutive_errors > 3:
        print("   🔴 High error rate - needs attention")
    elif total_return < -5:
        print("   🟡 Significant losses - monitor closely")
    elif len(trades) > 0 and minutes_ago < 60:
        print("   🟢 Active and trading normally")
    else:
        print("   🟡 Running but low activity")
    
    print("=" * 60)

def main():
    """Main status check"""
    
    print("🔍 Checking AI Trading Status...")
    
    session_name, session_dir = find_latest_ai_session()
    
    if not session_name:
        print("\n❌ No AI trading sessions found!")
        print("   Start a session with:")
        print("   • python production_ai_trader.py (recommended)")
        print("   • python aggressive_ai_trader.py (testing)")
        return
    
    print(f"📡 Found session: {session_name}")
    
    data = load_session_summary(session_dir)
    format_status_report(session_name, data)
    
    print("\n🎮 QUICK COMMANDS:")
    print("   • python monitor_production_ai.py  (full monitor)")
    print("   • python production_ai_trader.py   (start new session)")
    print("   • python check_ai_status.py        (this status check)")

if __name__ == "__main__":
    main()