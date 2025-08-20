"""
Real-time session monitoring and analysis companion.
Provides live updates and analysis of the active trading session.
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd

class SessionMonitor:
    """Monitors active trading sessions and provides real-time analysis."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.active_session = None
        self.last_update = None
    
    def find_active_session(self):
        """Find the most recent active trading session."""
        session_dirs = list(self.data_dir.glob("live_session_*"))
        
        if not session_dirs:
            return None
        
        # Find the most recent session
        latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
        
        # Check if it's active (modified within last 10 minutes)
        last_modified = datetime.fromtimestamp(latest_session.stat().st_mtime)
        if datetime.now() - last_modified < timedelta(minutes=10):
            return latest_session
        
        return None
    
    def load_session_data(self, session_dir):
        """Load all session data files."""
        data = {}
        
        files_to_load = [
            'session_metadata.json',
            'trades.json',
            'portfolio_history.json',
            'performance_summary.json',
            'price_history.json'
        ]
        
        for filename in files_to_load:
            file_path = session_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data[filename.replace('.json', '')] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        return data
    
    def analyze_session_performance(self, data):
        """Analyze current session performance."""
        if 'performance_summary' not in data:
            return {}
        
        perf = data['performance_summary']
        metadata = data.get('session_metadata', {})
        
        # Calculate session metrics
        start_time = datetime.fromisoformat(metadata.get('start_time', datetime.now().isoformat()))
        runtime_hours = (datetime.now() - start_time).total_seconds() / 3600
        
        analysis = {
            "session_info": {
                "session_id": metadata.get('session_id', 'Unknown'),
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "runtime_hours": runtime_hours,
                "runtime_formatted": str(timedelta(seconds=int((datetime.now() - start_time).total_seconds())))
            },
            "performance": {
                "initial_balance": perf.get('initial_balance', 0),
                "current_value": perf.get('current_value', 0),
                "total_return_pct": perf.get('total_return_pct', 0),
                "total_return_cad": perf.get('total_return_cad', 0),
                "total_trades": perf.get('total_trades', 0),
                "total_fees": perf.get('total_fees_paid', 0)
            },
            "projections": {
                "hourly_return": perf.get('total_return_pct', 0) / runtime_hours if runtime_hours > 0 else 0,
                "daily_projection": perf.get('total_return_pct', 0) / runtime_hours * 24 if runtime_hours > 0 else 0,
                "weekly_projection": perf.get('total_return_pct', 0) / runtime_hours * 24 * 7 if runtime_hours > 0 else 0,
                "monthly_projection": perf.get('total_return_pct', 0) / runtime_hours * 24 * 30 if runtime_hours > 0 else 0
            }
        }
        
        return analysis
    
    def analyze_trading_activity(self, data):
        """Analyze recent trading activity."""
        if 'trades' not in data or not data['trades']:
            return {"message": "No trades executed yet"}
        
        trades = data['trades']
        recent_trades = [t for t in trades if 
                        datetime.fromisoformat(t['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        # Group by pair
        pair_activity = {}
        for trade in trades:
            pair = trade['pair']
            if pair not in pair_activity:
                pair_activity[pair] = {'buy': 0, 'sell': 0, 'volume': 0}
            
            pair_activity[pair][trade['side']] += 1
            pair_activity[pair]['volume'] += trade['cost']
        
        return {
            "total_trades": len(trades),
            "recent_trades_1h": len(recent_trades),
            "pair_activity": pair_activity,
            "latest_trade": trades[-1] if trades else None
        }
    
    def analyze_market_conditions(self, data):
        """Analyze current market conditions."""
        if 'price_history' not in data:
            return {}
        
        price_history = data['price_history']
        conditions = {}
        
        for pair, prices in price_history.items():
            if not prices or len(prices) < 2:
                continue
            
            recent_prices = prices[-10:]  # Last 10 data points
            current_price = prices[-1]['price']
            start_price = prices[0]['price']
            
            # Calculate volatility
            price_values = [p['price'] for p in recent_prices]
            if len(price_values) > 1:
                returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                          for i in range(1, len(price_values))]
                volatility = (sum(r**2 for r in returns) / len(returns))**0.5
            else:
                volatility = 0
            
            conditions[pair] = {
                "current_price": current_price,
                "session_start_price": start_price,
                "session_movement_pct": (current_price - start_price) / start_price * 100,
                "recent_volatility": volatility * 100,  # Convert to percentage
                "min_price": min(p['price'] for p in prices),
                "max_price": max(p['price'] for p in prices),
                "price_range_pct": (max(p['price'] for p in prices) - min(p['price'] for p in prices)) / min(p['price'] for p in prices) * 100
            }
        
        return conditions
    
    def print_session_status(self, analysis, trading_activity, market_conditions):
        """Print comprehensive session status."""
        print("\n" + "="*80)
        print("🚀 LIVE TRADING SESSION STATUS")
        print("="*80)
        
        # Session Info
        session_info = analysis.get('session_info', {})
        print(f"📅 Session: {session_info.get('session_id', 'Unknown')}")
        print(f"⏰ Started: {session_info.get('start_time', 'Unknown')}")
        print(f"🕐 Runtime: {session_info.get('runtime_formatted', 'Unknown')}")
        
        # Performance
        perf = analysis.get('performance', {})
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Balance: ${perf.get('initial_balance', 0):.2f}")
        print(f"   Current Value:   ${perf.get('current_value', 0):.2f}")
        print(f"   Total Return:    {perf.get('total_return_pct', 0):+.2f}% (${perf.get('total_return_cad', 0):+.2f})")
        print(f"   Total Trades:    {perf.get('total_trades', 0)}")
        print(f"   Total Fees:      ${perf.get('total_fees', 0):.2f}")
        
        # Projections
        proj = analysis.get('projections', {})
        print(f"\n📊 RETURN PROJECTIONS:")
        print(f"   Hourly:   {proj.get('hourly_return', 0):+.3f}%")
        print(f"   Daily:    {proj.get('daily_projection', 0):+.2f}%")
        print(f"   Weekly:   {proj.get('weekly_projection', 0):+.2f}%")
        print(f"   Monthly:  {proj.get('monthly_projection', 0):+.2f}%")
        
        # Trading Activity
        print(f"\n🔄 TRADING ACTIVITY:")
        print(f"   Total Trades: {trading_activity.get('total_trades', 0)}")
        print(f"   Recent (1h):  {trading_activity.get('recent_trades_1h', 0)}")
        
        pair_activity = trading_activity.get('pair_activity', {})
        for pair, activity in pair_activity.items():
            print(f"   {pair}: {activity['buy']} buys, {activity['sell']} sells, ${activity['volume']:.2f} volume")
        
        latest_trade = trading_activity.get('latest_trade')
        if latest_trade:
            trade_time = datetime.fromisoformat(latest_trade['timestamp']).strftime('%H:%M:%S')
            print(f"   Latest: {latest_trade['side'].upper()} {latest_trade['pair']} at {trade_time}")
        
        # Market Conditions
        print(f"\n📈 MARKET CONDITIONS:")
        for pair, conditions in market_conditions.items():
            print(f"   {pair}:")
            print(f"     Current: ${conditions['current_price']:,.2f}")
            print(f"     Session: {conditions['session_movement_pct']:+.2f}%")
            print(f"     Volatility: {conditions['recent_volatility']:.2f}%")
            print(f"     Range: ${conditions['min_price']:,.2f} - ${conditions['max_price']:,.2f}")
        
        print("="*80)
    
    def monitor_session(self, update_interval=60):
        """Monitor active session with regular updates."""
        logger.info("🔍 Starting session monitor...")
        
        while True:
            try:
                # Find active session
                active_session = self.find_active_session()
                
                if not active_session:
                    if self.active_session:
                        print("\n⚠️  No active session found. Session may have ended.")
                        self.active_session = None
                    else:
                        print("⏳ Waiting for active trading session...")
                    time.sleep(update_interval)
                    continue
                
                # Check if this is a new session
                if self.active_session != active_session:
                    print(f"\n🎯 Found active session: {active_session.name}")
                    self.active_session = active_session
                
                # Load and analyze session data
                data = self.load_session_data(active_session)
                
                if data:
                    analysis = self.analyze_session_performance(data)
                    trading_activity = self.analyze_trading_activity(data)
                    market_conditions = self.analyze_market_conditions(data)
                    
                    # Print status
                    self.print_session_status(analysis, trading_activity, market_conditions)
                    
                    # Update timestamp
                    self.last_update = datetime.now()
                    print(f"🔄 Last updated: {self.last_update.strftime('%H:%M:%S')}")
                    print(f"📊 Dashboard: http://localhost:8505")
                    print(f"📁 Data: {active_session}")
                
                # Wait for next update
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\n👋 Session monitor stopped.")
                break
            except Exception as e:
                logger.error(f"Error in session monitor: {e}")
                time.sleep(update_interval)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Session Monitor")
    parser.add_argument('--interval', type=int, default=60, 
                       help='Update interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    monitor = SessionMonitor()
    monitor.monitor_session(args.interval)

if __name__ == "__main__":
    main()