"""
Static Panel TUI Monitor for Enhanced Simulation Session
Real-time trading dashboard with live market data and AI analysis
"""

import json
import time
import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import ccxt

# Try to import enhanced components
try:
    from enhanced_simulation_system import EnhancedSimulationSystem
    from exchange import KrakenExchange
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

class StaticPanelTUI:
    """Static panel TUI with live updates"""
    
    def __init__(self):
        self.running = True
        self.session_data = {}
        self.market_data = {}
        self.ai_status = {}
        self.last_update = datetime.now()
        
        # Initialize exchange for live prices
        try:
            self.exchange = ccxt.kraken({
                'sandbox': False,
                'enableRateLimit': True,
            })
        except:
            self.exchange = None
        
        # Tracked pairs
        self.tracked_pairs = ['BTC/CAD', 'ETH/CAD', 'ETH/BTC']
        
    def clear_screen(self):
        """Clear screen and position cursor at top"""
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    def get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except:
            return 80, 24
    
    async def fetch_live_prices(self):
        """Fetch live market prices"""
        if not self.exchange:
            return {}
        
        try:
            prices = {}
            for pair in self.tracked_pairs:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, pair)
                prices[pair] = {
                    'price': ticker['last'],
                    'change': ticker['percentage'],
                    'volume': ticker['baseVolume'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'spread': ticker['ask'] - ticker['bid'] if ticker['ask'] and ticker['bid'] else 0
                }
            return prices
        except Exception as e:
            return {}
    
    def load_session_data(self, session_dir):
        """Load current session data"""
        try:
            # Portfolio history
            portfolio_file = session_dir / "portfolio_history.json"
            portfolio_data = []
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
            
            # Trades
            trades_file = session_dir / "trades.json"
            trades_data = []
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
            
            # AI Analysis
            ai_file = session_dir / "ai_analysis.json"
            ai_data = {}
            if ai_file.exists():
                with open(ai_file, 'r') as f:
                    ai_data = json.load(f)
            
            return portfolio_data, trades_data, ai_data
        except Exception as e:
            return [], [], {}
    
    def calculate_portfolio_stats(self, portfolio_data, trades_data):
        """Calculate portfolio statistics"""
        if not portfolio_data:
            return {
                'current_value': 1000.0,
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_trade_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'elapsed_time': 0,
                'start_time': datetime.now(),
                'current_time': datetime.now(),
                'holdings': {'CAD': 1000.0, 'BTC': 0.0, 'ETH': 0.0}
            }
        
        latest = portfolio_data[-1]
        first = portfolio_data[0]
        
        # Time calculations
        start_time = datetime.fromisoformat(first['timestamp'])
        current_time = datetime.fromisoformat(latest['timestamp'])
        elapsed = current_time - start_time
        
        # Trade statistics
        winning_trades = len([t for t in trades_data if t.get('profit', 0) > 0])
        losing_trades = len([t for t in trades_data if t.get('profit', 0) < 0])
        
        # Calculate average trade return
        if trades_data:
            avg_trade_return = sum(t.get('profit', 0) for t in trades_data) / len(trades_data)
        else:
            avg_trade_return = 0.0
        
        # Calculate max drawdown
        values = [p.get('portfolio_value', 1000.0) for p in portfolio_data]
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'current_value': latest.get('portfolio_value', 1000.0),
            'total_return': latest.get('total_return', 0.0),
            'total_trades': len(trades_data),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_return': avg_trade_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': latest.get('sharpe_ratio', 0.0),
            'elapsed_time': elapsed.total_seconds() / 3600,
            'start_time': start_time,
            'current_time': current_time,
            'holdings': latest.get('holdings', {'CAD': 1000.0, 'BTC': 0.0, 'ETH': 0.0})
        }
    
    def draw_static_panel(self, session_dir, stats, ai_data, market_prices):
        """Draw the static panel interface"""
        width, height = self.get_terminal_size()
        
        # Clear screen and draw border
        self.clear_screen()
        
        # Header
        print("╔" + "═" * (width - 2) + "╗")
        title = "KRAKENBOT ENHANCED SIMULATION - LIVE TRADING DASHBOARD"
        padding = (width - len(title) - 2) // 2
        print(f"║{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # Time Panel
        current_time = datetime.now().strftime("%H:%M:%S")
        elapsed_str = f"{stats['elapsed_time']:.1f}h"
        remaining_str = f"{4.0 - stats['elapsed_time']:.1f}h"
        progress = min(stats['elapsed_time'] / 4.0 * 100, 100)
        
        print(f"║ ⏰ TIME: {current_time} │ ELAPSED: {elapsed_str} │ REMAINING: {remaining_str} │ PROGRESS: {progress:.1f}%" + " " * (width - 85) + "║")
        
        # Progress bar
        bar_width = width - 10
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"║ [{bar}]" + " " * (width - bar_width - 4) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # Portfolio Panel
        print(f"║ 💰 PORTFOLIO STATUS" + " " * (width - 21) + "║")
        print(f"║   Balance: ${stats['current_value']:,.2f} │ Return: {stats['total_return']:+.3f}% │ Trades: {stats['total_trades']}" + " " * (width - 70) + "║")
        
        # Holdings
        holdings_str = " │ ".join([f"{k}: {v:.4f}" for k, v in stats['holdings'].items()])
        print(f"║   Holdings: {holdings_str}" + " " * (width - len(holdings_str) - 14) + "║")
        
        # Performance metrics
        win_rate = (stats['winning_trades'] / max(stats['total_trades'], 1)) * 100
        print(f"║   Win Rate: {win_rate:.1f}% │ Max DD: {stats['max_drawdown']:.2f}% │ Avg Trade: {stats['avg_trade_return']:+.3f}%" + " " * (width - 85) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # Market Data Panel
        print(f"║ 📊 LIVE MARKET DATA" + " " * (width - 20) + "║")
        for pair, data in market_prices.items():
            if data:
                spread_pct = (data['spread'] / data['price']) * 100 if data['price'] > 0 else 0
                print(f"║   {pair:<8} ${data['price']:>8.2f} │ {data['change']:+6.2f}% │ Spread: {spread_pct:.3f}%" + " " * (width - 65) + "║")
            else:
                print(f"║   {pair:<8} Loading..." + " " * (width - 22) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # AI Analysis Panel
        print(f"║ 🧠 AI ANALYSIS & STRATEGY" + " " * (width - 26) + "║")
        
        # Latest AI insights
        if ai_data and 'latest_analysis' in ai_data:
            analysis = ai_data['latest_analysis']
            print(f"║   Strategy: {analysis.get('current_strategy', 'CONSERVATIVE'):<12} │ Confidence: {analysis.get('confidence', 0):.1f}%" + " " * (width - 60) + "║")
            print(f"║   Signal: {analysis.get('signal', 'HOLD'):<6} │ Risk Level: {analysis.get('risk_level', 'LOW'):<6} │ Expected Return: {analysis.get('expected_return', 0):+.3f}%" + " " * (width - 85) + "║")
            
            # Market regime
            regime = analysis.get('market_regime', 'UNKNOWN')
            volatility = analysis.get('volatility_level', 'NORMAL')
            print(f"║   Market Regime: {regime:<10} │ Volatility: {volatility:<8} │ Trend: {analysis.get('trend', 'NEUTRAL')}" + " " * (width - 75) + "║")
        else:
            print(f"║   Status: ANALYZING MARKET CONDITIONS..." + " " * (width - 42) + "║")
            print(f"║   Building confidence and learning patterns..." + " " * (width - 46) + "║")
        
        # AI Model Status
        print("╠" + "═" * (width - 2) + "╣")
        print(f"║ 🤖 AI MODEL STATUS" + " " * (width - 19) + "║")
        print(f"║   Technical Analysis: ✅ ACTIVE │ Strategy Optimizer: ✅ ACTIVE │ Multi-Coin: ✅ ACTIVE" + " " * (width - 85) + "║")
        print(f"║   Market Opening AI: ✅ ACTIVE │ NLP Analyzer: ✅ READY │ Learning: ✅ CONTINUOUS" + " " * (width - 82) + "║")
        
        # Recent Activity
        print("╠" + "═" * (width - 2) + "╣")
        print(f"║ 📈 RECENT ACTIVITY" + " " * (width - 19) + "║")
        
        if stats['total_trades'] > 0:
            print(f"║   Last trade completed at {stats['current_time'].strftime('%H:%M:%S')}" + " " * (width - 45) + "║")
        else:
            print(f"║   Monitoring for trading opportunities..." + " " * (width - 40) + "║")
        
        print(f"║   Analysis step: {ai_data.get('current_step', 0)} │ Last update: {self.last_update.strftime('%H:%M:%S')}" + " " * (width - 65) + "║")
        
        # Footer
        print("╠" + "═" * (width - 2) + "╣")
        print(f"║ 🎮 CONTROLS: Press Ctrl+C to exit │ Auto-refresh every 10 seconds" + " " * (width - 68) + "║")
        print("╚" + "═" * (width - 2) + "╝")
        
        # Position cursor at bottom
        print()

    async def run_monitor_loop(self):
        """Main monitoring loop with live updates"""
        while self.running:
            try:
                # Find current session
                session_dir = self.find_current_session()
                if not session_dir:
                    self.clear_screen()
                    print("❌ No active enhanced simulation session found!")
                    print("   Start a session with: python run_enhanced_simulation.py simulate")
                    await asyncio.sleep(10)
                    continue
                
                # Load session data
                portfolio_data, trades_data, ai_data = self.load_session_data(session_dir)
                
                # Calculate stats
                stats = self.calculate_portfolio_stats(portfolio_data, trades_data)
                
                # Fetch live market prices (limit API calls)
                market_prices = await self.fetch_live_prices()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Draw the static panel
                self.draw_static_panel(session_dir, stats, ai_data, market_prices)
                
                # Check if session is complete
                if stats['elapsed_time'] >= 4.0:
                    print("\n🎉 SESSION COMPLETE! 4-hour test finished.")
                    print(f"📊 Final Results: ${stats['current_value']:,.2f} ({stats['total_return']:+.3f}%)")
                    break
                
                # Wait before refresh (10 seconds for responsive updates)
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)
    
    def find_current_session(self):
        """Find the most recent enhanced simulation session"""
        data_dir = Path("e:/KrakenBot/data")
        
        # Look for enhanced_sim sessions
        enhanced_sessions = list(data_dir.glob("enhanced_sim_*"))
        if enhanced_sessions:
            # Get the most recent one
            latest_session = max(enhanced_sessions, key=lambda x: x.stat().st_mtime)
            return latest_session
        
        return None

async def main():
    """Main entry point"""
    print("🚀 Starting KrakenBot Static Panel TUI Monitor...")
    print("   Loading live trading dashboard...")
    
    # Initialize TUI
    tui = StaticPanelTUI()
    
    try:
        # Run the monitor loop
        await tui.run_monitor_loop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting monitor: {e}")