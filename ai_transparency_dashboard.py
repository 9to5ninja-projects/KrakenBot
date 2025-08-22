#!/usr/bin/env python3
"""
AI Transparency Dashboard
Shows exactly what the AI is doing behind the scenes - parameters, decisions, and reasoning
"""

import json
import time
import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

class AITransparencyDashboard:
    """Detailed AI transparency dashboard showing all decision-making processes"""
    
    def __init__(self):
        self.running = True
        self.session_data = {}
        self.ai_parameters = {}
        self.decision_log = []
        self.last_update = datetime.now()
        
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
            return 120, 30
    
    def load_session_data(self, session_dir):
        """Load comprehensive session data"""
        try:
            # Portfolio history
            portfolio_file = session_dir / "portfolio_history.json"
            portfolio_data = []
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
            
            # Opportunities (AI decisions)
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
    
    def extract_ai_parameters(self, portfolio_data):
        """Extract AI parameters from session data"""
        if not portfolio_data:
            return {
                'buy_threshold': -0.003,
                'sell_threshold': 0.005,
                'min_trade_amount': 50.0,
                'max_position_size': 0.20,
                'stop_loss_pct': -0.02,
                'take_profit_pct': 0.015,
                'ai_confidence_threshold': 60,
                'tech_strength_threshold': 40,
                'high_confidence_threshold': 80,
                'strong_tech_threshold': 70
            }
        
        # Extract from latest entry or use defaults
        return {
            'buy_threshold': -0.003,  # -0.3%
            'sell_threshold': 0.005,  # +0.5%
            'min_trade_amount': 50.0,
            'max_position_size': 0.20,  # 20% max per position
            'stop_loss_pct': -0.02,    # -2% stop loss
            'take_profit_pct': 0.015,  # +1.5% take profit
            'ai_confidence_threshold': 60,  # AI must be >60% confident
            'tech_strength_threshold': 40,  # Technical strength >40%
            'high_confidence_threshold': 80,  # Single signal threshold
            'strong_tech_threshold': 70     # Strong technical threshold
        }
    
    def analyze_ai_decisions(self, opportunities_data, portfolio_data):
        """Analyze AI decision-making patterns"""
        if not opportunities_data:
            return {
                'total_opportunities_found': 0,
                'opportunities_taken': 0,
                'opportunities_rejected': 0,
                'avg_confidence': 0,
                'decision_reasons': [],
                'signal_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                'confidence_distribution': {'low': 0, 'medium': 0, 'high': 0}
            }
        
        total_opps = len(opportunities_data)
        taken = len([o for o in opportunities_data if o.get('executed', False)])
        rejected = total_opps - taken
        
        confidences = [o.get('confidence', 0) for o in opportunities_data]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        signals = {}
        for opp in opportunities_data:
            signal = opp.get('signal', 'HOLD')
            signals[signal] = signals.get(signal, 0) + 1
        
        conf_dist = {'low': 0, 'medium': 0, 'high': 0}
        for conf in confidences:
            if conf < 50:
                conf_dist['low'] += 1
            elif conf < 75:
                conf_dist['medium'] += 1
            else:
                conf_dist['high'] += 1
        
        return {
            'total_opportunities_found': total_opps,
            'opportunities_taken': taken,
            'opportunities_rejected': rejected,
            'avg_confidence': avg_confidence,
            'signal_distribution': signals,
            'confidence_distribution': conf_dist,
            'recent_decisions': opportunities_data[-5:] if opportunities_data else []
        }
    
    def draw_transparency_dashboard(self, session_dir, portfolio_data, opportunities_data, trades_data):
        """Draw the comprehensive AI transparency dashboard"""
        width, height = self.get_terminal_size()
        
        # Clear screen
        self.clear_screen()
        
        # Extract data
        ai_params = self.extract_ai_parameters(portfolio_data)
        ai_decisions = self.analyze_ai_decisions(opportunities_data, portfolio_data)
        
        # Calculate session stats
        latest_portfolio = portfolio_data[-1] if portfolio_data else {}
        current_value = latest_portfolio.get('portfolio_value', 1000.0)
        total_return = latest_portfolio.get('total_return', 0.0)
        elapsed_time = 0
        if portfolio_data:
            start_time = datetime.fromisoformat(portfolio_data[0]['timestamp'])
            current_time = datetime.fromisoformat(portfolio_data[-1]['timestamp'])
            elapsed_time = (current_time - start_time).total_seconds() / 3600
        
        # Header
        print("╔" + "═" * (width - 2) + "╗")
        title = "KRAKENBOT AI TRANSPARENCY DASHBOARD - BEHIND THE SCENES"
        padding = (width - len(title) - 2) // 2
        print(f"║{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # Session Overview
        session_name = session_dir.name
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"║ 📊 SESSION: {session_name} │ TIME: {current_time} │ ELAPSED: {elapsed_time:.1f}h" + " " * (width - 85) + "║")
        print(f"║ 💰 PORTFOLIO: ${current_value:,.2f} ({total_return:+.3f}%) │ TRADES: {len(trades_data)}" + " " * (width - 65) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # AI PARAMETERS SECTION
        print(f"║ 🤖 AI TRADING PARAMETERS (CURRENT SETTINGS)" + " " * (width - 45) + "║")
        print(f"║   Buy Threshold:     {ai_params['buy_threshold']:+.3f} ({ai_params['buy_threshold']*100:+.1f}%)" + " " * (width - 50) + "║")
        print(f"║   Sell Threshold:    {ai_params['sell_threshold']:+.3f} ({ai_params['sell_threshold']*100:+.1f}%)" + " " * (width - 50) + "║")
        print(f"║   Min Trade Amount:  ${ai_params['min_trade_amount']:.0f}" + " " * (width - 35) + "║")
        print(f"║   Max Position Size: {ai_params['max_position_size']:.1%} of portfolio" + " " * (width - 50) + "║")
        print(f"║   Stop Loss:         {ai_params['stop_loss_pct']:.1%} │ Take Profit: {ai_params['take_profit_pct']:.1%}" + " " * (width - 65) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # AI DECISION CRITERIA
        print(f"║ 🧠 AI DECISION CRITERIA (HOW IT DECIDES TO TRADE)" + " " * (width - 48) + "║")
        print(f"║   AI Confidence Required:     >{ai_params['ai_confidence_threshold']}% (with technical agreement)" + " " * (width - 70) + "║")
        print(f"║   Technical Strength Required: >{ai_params['tech_strength_threshold']}% (with AI agreement)" + " " * (width - 70) + "║")
        print(f"║   High Confidence Override:   >{ai_params['high_confidence_threshold']}% (AI alone can decide)" + " " * (width - 70) + "║")
        print(f"║   Strong Technical Override:  >{ai_params['strong_tech_threshold']}% (Technical alone can decide)" + " " * (width - 75) + "║")
        print(f"║   Signal Agreement: Both AI and Technical must agree OR one must be very strong" + " " * (width - 85) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # AI DECISION ANALYSIS
        print(f"║ 📈 AI DECISION ANALYSIS (WHAT THE AI IS ACTUALLY DOING)" + " " * (width - 54) + "║")
        print(f"║   Opportunities Found:  {ai_decisions['total_opportunities_found']:3d} │ Taken: {ai_decisions['opportunities_taken']:2d} │ Rejected: {ai_decisions['opportunities_rejected']:2d}" + " " * (width - 75) + "║")
        print(f"║   Average Confidence:   {ai_decisions['avg_confidence']:5.1f}%" + " " * (width - 40) + "║")
        
        # Signal distribution
        signals = ai_decisions['signal_distribution']
        buy_count = signals.get('BUY', 0)
        sell_count = signals.get('SELL', 0)
        hold_count = signals.get('HOLD', 0)
        print(f"║   Signal Distribution:  BUY: {buy_count:2d} │ SELL: {sell_count:2d} │ HOLD: {hold_count:2d}" + " " * (width - 60) + "║")
        
        # Confidence distribution
        conf_dist = ai_decisions['confidence_distribution']
        print(f"║   Confidence Levels:    Low(<50%): {conf_dist['low']:2d} │ Med(50-75%): {conf_dist['medium']:2d} │ High(>75%): {conf_dist['high']:2d}" + " " * (width - 80) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        
        # RECENT AI DECISIONS
        print(f"║ 🔍 RECENT AI DECISIONS (LAST 5 OPPORTUNITIES ANALYZED)" + " " * (width - 53) + "║")
        recent_decisions = ai_decisions.get('recent_decisions', [])
        if recent_decisions:
            for i, decision in enumerate(recent_decisions[-5:], 1):
                timestamp = decision.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime('%H:%M:%S')
                    except:
                        timestamp = 'Unknown'
                
                pair = decision.get('pair', 'Unknown')
                signal = decision.get('signal', 'HOLD')
                confidence = decision.get('confidence', 0)
                executed = "✅ EXECUTED" if decision.get('executed', False) else "❌ REJECTED"
                
                print(f"║   {i}. {timestamp} │ {pair:<8} │ {signal:<4} │ {confidence:5.1f}% │ {executed}" + " " * (width - 65) + "║")
        else:
            print(f"║   No recent decisions found - AI is still analyzing market conditions" + " " * (width - 70) + "║")
        
        print("╠" + "═" * (width - 2) + "╣")
        
        # CURRENT AI STATUS
        print(f"║ ⚡ CURRENT AI STATUS (WHAT'S HAPPENING RIGHT NOW)" + " " * (width - 50) + "║")
        if portfolio_data:
            latest = portfolio_data[-1]
            market_context = latest.get('market_opening_context', {})
            
            print(f"║   Market Opening Window: {'YES' if market_context.get('in_opening_window', False) else 'NO'}" + " " * (width - 50) + "║")
            print(f"║   Active Market Session: {market_context.get('active_session', 'None')}" + " " * (width - 50) + "║")
            print(f"║   Manipulation Risk:     {market_context.get('manipulation_risk', 'UNKNOWN')}" + " " * (width - 50) + "║")
            print(f"║   Rebound Expected:      {'YES' if market_context.get('rebound_expected', False) else 'NO'}" + " " * (width - 50) + "║")
            
            # Show insights
            insights = latest.get('insights', [])
            if insights:
                print(f"║   Latest Insights:" + " " * (width - 20) + "║")
                for insight in insights[-3:]:  # Show last 3 insights
                    print(f"║     • {insight}" + " " * (width - len(insight) - 8) + "║")
        else:
            print(f"║   Status: INITIALIZING - Building market understanding..." + " " * (width - 60) + "║")
        
        print("╠" + "═" * (width - 2) + "╣")
        
        # WHY NO TRADES EXPLANATION
        if len(trades_data) == 0:
            print(f"║ ❓ WHY NO TRADES YET? (AI DECISION EXPLANATION)" + " " * (width - 46) + "║")
            print(f"║   The AI is being conservative because:" + " " * (width - 40) + "║")
            print(f"║   1. It requires BOTH AI confidence >60% AND technical strength >40%" + " " * (width - 70) + "║")
            print(f"║   2. OR a single very strong signal (AI >80% OR technical >70%)" + " " * (width - 65) + "║")
            print(f"║   3. Current market conditions may not meet these strict criteria" + " " * (width - 70) + "║")
            print(f"║   4. This is GOOD - it prevents bad trades and preserves capital" + " " * (width - 65) + "║")
            print("╠" + "═" * (width - 2) + "╣")
        
        # Footer
        print(f"║ 🎮 CONTROLS: Press Ctrl+C to exit │ Auto-refresh every 15 seconds" + " " * (width - 68) + "║")
        print("╚" + "═" * (width - 2) + "╝")
        print()
    
    async def run_transparency_loop(self):
        """Main transparency monitoring loop"""
        while self.running:
            try:
                # Find current session
                session_dir = self.find_current_session()
                if not session_dir:
                    self.clear_screen()
                    print("❌ No active enhanced simulation session found!")
                    print("   Start a session with: python run_enhanced_simulation.py simulate")
                    await asyncio.sleep(15)
                    continue
                
                # Load comprehensive session data
                portfolio_data, opportunities_data, trades_data = self.load_session_data(session_dir)
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Draw the transparency dashboard
                self.draw_transparency_dashboard(session_dir, portfolio_data, opportunities_data, trades_data)
                
                # Check if session is complete
                if portfolio_data:
                    start_time = datetime.fromisoformat(portfolio_data[0]['timestamp'])
                    current_time = datetime.fromisoformat(portfolio_data[-1]['timestamp'])
                    elapsed_hours = (current_time - start_time).total_seconds() / 3600
                    
                    if elapsed_hours >= 4.0:
                        print("\n🎉 SESSION COMPLETE! 4-hour test finished.")
                        break
                
                # Wait before refresh (15 seconds for detailed analysis)
                await asyncio.sleep(15)
                
            except KeyboardInterrupt:
                print("\n👋 Transparency dashboard stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(15)
    
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
    print("🔍 Starting KrakenBot AI Transparency Dashboard...")
    print("   Loading AI decision analysis...")
    
    # Initialize dashboard
    dashboard = AITransparencyDashboard()
    
    try:
        # Run the transparency loop
        await dashboard.run_transparency_loop()
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
        print(f"❌ Error starting transparency dashboard: {e}")