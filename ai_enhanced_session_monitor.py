"""
AI-Enhanced Session Monitor
Combines real-time session monitoring with AI-powered analysis and insights
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add AI modules to path
sys.path.append(str(Path(__file__).parent))

from ai.nlp.log_analyzer import TradingLogAnalyzer
from ai.nlp.report_generator import AIReportGenerator
from session_monitor import SessionMonitor

class AIEnhancedSessionMonitor(SessionMonitor):
    """
    Enhanced session monitor with AI-powered analysis and insights.
    """
    
    def __init__(self, session_dir: str = None, ai_analysis_interval: int = 1800):
        """
        Initialize AI-enhanced session monitor.
        
        Args:
            session_dir: Session directory to monitor
            ai_analysis_interval: Interval for AI analysis in seconds (default: 30 minutes)
        """
        super().__init__()
        
        # Set the session directory
        if session_dir:
            self.session_dir = session_dir
        else:
            # Find active session
            active_session = self.find_active_session()
            if active_session:
                self.session_dir = str(active_session)
            else:
                raise ValueError("No active session found and no session_dir provided")
        
        # Initialize AI components
        self.ai_analyzer = TradingLogAnalyzer(llm_provider="mock")
        self.ai_report_generator = AIReportGenerator(llm_provider="mock")
        
        # AI analysis settings
        self.ai_analysis_interval = ai_analysis_interval
        self.last_ai_analysis = None
        self.ai_insights_history = []
        
        print("🤖 AI-Enhanced Session Monitor initialized")
        print(f"   Session Directory: {self.session_dir}")
        print(f"   AI Analysis Interval: {ai_analysis_interval/60:.1f} minutes")
    
    def run_monitoring(self, duration_minutes: int = None):
        """
        Run enhanced monitoring with AI analysis.
        
        Args:
            duration_minutes: Duration to monitor (None for indefinite)
        """
        print(f"\n🚀 Starting AI-Enhanced Session Monitoring")
        print(f"Session: {self.session_dir}")
        print(f"AI Analysis: Every {self.ai_analysis_interval/60:.1f} minutes")
        print("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes) if duration_minutes else None
        
        try:
            while True:
                current_time = datetime.now()
                
                # Check if we should stop
                if end_time and current_time >= end_time:
                    print(f"\n⏰ Monitoring duration completed ({duration_minutes} minutes)")
                    break
                
                # Run standard monitoring
                session_data = self.load_session_data(self.session_dir)
                if session_data:
                    self.analyze_session_performance(session_data)
                
                # Run AI analysis if it's time
                if self._should_run_ai_analysis():
                    self._run_ai_analysis()
                
                # Display enhanced status
                self._display_enhanced_status()
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
        finally:
            # Generate final AI report
            self._generate_final_ai_report()
            print(f"\n✅ AI-Enhanced monitoring completed")
    
    def _should_run_ai_analysis(self) -> bool:
        """Check if it's time to run AI analysis."""
        if self.last_ai_analysis is None:
            return True
        
        time_since_last = (datetime.now() - self.last_ai_analysis).total_seconds()
        return time_since_last >= self.ai_analysis_interval
    
    def _run_ai_analysis(self):
        """Run AI analysis on current session data."""
        try:
            print(f"\n🤖 Running AI Analysis...")
            
            # Perform AI analysis
            analysis = self.ai_analyzer.analyze_session_data(self.session_dir)
            
            if "error" not in analysis:
                # Store insights
                ai_insight = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis,
                    "summary": analysis.get('ai_summary', 'No summary available')
                }
                
                self.ai_insights_history.append(ai_insight)
                
                # Display key insights
                self._display_ai_insights(analysis)
                
                # Save AI analysis to session
                self._save_ai_analysis(analysis)
                
            else:
                print(f"❌ AI Analysis failed: {analysis['error']}")
            
            self.last_ai_analysis = datetime.now()
            
        except Exception as e:
            print(f"❌ Error during AI analysis: {e}")
    
    def _display_ai_insights(self, analysis: dict):
        """Display key AI insights."""
        print(f"\n🧠 AI INSIGHTS")
        print("-" * 40)
        
        # Performance insights
        performance = analysis.get('trading_performance', {})
        print(f"📊 Performance: {performance.get('total_return_percent', 0):.4f}% return")
        print(f"🎯 Activity: {performance.get('total_trades', 0)} trades executed")
        
        # Market insights
        market_conditions = analysis.get('market_conditions', {})
        market_data = market_conditions.get('market_data', {})
        
        if market_data:
            print(f"📈 Market Conditions:")
            for pair, data in market_data.items():
                trend = data.get('trend', 'unknown')
                change = data.get('percent_change', 0)
                print(f"   {pair}: {change:+.4f}% ({trend})")
        
        # Top recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:2], 1):
                print(f"   {i}. {rec}")
        
        # Missed opportunities
        missed_opportunities = analysis.get('missed_opportunities', [])
        if missed_opportunities:
            print(f"⚠️ Missed Opportunities: {len(missed_opportunities)} identified")
        
        print("-" * 40)
    
    def _display_enhanced_status(self):
        """Display enhanced status with AI insights."""
        # Display basic session info
        print(f"\n📊 SESSION STATUS: {Path(self.session_dir).name}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add AI status
        if self.ai_insights_history:
            latest_insight = self.ai_insights_history[-1]
            analysis = latest_insight['analysis']
            
            print(f"\n🤖 LATEST AI INSIGHTS:")
            
            # Quick performance summary
            performance = analysis.get('trading_performance', {})
            print(f"   Return: {performance.get('total_return_percent', 0):.4f}%")
            print(f"   Trades: {performance.get('total_trades', 0)}")
            
            # Market trend
            market_data = analysis.get('market_conditions', {}).get('market_data', {})
            if market_data:
                trends = [data.get('trend', 'unknown') for data in market_data.values()]
                bullish_count = trends.count('bullish')
                bearish_count = trends.count('bearish')
                
                if bullish_count > bearish_count:
                    print(f"   Market: Mostly bullish ({bullish_count}/{len(trends)} pairs)")
                elif bearish_count > bullish_count:
                    print(f"   Market: Mostly bearish ({bearish_count}/{len(trends)} pairs)")
                else:
                    print(f"   Market: Mixed conditions")
            
            # Time since last AI analysis
            time_since = (datetime.now() - self.last_ai_analysis).total_seconds() / 60
            print(f"   Last AI Analysis: {time_since:.1f} minutes ago")
        
        else:
            print(f"\n🤖 AI ANALYSIS: Pending (will run in {self.ai_analysis_interval/60:.1f} minutes)")
    
    def _save_ai_analysis(self, analysis: dict):
        """Save AI analysis to session directory."""
        try:
            ai_dir = Path(self.session_dir) / "ai_analysis"
            ai_dir.mkdir(exist_ok=True)
            
            # Save timestamped analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = ai_dir / f"analysis_{timestamp}.json"
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis
                }, f, indent=2)
            
            # Update latest analysis
            latest_file = ai_dir / "latest_analysis.json"
            with open(latest_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis
                }, f, indent=2)
            
        except Exception as e:
            print(f"Warning: Could not save AI analysis: {e}")
    
    def _generate_final_ai_report(self):
        """Generate comprehensive AI report at the end of monitoring."""
        try:
            print(f"\n📊 Generating Final AI Report...")
            
            # Create AI reports directory
            reports_dir = Path(self.session_dir) / "ai_reports"
            
            # Generate comprehensive report
            report = self.ai_report_generator.generate_session_report(
                self.session_dir, 
                str(reports_dir)
            )
            
            if "error" not in report:
                print(f"✅ Final AI report generated: {reports_dir}")
                
                # Display executive summary
                exec_summary = report.get('executive_summary', {})
                print(f"\n⭐ FINAL SESSION RATING: {exec_summary.get('session_rating', 'Unknown')}")
                
                key_metrics = exec_summary.get('key_metrics', {})
                print(f"📊 Final Metrics:")
                print(f"   Duration: {key_metrics.get('duration_hours', 0):.2f} hours")
                print(f"   Return: {key_metrics.get('total_return_percent', 0):.4f}%")
                print(f"   Trades: {key_metrics.get('total_trades', 0)}")
                print(f"   Success Rate: {key_metrics.get('success_rate_percent', 0):.2f}%")
                
            else:
                print(f"❌ Failed to generate final report: {report['error']}")
                
        except Exception as e:
            print(f"❌ Error generating final AI report: {e}")
    
    def get_ai_insights_summary(self) -> dict:
        """Get summary of all AI insights collected."""
        if not self.ai_insights_history:
            return {"status": "No AI insights available"}
        
        # Get latest analysis
        latest = self.ai_insights_history[-1]['analysis']
        
        # Calculate insights over time
        total_analyses = len(self.ai_insights_history)
        
        return {
            "total_ai_analyses": total_analyses,
            "latest_analysis_time": self.ai_insights_history[-1]['timestamp'],
            "latest_performance": latest.get('trading_performance', {}),
            "latest_recommendations": latest.get('recommendations', []),
            "session_rating": self._calculate_session_rating(latest),
            "ai_summary": latest.get('ai_summary', 'No summary available')
        }
    
    def _calculate_session_rating(self, analysis: dict) -> str:
        """Calculate overall session rating based on AI analysis."""
        performance = analysis.get('trading_performance', {})
        total_return = performance.get('total_return_percent', 0)
        success_rate = performance.get('success_rate_percent', 0)
        
        if total_return > 2 and success_rate > 70:
            return "Excellent"
        elif total_return > 0.5 and success_rate > 50:
            return "Good"
        elif total_return > -0.5 and success_rate > 30:
            return "Fair"
        else:
            return "Poor"

def main():
    """Main function for AI-enhanced session monitoring."""
    print("🤖 KrakenBot AI-Enhanced Session Monitor")
    print("=" * 50)
    
    # Find the most recent live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found. Please start a trading session first.")
        return
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    print(f"📁 Monitoring session: {latest_session.name}")
    
    # Initialize AI-enhanced monitor
    monitor = AIEnhancedSessionMonitor(str(latest_session), ai_analysis_interval=1800)  # 30 minutes
    
    # Ask user for monitoring duration
    try:
        duration_input = input("\nEnter monitoring duration in minutes (or press Enter for indefinite): ").strip()
        duration = int(duration_input) if duration_input else None
    except ValueError:
        duration = None
    
    # Start monitoring
    monitor.run_monitoring(duration)
    
    # Display final AI insights summary
    print(f"\n🧠 FINAL AI INSIGHTS SUMMARY")
    print("=" * 50)
    
    insights_summary = monitor.get_ai_insights_summary()
    if insights_summary.get("status") != "No AI insights available":
        print(f"Total AI Analyses: {insights_summary.get('total_ai_analyses', 0)}")
        print(f"Session Rating: {insights_summary.get('session_rating', 'Unknown')}")
        print(f"AI Summary: {insights_summary.get('ai_summary', 'No summary')}")
    else:
        print("No AI insights were generated during this monitoring session.")

if __name__ == "__main__":
    main()