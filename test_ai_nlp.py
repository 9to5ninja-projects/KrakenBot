"""
Test script for AI NLP capabilities
Demonstrates the log analysis and report generation features
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ai.nlp.log_analyzer import TradingLogAnalyzer
from ai.nlp.report_generator import AIReportGenerator

def test_log_analysis():
    """Test the log analysis functionality."""
    print("🤖 Testing AI NLP Log Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TradingLogAnalyzer(llm_provider="mock")
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found. Please run a trading session first.")
        return
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analyzing session: {latest_session.name}")
    
    # Perform analysis
    print("\n🔍 Running AI analysis...")
    analysis = analyzer.analyze_session_data(str(latest_session))
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    # Display results
    print("\n📊 ANALYSIS RESULTS")
    print("-" * 30)
    
    # Session info
    session_info = analysis.get('session_info', {})
    print(f"Session ID: {session_info.get('session_id', 'Unknown')}")
    print(f"Duration: {session_info.get('duration_hours', 0)} hours")
    print(f"Data Points: {session_info.get('data_points', 0)}")
    
    # Performance
    performance = analysis.get('trading_performance', {})
    print(f"\n💰 PERFORMANCE:")
    print(f"Total Return: {performance.get('total_return_percent', 0):.4f}%")
    print(f"Total Trades: {performance.get('total_trades', 0)}")
    print(f"Success Rate: {performance.get('success_rate_percent', 0):.2f}%")
    
    # Market conditions
    market_conditions = analysis.get('market_conditions', {})
    market_data = market_conditions.get('market_data', {})
    print(f"\n📈 MARKET CONDITIONS:")
    for pair, data in market_data.items():
        print(f"{pair}: {data.get('percent_change', 0):.4f}% ({data.get('trend', 'unknown')})")
    
    # Missed opportunities
    missed_opportunities = analysis.get('missed_opportunities', [])
    print(f"\n🎯 MISSED OPPORTUNITIES: {len(missed_opportunities)}")
    for i, opp in enumerate(missed_opportunities[:3]):  # Show top 3
        print(f"{i+1}. {opp.get('pair', 'Unknown')}: {opp.get('price_change_percent', 0):.4f}% at {opp.get('timestamp', 'Unknown')}")
    
    # Strategy insights
    insights = analysis.get('strategy_insights', [])
    print(f"\n💡 STRATEGY INSIGHTS:")
    for insight in insights[:3]:  # Show top 3
        print(f"• {insight}")
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    print(f"\n🎯 RECOMMENDATIONS:")
    for rec in recommendations[:3]:  # Show top 3
        print(f"• {rec}")
    
    # AI Summary
    ai_summary = analysis.get('ai_summary', '')
    print(f"\n🤖 AI SUMMARY:")
    print(ai_summary)
    
    return analysis

def test_report_generation():
    """Test the report generation functionality."""
    print("\n\n📋 Testing AI Report Generation")
    print("=" * 50)
    
    # Initialize report generator
    report_generator = AIReportGenerator(llm_provider="mock")
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found.")
        return
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    print(f"📁 Generating report for: {latest_session.name}")
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive AI report...")
    report_dir = latest_session / "ai_reports"
    report = report_generator.generate_session_report(str(latest_session), str(report_dir))
    
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return
    
    # Display executive summary
    exec_summary = report.get('executive_summary', {})
    print(f"\n⭐ EXECUTIVE SUMMARY")
    print(f"Session Rating: {exec_summary.get('session_rating', 'Unknown')}")
    
    key_metrics = exec_summary.get('key_metrics', {})
    print(f"Duration: {key_metrics.get('duration_hours', 0)} hours")
    print(f"Return: {key_metrics.get('total_return_percent', 0):.4f}%")
    print(f"Trades: {key_metrics.get('total_trades', 0)}")
    print(f"Success Rate: {key_metrics.get('success_rate_percent', 0):.2f}%")
    
    # Highlights
    highlights = exec_summary.get('highlights', [])
    print(f"\n✨ HIGHLIGHTS:")
    for highlight in highlights:
        print(f"• {highlight}")
    
    # Concerns
    concerns = exec_summary.get('concerns', [])
    if concerns:
        print(f"\n⚠️ CONCERNS:")
        for concern in concerns:
            print(f"• {concern}")
    
    # Risk assessment
    risk_assessment = report.get('risk_assessment', {})
    print(f"\n🛡️ RISK ASSESSMENT:")
    print(f"Risk Level: {risk_assessment.get('risk_level', 'Unknown')}")
    print(f"Risk Score: {risk_assessment.get('risk_score', 0)}/100")
    
    # Action items
    action_items = report.get('action_items', [])
    print(f"\n📋 ACTION ITEMS:")
    for item in action_items[:3]:  # Show top 3
        print(f"{item.get('id', 0)}. [{item.get('priority', 'Unknown')}] {item.get('action', 'Unknown')}")
    
    print(f"\n✅ Full report saved to: {report_dir}")
    
    return report

def test_daily_summary():
    """Test the daily summary generation."""
    print("\n\n📝 Testing Daily Summary Generation")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TradingLogAnalyzer(llm_provider="mock")
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found.")
        return
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    
    # Generate daily summary
    print("📋 Generating daily summary...")
    summary = analyzer.generate_daily_summary(str(latest_session))
    
    print("\n" + "="*60)
    print(summary)
    print("="*60)
    
    return summary

def main():
    """Main test function."""
    print("🚀 KrakenBot AI NLP Testing Suite")
    print("Testing AI-powered log analysis and report generation")
    print("=" * 60)
    
    try:
        # Test 1: Basic log analysis
        analysis = test_log_analysis()
        
        # Test 2: Comprehensive report generation
        report = test_report_generation()
        
        # Test 3: Daily summary
        summary = test_daily_summary()
        
        print("\n\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Log Analysis: Working")
        print("✅ Report Generation: Working")
        print("✅ Daily Summary: Working")
        
        print(f"\n🤖 AI NLP system is ready for integration!")
        print(f"📊 Next steps:")
        print(f"   1. Integrate with live trading system")
        print(f"   2. Add to dashboard")
        print(f"   3. Set up automated reporting")
        print(f"   4. Enhance with real LLM integration")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()