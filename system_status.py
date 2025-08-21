"""
KrakenBot System Status and Feature Overview
Complete system status check and feature demonstration
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

def check_system_status():
    """Check the status of all KrakenBot components."""
    print("🚀 KrakenBot System Status Check")
    print("=" * 60)
    
    status = {
        'core_features': {},
        'ai_features': {},
        'data_status': {},
        'dashboard_status': {},
        'recommendations': []
    }
    
    # Check core features
    print("\n📊 Core Features Status:")
    print("-" * 30)
    
    core_files = {
        'Exchange Manager': 'exchange.py',
        'Arbitrage Calculator': 'arbitrage.py',
        'Simple Pair Monitor': 'run_simple_pair_monitor.py',
        'Configuration': 'config.py',
        'Dashboard': 'simple_dashboard.py'
    }
    
    for feature, filename in core_files.items():
        if Path(filename).exists():
            print(f"✅ {feature}: Available")
            status['core_features'][feature] = 'available'
        else:
            print(f"❌ {feature}: Missing")
            status['core_features'][feature] = 'missing'
    
    # Check AI features
    print("\n🤖 AI Features Status:")
    print("-" * 30)
    
    ai_components = {
        'Price Predictor': 'ai/prediction/price_predictor.py',
        'Volatility Analyzer': 'ai/prediction/volatility_analyzer.py',
        'Strategy Optimizer': 'ai/strategy/strategy_optimizer.py',
        'AI Dashboard': 'ai_dashboard_tab.py',
        'Prediction Dashboard': 'ai_prediction_dashboard.py'
    }
    
    for feature, filepath in ai_components.items():
        if Path(filepath).exists():
            print(f"✅ {feature}: Available")
            status['ai_features'][feature] = 'available'
        else:
            print(f"❌ {feature}: Missing")
            status['ai_features'][feature] = 'missing'
    
    # Check data status
    print("\n📁 Data Status:")
    print("-" * 30)
    
    data_dir = Path("data")
    if data_dir.exists():
        print(f"✅ Data Directory: Available")
        
        # Check for live sessions
        live_sessions = list(data_dir.glob("live_session_*"))
        if live_sessions:
            latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
            print(f"✅ Live Sessions: {len(live_sessions)} found")
            print(f"   Latest: {latest_session.name}")
            status['data_status']['live_sessions'] = len(live_sessions)
            status['data_status']['latest_session'] = latest_session.name
        else:
            print(f"⚠️ Live Sessions: None found")
            status['data_status']['live_sessions'] = 0
            status['recommendations'].append("Run a trading session to generate data")
        
        # Check for monitor sessions
        monitor_sessions = list(data_dir.glob("simple_pair_monitor_*"))
        if monitor_sessions:
            print(f"✅ Monitor Sessions: {len(monitor_sessions)} found")
            status['data_status']['monitor_sessions'] = len(monitor_sessions)
        else:
            print(f"⚠️ Monitor Sessions: None found")
            status['data_status']['monitor_sessions'] = 0
    else:
        print(f"❌ Data Directory: Missing")
        status['data_status']['directory'] = 'missing'
        status['recommendations'].append("Create data directory and run initial session")
    
    # Check dashboard components
    print("\n📊 Dashboard Components:")
    print("-" * 30)
    
    dashboard_tabs = [
        "Overview",
        "Triangular Arbitrage", 
        "Simple Pair Trading",
        "🤖 AI Analysis",
        "📈 Price Prediction",
        "⚡ Volatility Trading",
        "🎯 Strategy Optimization"
    ]
    
    available_tabs = len([tab for tab in dashboard_tabs[:3]])  # Core tabs always available
    
    try:
        # Check if AI components can be imported
        from ai.prediction.price_predictor import PricePredictor
        from ai.prediction.volatility_analyzer import VolatilityAnalyzer
        from ai.strategy.strategy_optimizer import StrategyOptimizer
        available_tabs += 4  # AI tabs available
        print(f"✅ AI Dashboard Tabs: Available")
        status['dashboard_status']['ai_tabs'] = 'available'
    except ImportError as e:
        print(f"❌ AI Dashboard Tabs: Import Error - {e}")
        status['dashboard_status']['ai_tabs'] = 'error'
        status['recommendations'].append("Check AI module dependencies")
    
    print(f"✅ Total Dashboard Tabs: {available_tabs}/7 available")
    status['dashboard_status']['total_tabs'] = available_tabs
    
    return status

def show_feature_overview():
    """Show overview of all available features."""
    print("\n\n🎯 KrakenBot Feature Overview")
    print("=" * 60)
    
    features = {
        "📊 Core Trading Features": [
            "Triangular Arbitrage Detection",
            "Simple Pair Trading Strategy", 
            "Real-time Price Monitoring",
            "Portfolio Tracking",
            "Performance Analytics",
            "Risk Management"
        ],
        "🤖 AI-Powered Features": [
            "Price Prediction (Statistical + ML)",
            "Volatility Analysis & Trading",
            "Strategy Optimization",
            "Trading Signal Generation",
            "Market Regime Detection",
            "Risk-Adjusted Position Sizing"
        ],
        "📈 Dashboard Features": [
            "Interactive Visualizations",
            "Real-time Monitoring",
            "AI Analysis Interface",
            "Strategy Comparison",
            "Performance Metrics",
            "Parameter Optimization"
        ],
        "⚙️ System Features": [
            "Automated Data Collection",
            "Session Management",
            "Configuration Management",
            "Error Handling & Recovery",
            "Comprehensive Logging",
            "Modular Architecture"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}:")
        print("-" * (len(category) - 2))
        for feature in feature_list:
            print(f"  ✅ {feature}")

def show_quick_start_guide():
    """Show quick start guide for new users."""
    print("\n\n🚀 Quick Start Guide")
    print("=" * 60)
    
    steps = [
        {
            "step": "1. Run Initial Test",
            "command": "python run_simple_pair_monitor.py --test",
            "description": "Generate initial trading data (5 minutes)"
        },
        {
            "step": "2. Launch Dashboard", 
            "command": "streamlit run simple_dashboard.py --server.port 8503",
            "description": "Open the interactive dashboard"
        },
        {
            "step": "3. Explore AI Features",
            "command": "Navigate to AI tabs in dashboard",
            "description": "Try Price Prediction and Volatility Trading"
        },
        {
            "step": "4. Run Full Demo",
            "command": "python demo_ai_prediction_full.py",
            "description": "See all AI features with mock data"
        },
        {
            "step": "5. Optimize Strategy",
            "command": "Use Strategy Optimization tab",
            "description": "Get AI-powered parameter recommendations"
        }
    ]
    
    for step_info in steps:
        print(f"\n{step_info['step']}:")
        print(f"   Command: {step_info['command']}")
        print(f"   Purpose: {step_info['description']}")

def show_system_recommendations(status):
    """Show personalized recommendations based on system status."""
    print("\n\n💡 System Recommendations")
    print("=" * 60)
    
    recommendations = status.get('recommendations', [])
    
    # Add automatic recommendations based on status
    if status['data_status'].get('live_sessions', 0) == 0:
        recommendations.append("Run your first trading session to start collecting data")
    
    if status['data_status'].get('live_sessions', 0) < 3:
        recommendations.append("Run more trading sessions to improve AI model accuracy")
    
    if status['dashboard_status'].get('ai_tabs') == 'available':
        recommendations.append("Explore the new AI features in the dashboard")
    
    if len(recommendations) == 0:
        recommendations.append("System is fully operational - explore advanced features!")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def main():
    """Main system status function."""
    print("🔍 KrakenBot Complete System Analysis")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check system status
    status = check_system_status()
    
    # Show feature overview
    show_feature_overview()
    
    # Show quick start guide
    show_quick_start_guide()
    
    # Show recommendations
    show_system_recommendations(status)
    
    # Final summary
    print("\n\n🎉 System Analysis Complete!")
    print("=" * 70)
    
    core_available = sum(1 for v in status['core_features'].values() if v == 'available')
    ai_available = sum(1 for v in status['ai_features'].values() if v == 'available')
    total_features = len(status['core_features']) + len(status['ai_features'])
    available_features = core_available + ai_available
    
    print(f"📊 Feature Availability: {available_features}/{total_features} ({available_features/total_features*100:.1f}%)")
    print(f"🤖 AI Features: {ai_available}/{len(status['ai_features'])} available")
    print(f"📁 Data Sessions: {status['data_status'].get('live_sessions', 0)} live sessions")
    print(f"📈 Dashboard Tabs: {status['dashboard_status'].get('total_tabs', 0)}/7 available")
    
    if available_features == total_features:
        print(f"\n✅ ALL SYSTEMS OPERATIONAL!")
        print("🚀 KrakenBot is ready for advanced AI-powered trading!")
    elif available_features >= total_features * 0.8:
        print(f"\n⚠️ MOSTLY OPERATIONAL")
        print("Most features available, check recommendations above.")
    else:
        print(f"\n❌ SETUP REQUIRED")
        print("Please follow the recommendations to complete setup.")
    
    print(f"\n🌐 Dashboard URL: http://localhost:8503")
    print(f"📚 Documentation: AI_FEATURES_README.md")
    print(f"🔧 Configuration: config.py")

if __name__ == "__main__":
    main()