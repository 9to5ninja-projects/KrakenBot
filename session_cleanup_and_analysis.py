"""
Session Cleanup and Comprehensive Analysis
Closes all running sessions, analyzes data, and prepares for next optimization cycle
"""

import os
import sys
import json
import psutil
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

def stop_all_sessions():
    """Stop all running trading sessions and monitors."""
    print("🛑 Stopping All Trading Sessions")
    print("=" * 50)
    
    stopped_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline:
                        # Check for trading session processes
                        session_keywords = [
                            'run_simple_pair_monitor.py',
                            'start_live_trading_session.py',
                            'session_monitor.py'
                        ]
                        
                        for keyword in session_keywords:
                            if any(keyword in str(cmd) for cmd in cmdline):
                                print(f"🔄 Stopping process {proc.info['pid']}: {keyword}")
                                proc.terminate()
                                stopped_processes.append({
                                    'pid': proc.info['pid'],
                                    'process': keyword
                                })
                                break
                                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception as e:
        print(f"❌ Error stopping processes: {e}")
    
    if stopped_processes:
        print(f"✅ Stopped {len(stopped_processes)} trading processes")
        for proc in stopped_processes:
            print(f"   • PID {proc['pid']}: {proc['process']}")
    else:
        print("ℹ️ No trading processes found to stop")
    
    return stopped_processes

def analyze_session_data():
    """Analyze all session data and generate comprehensive report."""
    print("\n📊 Analyzing Session Data")
    print("=" * 50)
    
    data_dir = Path("data")
    analysis_results = {
        'sessions_analyzed': [],
        'baseline_performance': {},
        'optimized_performance': {},
        'comparison': {},
        'recommendations': []
    }
    
    # Find all relevant sessions
    session_patterns = [
        "live_session_*",
        "optimized_session_*",
        "simple_pair_monitor_*"
    ]
    
    all_sessions = []
    for pattern in session_patterns:
        all_sessions.extend(data_dir.glob(pattern))
    
    print(f"📁 Found {len(all_sessions)} sessions to analyze")
    
    baseline_sessions = []
    optimized_sessions = []
    
    for session_dir in all_sessions:
        session_name = session_dir.name
        print(f"\n🔍 Analyzing: {session_name}")
        
        # Check for required files
        portfolio_file = session_dir / "portfolio_history.json"
        trades_file = session_dir / "trades.json"
        price_file = session_dir / "price_history.json"
        
        if not portfolio_file.exists():
            print(f"   ⚠️ No portfolio data found")
            continue
            
        try:
            # Load portfolio data
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
            
            if not portfolio_data:
                print(f"   ⚠️ Empty portfolio data")
                continue
            
            # Basic session analysis
            session_analysis = {
                'name': session_name,
                'data_points': len(portfolio_data),
                'duration_hours': 0,
                'final_value': 100.0,
                'total_return': 0.0,
                'trades_executed': 0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
            
            # Calculate duration
            if len(portfolio_data) >= 2:
                start_time = datetime.fromisoformat(portfolio_data[0]['timestamp'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(portfolio_data[-1]['timestamp'].replace('Z', '+00:00'))
                duration = end_time - start_time
                session_analysis['duration_hours'] = duration.total_seconds() / 3600
            
            # Get final performance
            final_data = portfolio_data[-1]
            session_analysis['final_value'] = final_data['portfolio_value']
            session_analysis['total_return'] = final_data['total_return']
            
            # Calculate max drawdown and volatility
            values = [point['portfolio_value'] for point in portfolio_data]
            returns = [point['total_return'] for point in portfolio_data]
            
            if len(values) > 1:
                # Max drawdown
                peak = values[0]
                max_dd = 0
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak * 100
                    max_dd = max(max_dd, drawdown)
                session_analysis['max_drawdown'] = max_dd
                
                # Volatility (std dev of returns)
                if len(returns) > 1:
                    session_analysis['volatility'] = pd.Series(returns).std()
            
            # Load trades data
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
                session_analysis['trades_executed'] = len(trades_data)
            
            print(f"   ✅ Duration: {session_analysis['duration_hours']:.1f}h")
            print(f"   ✅ Data points: {session_analysis['data_points']}")
            print(f"   ✅ Final return: {session_analysis['total_return']:.4f}%")
            print(f"   ✅ Trades: {session_analysis['trades_executed']}")
            print(f"   ✅ Max drawdown: {session_analysis['max_drawdown']:.2f}%")
            
            # Categorize sessions
            if 'optimized' in session_name:
                optimized_sessions.append(session_analysis)
            else:
                baseline_sessions.append(session_analysis)
            
            analysis_results['sessions_analyzed'].append(session_analysis)
            
        except Exception as e:
            print(f"   ❌ Error analyzing {session_name}: {e}")
    
    # Generate comparison
    if baseline_sessions and optimized_sessions:
        print(f"\n📈 Performance Comparison")
        print("=" * 30)
        
        # Get best baseline and optimized sessions
        best_baseline = max(baseline_sessions, key=lambda x: x['total_return'])
        best_optimized = max(optimized_sessions, key=lambda x: x['total_return'])
        
        print(f"🔵 Best Baseline: {best_baseline['name']}")
        print(f"   Return: {best_baseline['total_return']:.4f}%")
        print(f"   Trades: {best_baseline['trades_executed']}")
        print(f"   Duration: {best_baseline['duration_hours']:.1f}h")
        
        print(f"🟢 Best Optimized: {best_optimized['name']}")
        print(f"   Return: {best_optimized['total_return']:.4f}%")
        print(f"   Trades: {best_optimized['trades_executed']}")
        print(f"   Duration: {best_optimized['duration_hours']:.1f}h")
        
        # Calculate improvement
        return_improvement = best_optimized['total_return'] - best_baseline['total_return']
        trade_improvement = best_optimized['trades_executed'] - best_baseline['trades_executed']
        
        print(f"\n📊 Improvement Analysis:")
        print(f"   Return change: {return_improvement:+.4f}%")
        print(f"   Trade change: {trade_improvement:+d}")
        
        analysis_results['baseline_performance'] = best_baseline
        analysis_results['optimized_performance'] = best_optimized
        analysis_results['comparison'] = {
            'return_improvement': return_improvement,
            'trade_improvement': trade_improvement
        }
    
    return analysis_results

def generate_next_strategy_recommendations(analysis_results):
    """Generate recommendations for next strategy iteration."""
    print(f"\n🎯 Next Strategy Recommendations")
    print("=" * 50)
    
    recommendations = []
    
    # Analyze trade frequency
    total_trades = sum(session['trades_executed'] for session in analysis_results['sessions_analyzed'])
    total_hours = sum(session['duration_hours'] for session in analysis_results['sessions_analyzed'])
    
    if total_trades == 0:
        print("❌ CRITICAL: No trades executed in any session!")
        recommendations.extend([
            "IMMEDIATE: Make thresholds much more aggressive",
            "Reduce BUY_THRESHOLD to -0.003 (-0.3%) or lower",
            "Reduce SELL_THRESHOLD to 0.005 (+0.5%) or lower", 
            "Reduce LOOKBACK_PERIODS to 5 or lower",
            "Add trailing stop-loss mechanism",
            "Implement take-profit levels"
        ])
    else:
        trades_per_hour = total_trades / total_hours if total_hours > 0 else 0
        print(f"📊 Average trade frequency: {trades_per_hour:.2f} trades/hour")
        
        if trades_per_hour < 0.5:  # Less than 1 trade per 2 hours
            recommendations.extend([
                "Increase trade frequency - thresholds still too conservative",
                "Reduce BUY_THRESHOLD by additional 20-30%",
                "Reduce SELL_THRESHOLD by additional 20-30%",
                "Consider market-making approach with tighter spreads"
            ])
    
    # Risk management recommendations
    recommendations.extend([
        "Implement trailing stop-loss at 0.2-0.3% below entry",
        "Add take-profit levels at 0.4-0.6% above entry",
        "Implement position sizing based on volatility",
        "Add maximum position hold time (30-60 minutes)",
        "Implement dynamic threshold adjustment based on market volatility"
    ])
    
    # Technical improvements
    recommendations.extend([
        "Add real-time volatility-based threshold adjustment",
        "Implement momentum indicators for entry timing",
        "Add volume analysis for trade validation",
        "Consider multiple timeframe analysis",
        "Add correlation analysis between pairs"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    analysis_results['recommendations'] = recommendations
    return recommendations

def create_aggressive_config():
    """Create a more aggressive configuration for next iteration."""
    print(f"\n⚡ Creating Aggressive Configuration")
    print("=" * 50)
    
    # Backup current config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"config_backup_post_analysis_{timestamp}.py"
    
    if Path("config.py").exists():
        shutil.copy("config.py", backup_file)
        print(f"✅ Config backed up to: {backup_file}")
    
    # Read current config
    with open("config.py", 'r') as f:
        config_content = f.read()
    
    # Define aggressive parameters
    aggressive_params = {
        'BUY_THRESHOLD': -0.003,    # Very aggressive: buy on 0.3% drop
        'SELL_THRESHOLD': 0.005,    # Quick profit: sell on 0.5% gain
        'LOOKBACK_PERIODS': 5,      # Very responsive
        'MIN_TRADE_AMOUNT': 25.0,   # Keep minimum trade size
        'MAX_POSITION_SIZE': 0.2,   # Reduce position size for safety
        'STOP_LOSS_PCT': 0.002,     # 0.2% stop loss
        'TAKE_PROFIT_PCT': 0.006,   # 0.6% take profit
        'MAX_HOLD_TIME': 1800,      # 30 minutes max hold
        'TRAILING_STOP': True,      # Enable trailing stop
        'DYNAMIC_THRESHOLDS': True  # Enable dynamic adjustment
    }
    
    print("🎯 Aggressive Parameters:")
    for param, value in aggressive_params.items():
        print(f"   {param}: {value}")
    
    # Update config with aggressive parameters
    import re
    
    for param_name, new_value in aggressive_params.items():
        pattern = f"{param_name}\\s*=\\s*[\\w\\.-]+"
        replacement = f"{param_name} = {new_value}"
        
        if re.search(pattern, config_content):
            config_content = re.sub(pattern, replacement, config_content)
            print(f"✅ Updated {param_name}")
        else:
            # Add new parameter
            config_content += f"\n# Aggressive Trading Parameter\n{param_name} = {new_value}\n"
            print(f"✅ Added {param_name}")
    
    # Write updated config
    with open("config.py", 'w') as f:
        f.write(config_content)
    
    print(f"✅ Aggressive configuration applied!")
    return aggressive_params

def save_comprehensive_report(analysis_results, recommendations, aggressive_params):
    """Save comprehensive analysis report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"data/comprehensive_analysis_{timestamp}.json"
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'session_analysis': analysis_results,
        'recommendations': recommendations,
        'aggressive_parameters': aggressive_params,
        'next_steps': [
            "Test aggressive configuration with 30-minute session",
            "Monitor trade execution frequency",
            "Validate stop-loss and take-profit mechanisms",
            "Analyze risk-adjusted returns",
            "Iterate based on results"
        ]
    }
    
    os.makedirs("data", exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📋 Comprehensive report saved: {report_file}")
    return report_file

def main():
    """Main cleanup and analysis function."""
    print("🧹 KrakenBot Session Cleanup and Analysis")
    print("Complete session termination, data analysis, and strategy optimization")
    print("=" * 80)
    
    try:
        # Step 1: Stop all sessions
        stopped_processes = stop_all_sessions()
        
        # Step 2: Analyze session data
        analysis_results = analyze_session_data()
        
        # Step 3: Generate recommendations
        recommendations = generate_next_strategy_recommendations(analysis_results)
        
        # Step 4: Create aggressive configuration
        aggressive_params = create_aggressive_config()
        
        # Step 5: Save comprehensive report
        report_file = save_comprehensive_report(analysis_results, recommendations, aggressive_params)
        
        print(f"\n🎉 CLEANUP AND ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"✅ Sessions stopped: {len(stopped_processes)}")
        print(f"✅ Sessions analyzed: {len(analysis_results['sessions_analyzed'])}")
        print(f"✅ Recommendations generated: {len(recommendations)}")
        print(f"✅ Aggressive config applied")
        print(f"✅ Report saved: {report_file}")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Review the aggressive configuration parameters")
        print("2. Run a 30-minute test with new aggressive settings:")
        print("   python run_simple_pair_monitor.py --duration 0.5 --session-name aggressive_test")
        print("3. Monitor for actual trade execution")
        print("4. Analyze results and iterate")
        
        print(f"\n⚠️ FOCUS: The new configuration is designed to execute trades!")
        print("Monitor closely for risk management effectiveness.")
        
    except Exception as e:
        print(f"\n❌ Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()