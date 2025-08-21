"""
Apply AI-Generated Strategy Optimization
Automatically applies the AI-recommended parameters and runs an optimized trading session
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def load_optimized_parameters():
    """Load the latest AI-optimized parameters."""
    data_dir = Path("data")
    
    # Find the latest session with optimized parameters
    optimization_files = []
    for session_dir in data_dir.glob("live_session_*"):
        opt_file = session_dir / "test_optimized_parameters.json"
        if opt_file.exists():
            optimization_files.append(opt_file)
    
    if not optimization_files:
        print("❌ No optimized parameters found. Run strategy optimization first.")
        return None
    
    # Use the most recent optimization
    latest_opt_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Loading optimized parameters from: {latest_opt_file.parent.name}")
    
    with open(latest_opt_file, 'r') as f:
        optimization_data = json.load(f)
    
    return optimization_data

def backup_current_config():
    """Backup current configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"config_backup_{timestamp}.py"
    
    if Path("config.py").exists():
        shutil.copy("config.py", backup_file)
        print(f"✅ Current config backed up to: {backup_file}")
        return backup_file
    return None

def apply_optimized_parameters(optimization_data):
    """Apply the optimized parameters to the configuration."""
    optimized_params = optimization_data['optimized_parameters']
    applied_changes = optimization_data['applied_changes']
    
    print("\n🎯 Applying AI-Optimized Parameters:")
    print("=" * 50)
    
    # Read current config
    with open("config.py", 'r') as f:
        config_content = f.read()
    
    # Apply parameter changes
    parameter_mapping = {
        'buy_threshold': 'BUY_THRESHOLD',
        'sell_threshold': 'SELL_THRESHOLD',
        'lookback_periods': 'LOOKBACK_PERIODS',
        'min_trade_amount': 'MIN_TRADE_AMOUNT',
        'max_position_size': 'MAX_POSITION_SIZE'
    }
    
    changes_applied = []
    
    for param_name, config_name in parameter_mapping.items():
        if param_name in optimized_params:
            new_value = optimized_params[param_name]
            
            # Find the parameter in config and update it
            import re
            pattern = f"{config_name}\\s*=\\s*[\\d\\.-]+"
            replacement = f"{config_name} = {new_value}"
            
            if re.search(pattern, config_content):
                old_content = config_content
                config_content = re.sub(pattern, replacement, config_content)
                
                if old_content != config_content:
                    changes_applied.append({
                        'parameter': config_name,
                        'new_value': new_value
                    })
                    print(f"✅ Updated {config_name} = {new_value}")
            else:
                # Add the parameter if it doesn't exist
                config_content += f"\n# AI-Optimized Parameter\n{config_name} = {new_value}\n"
                changes_applied.append({
                    'parameter': config_name,
                    'new_value': new_value
                })
                print(f"✅ Added {config_name} = {new_value}")
    
    # Write updated config
    with open("config.py", 'w') as f:
        f.write(config_content)
    
    print(f"\n📊 Applied {len(changes_applied)} parameter changes")
    
    # Show rationale for changes
    print("\n💡 AI Rationale for Changes:")
    print("-" * 30)
    for change in applied_changes:
        print(f"• {change['parameter']}: {change['rationale']}")
        print(f"  Change: {change['old_value']} → {change['new_value']} ({change['change_percent']:+.0f}%)")
    
    return changes_applied

def run_optimized_session():
    """Run a new trading session with optimized parameters."""
    print("\n🚀 Starting Optimized Trading Session")
    print("=" * 50)
    
    # Create session name with optimization marker
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_name = f"optimized_session_{timestamp}"
    
    print(f"📊 Session Name: {session_name}")
    print("⏱️ Duration: 1 hour (for comprehensive analysis)")
    print("🎯 Strategy: AI-Optimized Simple Pair Trading")
    
    # Run the optimized session
    import subprocess
    
    try:
        print("\n🔄 Launching optimized trading session...")
        
        # Run for 1 hour to get good data
        cmd = [
            "python", "run_simple_pair_monitor.py",
            "--duration", "60",  # 60 minutes
            "--session-name", session_name
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("\n📈 Session is running... This will take 60 minutes.")
        print("💡 You can monitor progress in the dashboard while it runs.")
        print("🌐 Dashboard: http://localhost:8503")
        
        # Run in background so user can use dashboard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Session started with PID: {process.pid}")
        print("📊 Check the dashboard for real-time progress!")
        
        return session_name, process.pid
        
    except Exception as e:
        print(f"❌ Error starting session: {e}")
        return None, None

def create_optimization_report(optimization_data, changes_applied, session_name):
    """Create a report of the optimization process."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "optimization_timestamp": timestamp,
        "session_name": session_name,
        "ai_recommendations": optimization_data,
        "applied_changes": changes_applied,
        "expected_improvements": optimization_data['optimization_summary']['expected_improvements'],
        "next_steps": [
            "Monitor the optimized session performance",
            "Compare results with previous sessions",
            "Run strategy optimization again after session completion",
            "Consider further parameter refinements based on results"
        ]
    }
    
    # Save report
    report_file = f"data/ai_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data", exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📋 Optimization report saved: {report_file}")
    return report_file

def main():
    """Main optimization application function."""
    print("🤖 KrakenBot AI Strategy Optimization Application")
    print("Applying AI-recommended parameters and starting optimized session")
    print("=" * 70)
    
    try:
        # Step 1: Load optimized parameters
        optimization_data = load_optimized_parameters()
        if not optimization_data:
            return
        
        # Step 2: Backup current config
        backup_file = backup_current_config()
        
        # Step 3: Apply optimized parameters
        changes_applied = apply_optimized_parameters(optimization_data)
        
        # Step 4: Run optimized session
        session_name, process_id = run_optimized_session()
        
        if session_name:
            # Step 5: Create optimization report
            report_file = create_optimization_report(optimization_data, changes_applied, session_name)
            
            print("\n🎉 AI OPTIMIZATION APPLIED SUCCESSFULLY!")
            print("=" * 50)
            print(f"✅ Parameters optimized and applied")
            print(f"✅ Optimized session started: {session_name}")
            print(f"✅ Process ID: {process_id}")
            print(f"✅ Report saved: {report_file}")
            
            if backup_file:
                print(f"✅ Config backup: {backup_file}")
            
            print(f"\n📊 MONITORING RECOMMENDATIONS:")
            print("1. Open dashboard: http://localhost:8503")
            print("2. Monitor session progress in real-time")
            print("3. Compare performance with previous sessions")
            print("4. Run strategy optimization again after completion")
            
            print(f"\n⏱️ SESSION DETAILS:")
            print(f"• Duration: 60 minutes")
            print(f"• Strategy: AI-Optimized Simple Pair Trading")
            print(f"• Expected improvements:")
            for improvement in optimization_data['optimization_summary']['expected_improvements']:
                print(f"  - {improvement}")
            
            print(f"\n🔄 The session will run in the background.")
            print("You can close this terminal and monitor via dashboard.")
            
        else:
            print("\n❌ Failed to start optimized session")
            
            # Restore backup if session failed
            if backup_file and Path(backup_file).exists():
                shutil.copy(backup_file, "config.py")
                print(f"🔄 Restored original configuration from {backup_file}")
    
    except Exception as e:
        print(f"\n❌ Optimization error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()