"""
Cleanup Script for KrakenBot Test Files
Removes unused test files and keeps only the essential working system
"""

import os
import shutil
from pathlib import Path

def cleanup_test_files():
    """Remove test files and keep only essential working system."""
    
    base_dir = Path("e:/KrakenBot")
    
    # Files to KEEP (essential working system)
    essential_files = {
        # Core system
        'main.py', 'config.py', 'exchange.py', 'arbitrage.py', 'monitor.py',
        'dashboard.py', 'logger.py', 'trader.py', 'backtest.py', 'setup_wizard.py',
        'reports.py', 'notifications.py', 'health_monitor.py',
        
        # Enhanced AI system (working)
        'advanced_technical_indicators.py', 'ai_strategy_optimizer.py',
        'multi_coin_analyzer.py', 'enhanced_nlp_analyzer.py',
        'enhanced_simulation_system.py', 'ai_enhanced_dashboard.py',
        'run_enhanced_simulation.py', 'market_opening_ai.py',
        
        # Current monitoring
        'current_session_monitor.py', 'cleanup_test_files.py',
        
        # Documentation
        'README.md', 'ENHANCED_SYSTEM_README.md', 'SESSION_ANALYSIS_INSIGHTS.md',
        
        # Configuration
        '.env', '.env.example', 'requirements.txt',
        
        # Batch files
        'start_krakenbot.bat', 'start_krakenbot.sh',
        'run_diagnostics.bat', 'run_diagnostics.sh',
        'fix_issues.bat', 'fix_issues.sh',
        
        # Essential utilities
        'diagnose.py', 'fix_common_issues.py'
    }
    
    # Directories to KEEP
    essential_dirs = {
        'docs', 'tests', 'scripts', 'logs', 'venv', '__pycache__', 'ai', '.zencoder'
    }
    
    # Data sessions to KEEP (recent working sessions)
    keep_data_sessions = {
        'working_1000_session_20250821_021951',
        'working_1000_session_20250821_022551', 
        'enhanced_sim_20250821_141431',  # Current running session
        'market_openings'
    }
    
    print("🧹 KrakenBot Test File Cleanup")
    print("="*50)
    print("Removing unused test files and keeping essential system...")
    print()
    
    removed_count = 0
    
    # Clean up root directory files
    print("📁 Cleaning root directory...")
    for file_path in base_dir.iterdir():
        if file_path.is_file():
            if file_path.name not in essential_files:
                print(f"   🗑️  Removing: {file_path.name}")
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"   ❌ Error removing {file_path.name}: {e}")
        elif file_path.is_dir() and file_path.name == 'data':
            # Special handling for data directory
            print("📊 Cleaning data directory...")
            data_dir = file_path
            for data_item in data_dir.iterdir():
                if data_item.is_dir():
                    if data_item.name not in keep_data_sessions:
                        print(f"   🗑️  Removing data session: {data_item.name}")
                        try:
                            shutil.rmtree(data_item)
                            removed_count += 1
                        except Exception as e:
                            print(f"   ❌ Error removing {data_item.name}: {e}")
                elif data_item.is_file():
                    # Keep essential data files
                    if data_item.name not in ['README.md', 'optimal_triangles.json']:
                        if not data_item.name.startswith('health') and not data_item.name.startswith('stats'):
                            print(f"   🗑️  Removing data file: {data_item.name}")
                            try:
                                data_item.unlink()
                                removed_count += 1
                            except Exception as e:
                                print(f"   ❌ Error removing {data_item.name}: {e}")
    
    print()
    print("✅ Cleanup Complete!")
    print(f"📊 Removed {removed_count} test files/directories")
    print()
    print("🎯 REMAINING ESSENTIAL SYSTEM:")
    print("   ✅ Core KrakenBot system")
    print("   ✅ Enhanced AI trading system") 
    print("   ✅ Current session monitor")
    print("   ✅ Recent working sessions data")
    print("   ✅ Documentation and configuration")
    print()
    print("🚀 Your system is now clean and optimized!")

def list_current_files():
    """List current files to see what we have."""
    base_dir = Path("e:/KrakenBot")
    
    print("📋 CURRENT FILES IN KRAKENBOT:")
    print("="*50)
    
    # List Python files
    python_files = list(base_dir.glob("*.py"))
    print(f"🐍 Python Files ({len(python_files)}):")
    for file in sorted(python_files):
        size_kb = file.stat().st_size / 1024
        print(f"   {file.name:<40} ({size_kb:.1f} KB)")
    
    print()
    
    # List data sessions
    data_dir = base_dir / "data"
    if data_dir.exists():
        sessions = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"📊 Data Sessions ({len(sessions)}):")
        for session in sorted(sessions):
            print(f"   {session.name}")
    
    print()

if __name__ == "__main__":
    print("🤖 KrakenBot File Management")
    print("="*40)
    print("Running automatic cleanup...")
    cleanup_test_files()