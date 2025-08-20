"""
KrakenBot Fix Common Issues Tool

This script attempts to fix common issues with the KrakenBot installation.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "-"))
    print("=" * 60)

def fix_directories():
    """Create missing directories."""
    print_header("Fixing Directories")
    
    directories = ["data", "logs", "data/reports", "data/historical"]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            print(f"Creating missing directory: {directory}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")
    
    print("\nAll required directories have been created.")

def fix_dependencies():
    """Fix Python dependencies."""
    print_header("Fixing Dependencies")
    
    print("Reinstalling all dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--force-reinstall"],
            check=True
        )
        print("\nAll dependencies have been reinstalled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError reinstalling dependencies: {e}")
        print("Trying to install core dependencies individually...")
        
        core_packages = [
            "ccxt",
            "python-dotenv",
            "pandas",
            "numpy",
            "matplotlib",
            "loguru",
            "psutil",
            "requests"
        ]
        
        for package in core_packages:
            try:
                print(f"Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True
                )
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")

def fix_env_file():
    """Fix .env file."""
    print_header("Fixing .env File")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists():
        if env_example_path.exists():
            print(".env file not found, but .env.example exists.")
            print("Creating .env file from .env.example...")
            
            shutil.copy(env_example_path, env_path)
            print(".env file created successfully.")
        else:
            print("Neither .env nor .env.example found.")
            print("Creating a basic .env file...")
            
            with open(env_path, "w") as f:
                f.write("""# KrakenBot Configuration
# Created by fix_common_issues.py

# API Keys (optional for monitoring)
KRAKEN_API_KEY=
KRAKEN_SECRET=

# Trading Parameters
START_AMOUNT=10000
PROFIT_THRESHOLD=5
MAKER_FEE=0.0025
TAKER_FEE=0.0040
SLIPPAGE_BUFFER=0.001

# Monitoring Settings
CHECK_INTERVAL=10
LOG_LEVEL=INFO

# Trading Pairs
PAIR_1=BTC/CAD
PAIR_2=ETH/CAD
PAIR_3=ETH/BTC

# Trading Mode
TRADING_MODE=monitor

# Notifications
ENABLE_NOTIFICATIONS=false
NOTIFICATION_METHOD=Console Only
""")
            print("Basic .env file created successfully.")
    else:
        print(".env file already exists.")
        
        # Check if it has the minimum required variables
        with open(env_path, "r") as f:
            content = f.read()
        
        required_vars = [
            "START_AMOUNT",
            "PROFIT_THRESHOLD",
            "CHECK_INTERVAL",
            "LOG_LEVEL",
            "PAIR_1",
            "PAIR_2",
            "PAIR_3"
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"The following required variables are missing: {', '.join(missing_vars)}")
            print("Adding missing variables to .env file...")
            
            with open(env_path, "a") as f:
                f.write("\n# Added by fix_common_issues.py\n")
                
                if "START_AMOUNT" in missing_vars:
                    f.write("START_AMOUNT=10000\n")
                
                if "PROFIT_THRESHOLD" in missing_vars:
                    f.write("PROFIT_THRESHOLD=5\n")
                
                if "CHECK_INTERVAL" in missing_vars:
                    f.write("CHECK_INTERVAL=10\n")
                
                if "LOG_LEVEL" in missing_vars:
                    f.write("LOG_LEVEL=INFO\n")
                
                if "PAIR_1" in missing_vars:
                    f.write("PAIR_1=BTC/CAD\n")
                
                if "PAIR_2" in missing_vars:
                    f.write("PAIR_2=ETH/CAD\n")
                
                if "PAIR_3" in missing_vars:
                    f.write("PAIR_3=ETH/BTC\n")
            
            print("Missing variables added to .env file.")
        else:
            print("All required variables are present in the .env file.")

def fix_permissions():
    """Fix file permissions."""
    print_header("Fixing Permissions")
    
    # This is more relevant on Linux/Mac, but we'll include it for completeness
    if os.name == "posix":  # Linux/Mac
        print("Setting correct permissions for scripts...")
        
        scripts = [
            "start_krakenbot.sh",
            "run_diagnostics.sh"
        ]
        
        for script in scripts:
            if os.path.exists(script):
                try:
                    os.chmod(script, 0o755)  # rwxr-xr-x
                    print(f"Set executable permissions for {script}")
                except Exception as e:
                    print(f"Error setting permissions for {script}: {e}")
        
        print("\nPermissions have been updated.")
    else:
        print("Permission fixing is only needed on Linux/Mac systems.")
        print("No changes were made.")

def create_test_file():
    """Create a simple test file to verify write permissions."""
    print_header("Testing Write Permissions")
    
    test_dirs = [".", "data", "logs"]
    
    for directory in test_dirs:
        path = Path(directory)
        if path.exists():
            test_file = path / f"write_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(test_file, "w") as f:
                    f.write("This is a test file to verify write permissions.")
                
                print(f"Successfully created test file in {directory}")
                
                # Clean up
                os.remove(test_file)
                print(f"Test file removed from {directory}")
            except Exception as e:
                print(f"Error writing to {directory}: {e}")
        else:
            print(f"Directory {directory} does not exist, skipping write test.")

def fix_config_module():
    """Fix config.py if it's missing or corrupted."""
    print_header("Checking Config Module")
    
    config_path = Path("config.py")
    
    if not config_path.exists():
        print("config.py is missing. Creating a basic version...")
        
        with open(config_path, "w") as f:
            f.write("""\"\"\"
Configuration module for KrakenBot.
Loads settings from environment variables or .env file.
\"\"\"
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Trading parameters
START_AMOUNT = float(os.getenv("START_AMOUNT", "10000"))
PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", "5"))
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.0025"))  # 0.25%
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.0040"))  # 0.40%
SLIPPAGE_BUFFER = float(os.getenv("SLIPPAGE_BUFFER", "0.001"))  # 0.1%

# Monitoring settings
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Trading mode
TRADING_MODE = os.getenv("TRADING_MODE", "monitor")

# Trading pairs
TRADING_PAIRS = {
    "PAIR_1": os.getenv("PAIR_1", "BTC/CAD"),
    "PAIR_2": os.getenv("PAIR_2", "ETH/CAD"),
    "PAIR_3": os.getenv("PAIR_3", "ETH/BTC")
}

# API keys (optional for monitoring)
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_SECRET", "")

# Notification settings
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
NOTIFICATION_METHOD = os.getenv("NOTIFICATION_METHOD", "Console Only")
""")
        print("Basic config.py file created successfully.")
    else:
        print("config.py exists. Checking if it's valid...")
        
        try:
            # Try to import it
            sys.path.insert(0, os.getcwd())
            import config
            
            # Check for essential attributes
            required_attrs = [
                "START_AMOUNT",
                "PROFIT_THRESHOLD",
                "CHECK_INTERVAL",
                "LOG_LEVEL",
                "TRADING_PAIRS",
                "DATA_DIR",
                "LOG_DIR"
            ]
            
            missing_attrs = []
            for attr in required_attrs:
                if not hasattr(config, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                print(f"The following required attributes are missing: {', '.join(missing_attrs)}")
                print("The config.py file may be corrupted. Consider running the setup wizard.")
            else:
                print("config.py appears to be valid.")
        
        except Exception as e:
            print(f"Error importing config.py: {e}")
            print("The file may be corrupted. Consider backing it up and running the fix again.")

def main():
    """Run all fixes."""
    print("\nKrakenBot Fix Common Issues Tool")
    print("=" * 60)
    print("This tool will attempt to fix common issues with your KrakenBot installation.")
    print("It will create missing directories, fix dependencies, and check configuration files.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Run all fixes
    fix_directories()
    fix_dependencies()
    fix_env_file()
    fix_permissions()
    create_test_file()
    fix_config_module()
    
    print_header("All Fixes Completed")
    print("The most common issues have been addressed.")
    print("You should now be able to run KrakenBot without errors.")
    print("\nIf you still encounter issues, please run the diagnostic tool:")
    print("  Windows: run_diagnostics.bat")
    print("  Linux/Mac: ./run_diagnostics.sh")
    print("\nThen try running KrakenBot again:")
    print("  Windows: start_krakenbot.bat")
    print("  Linux/Mac: ./start_krakenbot.sh")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)