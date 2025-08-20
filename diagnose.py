"""
KrakenBot Diagnostic Tool

This script performs comprehensive diagnostics on the KrakenBot system,
capturing all errors to a log file even if the terminal crashes.
"""
import os
import sys
import time
import platform
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

# Create a diagnostic log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"diagnostic_{timestamp}.log"

def log(message):
    """Log a message to both console and log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def run_test(name, func):
    """Run a test function and log the result."""
    log(f"\n{'=' * 50}")
    log(f"TEST: {name}")
    log(f"{'-' * 50}")
    try:
        result = func()
        log(f"RESULT: PASSED")
        return result
    except Exception as e:
        log(f"RESULT: FAILED - {str(e)}")
        log(f"TRACEBACK:")
        log(traceback.format_exc())
        return None

def test_system_info():
    """Test system information."""
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python": sys.version,
        "Working Directory": os.getcwd(),
        "User": os.getlogin(),
        "Hostname": platform.node(),
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Python Path": sys.executable,
        "System Path": os.environ.get("PATH", "")
    }
    
    for key, value in info.items():
        log(f"{key}: {value}")
    
    return info

def test_directories():
    """Test directory structure and permissions."""
    directories = {
        "Current": ".",
        "Data": "data",
        "Logs": "logs",
        "Scripts": "scripts",
        "Docs": "docs"
    }
    
    results = {}
    for name, path in directories.items():
        full_path = os.path.abspath(path)
        exists = os.path.exists(full_path)
        is_dir = os.path.isdir(full_path) if exists else False
        writable = os.access(full_path, os.W_OK) if exists else False
        
        status = f"{'Exists' if exists else 'Missing'}"
        if exists:
            status += f", {'Directory' if is_dir else 'Not a directory'}"
            status += f", {'Writable' if writable else 'Not writable'}"
            
            # List contents if it's a directory
            if is_dir:
                try:
                    contents = os.listdir(full_path)
                    status += f", Contains {len(contents)} items"
                    if len(contents) <= 10:  # Only show if not too many
                        status += f": {', '.join(contents)}"
                except Exception as e:
                    status += f", Error listing contents: {str(e)}"
        
        log(f"{name} Directory ({full_path}): {status}")
        results[name] = {
            "path": full_path,
            "exists": exists,
            "is_dir": is_dir,
            "writable": writable
        }
    
    return results

def test_core_files():
    """Test core files existence and size."""
    core_files = [
        "main.py",
        "config.py",
        "monitor.py",
        "exchange.py",
        "arbitrage.py",
        "health_monitor.py",
        "notifications.py",
        "reports.py",
        "backtest.py",
        "setup_wizard.py",
        "requirements.txt",
        ".env"
    ]
    
    results = {}
    for file_name in core_files:
        path = os.path.abspath(file_name)
        exists = os.path.exists(path)
        is_file = os.path.isfile(path) if exists else False
        size = os.path.getsize(path) if is_file else 0
        readable = os.access(path, os.R_OK) if exists else False
        
        status = f"{'Exists' if exists else 'Missing'}"
        if exists:
            status += f", {'File' if is_file else 'Not a file'}"
            status += f", {size} bytes"
            status += f", {'Readable' if readable else 'Not readable'}"
        
        log(f"{file_name}: {status}")
        results[file_name] = {
            "path": path,
            "exists": exists,
            "is_file": is_file,
            "size": size,
            "readable": readable
        }
    
    return results

def test_env_file():
    """Test .env file content (without showing sensitive data)."""
    env_path = os.path.abspath(".env")
    if not os.path.exists(env_path):
        log(".env file does not exist")
        return None
    
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        env_vars = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if "=" in line:
                key, value = line.split("=", 1)
                # Mask sensitive values
                if any(sensitive in key.lower() for sensitive in ["key", "secret", "password", "token"]):
                    value = "*****MASKED*****"
                env_vars[key] = value
        
        log(f"Found {len(env_vars)} environment variables in .env file:")
        for key, value in env_vars.items():
            log(f"  {key}={value}")
        
        return env_vars
    except Exception as e:
        log(f"Error reading .env file: {str(e)}")
        return None

def test_python_dependencies():
    """Test Python dependencies."""
    required_packages = [
        "ccxt",
        "python-dotenv",
        "pandas",
        "numpy",
        "matplotlib",
        "streamlit",
        "loguru",
        "psutil",
        "requests",
        "pytest"
    ]
    
    results = {}
    for package in required_packages:
        try:
            # Try to import the package
            if package == "python-dotenv":
                module_name = "dotenv"
            else:
                module_name = package
            
            module = __import__(module_name)
            
            # Get version if available
            version = getattr(module, "__version__", "Unknown")
            
            log(f"{package}: Installed (version: {version})")
            results[package] = {
                "installed": True,
                "version": version
            }
        except ImportError:
            log(f"{package}: Not installed")
            results[package] = {
                "installed": False,
                "version": None
            }
        except Exception as e:
            log(f"{package}: Error checking - {str(e)}")
            results[package] = {
                "installed": False,
                "version": None,
                "error": str(e)
            }
    
    return results

def test_api_connection():
    """Test connection to Kraken API."""
    try:
        import requests
        log("Testing connection to Kraken API...")
        
        response = requests.get("https://api.kraken.com/0/public/Time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            log(f"Kraken API is accessible (status code: {response.status_code})")
            log(f"Server time: {data.get('result', {}).get('rfc1123', 'Unknown')}")
            return {
                "accessible": True,
                "status_code": response.status_code,
                "data": data
            }
        else:
            log(f"Kraken API returned error status code: {response.status_code}")
            log(f"Response: {response.text}")
            return {
                "accessible": False,
                "status_code": response.status_code,
                "data": response.text
            }
    except Exception as e:
        log(f"Error connecting to Kraken API: {str(e)}")
        return {
            "accessible": False,
            "error": str(e)
        }

def test_exchange_module():
    """Test the exchange module."""
    try:
        # First try to import the module
        log("Importing exchange module...")
        sys.path.insert(0, os.getcwd())
        from exchange import ExchangeManager
        
        # Try to initialize without API keys
        log("Initializing ExchangeManager without API keys...")
        exchange = ExchangeManager(use_api_keys=False)
        
        # Try to fetch a ticker
        log("Fetching BTC/USD ticker...")
        ticker = exchange.fetch_ticker("BTC/USD")
        log(f"Ticker: {ticker}")
        
        return {
            "module_imported": True,
            "initialized": True,
            "ticker_fetched": True,
            "ticker_data": ticker
        }
    except Exception as e:
        log(f"Error testing exchange module: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_config_module():
    """Test the config module."""
    try:
        # Import the module
        log("Importing config module...")
        sys.path.insert(0, os.getcwd())
        import config
        
        # Get configuration values (excluding sensitive data)
        config_values = {}
        for attr in dir(config):
            if not attr.startswith("__") and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                
                # Mask sensitive values
                if any(sensitive in attr.lower() for sensitive in ["key", "secret", "password", "token"]):
                    value = "*****MASKED*****"
                
                config_values[attr] = value
        
        log(f"Config values:")
        for key, value in config_values.items():
            log(f"  {key}: {value}")
        
        return {
            "module_imported": True,
            "config_values": config_values
        }
    except Exception as e:
        log(f"Error testing config module: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_health_monitor():
    """Test the health monitor module."""
    try:
        # Import the module
        log("Importing health_monitor module...")
        sys.path.insert(0, os.getcwd())
        from health_monitor import get_monitor
        
        # Initialize the monitor
        log("Initializing health monitor...")
        monitor = get_monitor()
        
        # Run a health check
        log("Running health check...")
        monitor.check_health()
        
        # Get health data
        health_data = monitor.get_health_data()
        
        # Log summary
        log("Health check completed successfully")
        log(f"Health summary: {monitor.get_health_summary()}")
        
        return {
            "module_imported": True,
            "initialized": True,
            "health_check_run": True,
            "health_data": health_data
        }
    except Exception as e:
        log(f"Error testing health monitor: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_monitor_module():
    """Test the monitor module."""
    try:
        # Import the module
        log("Importing monitor module...")
        sys.path.insert(0, os.getcwd())
        from monitor import ArbitrageMonitor
        
        # Initialize the monitor
        log("Initializing ArbitrageMonitor...")
        monitor = ArbitrageMonitor()
        
        # Don't actually start the monitor, just check if it initialized
        log("ArbitrageMonitor initialized successfully")
        
        return {
            "module_imported": True,
            "initialized": True
        }
    except Exception as e:
        log(f"Error testing monitor module: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_main_module():
    """Test the main module."""
    try:
        # Import the module
        log("Importing main module...")
        sys.path.insert(0, os.getcwd())
        import main
        
        # Check if parse_args function exists
        log("Checking parse_args function...")
        if hasattr(main, "parse_args") and callable(main.parse_args):
            log("parse_args function exists")
        else:
            log("parse_args function not found")
        
        # Check if main function exists
        log("Checking main function...")
        if hasattr(main, "main") and callable(main.main):
            log("main function exists")
        else:
            log("main function not found")
        
        return {
            "module_imported": True,
            "has_parse_args": hasattr(main, "parse_args") and callable(main.parse_args),
            "has_main": hasattr(main, "main") and callable(main.main)
        }
    except Exception as e:
        log(f"Error testing main module: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_run_health_command():
    """Test running the health command."""
    try:
        log("Running 'python main.py health' command...")
        
        # Use subprocess to run the command and capture output
        process = subprocess.Popen(
            [sys.executable, "main.py", "health"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Set a timeout for the process
        timeout = 30  # seconds
        start_time = time.time()
        
        stdout, stderr = "", ""
        while process.poll() is None:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                process.kill()
                log(f"Command timed out after {timeout} seconds")
                break
            
            # Read output without blocking
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    stdout += line
            
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    stderr += line
            
            # Don't hog the CPU
            time.sleep(0.1)
        
        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout += remaining_stdout
        if remaining_stderr:
            stderr += remaining_stderr
        
        exit_code = process.returncode
        
        log(f"Command completed with exit code: {exit_code}")
        log(f"STDOUT:")
        log(stdout)
        log(f"STDERR:")
        log(stderr)
        
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        }
    except Exception as e:
        log(f"Error running health command: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_run_monitor_command():
    """Test running the monitor command briefly."""
    try:
        log("Running 'python main.py run --mode monitor' command for 5 seconds...")
        
        # Use subprocess to run the command and capture output
        process = subprocess.Popen(
            [sys.executable, "main.py", "run", "--mode", "monitor"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let it run for a short time
        timeout = 5  # seconds
        start_time = time.time()
        
        stdout, stderr = "", ""
        while process.poll() is None:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                process.terminate()
                log(f"Terminated monitor after {timeout} seconds (as planned)")
                break
            
            # Read output without blocking
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    stdout += line
            
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    stderr += line
            
            # Don't hog the CPU
            time.sleep(0.1)
        
        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout += remaining_stdout
        if remaining_stderr:
            stderr += remaining_stderr
        
        exit_code = process.returncode
        
        log(f"Command completed with exit code: {exit_code}")
        log(f"STDOUT:")
        log(stdout)
        log(f"STDERR:")
        log(stderr)
        
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        }
    except Exception as e:
        log(f"Error running monitor command: {str(e)}")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """Run all diagnostic tests."""
    log(f"Starting KrakenBot diagnostics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Diagnostic log file: {os.path.abspath(log_file)}")
    log(f"{'=' * 50}")
    
    # Run all tests
    run_test("System Information", test_system_info)
    run_test("Directory Structure", test_directories)
    run_test("Core Files", test_core_files)
    run_test("Environment File", test_env_file)
    run_test("Python Dependencies", test_python_dependencies)
    run_test("API Connection", test_api_connection)
    run_test("Config Module", test_config_module)
    run_test("Exchange Module", test_exchange_module)
    run_test("Health Monitor Module", test_health_monitor)
    run_test("Monitor Module", test_monitor_module)
    run_test("Main Module", test_main_module)
    run_test("Health Command", test_run_health_command)
    run_test("Monitor Command", test_run_monitor_command)
    
    log(f"{'=' * 50}")
    log(f"Diagnostics completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file: {os.path.abspath(log_file)}")
    
    print(f"\nDiagnostics completed. Please check the log file for details:")
    print(f"{os.path.abspath(log_file)}")

if __name__ == "__main__":
    main()