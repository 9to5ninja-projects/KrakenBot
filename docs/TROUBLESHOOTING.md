# KrakenBot Troubleshooting Guide

This guide will help you diagnose and fix common issues with KrakenBot.

## Quick Fix Tools

KrakenBot includes two tools to help diagnose and fix issues:

1. **Diagnostic Tool**: Runs comprehensive diagnostics and logs the results
   - Windows: Run `run_diagnostics.bat`
   - Linux/Mac: Run `./run_diagnostics.sh`

2. **Fix Common Issues Tool**: Automatically fixes the most common problems
   - Windows: Run `fix_issues.bat`
   - Linux/Mac: Run `./fix_issues.sh`

## Common Issues and Solutions

### Terminal Crashes When Running KrakenBot

**Symptoms:**
- Terminal/command prompt closes unexpectedly
- No error message is displayed

**Solutions:**
1. Run the diagnostic tool to capture errors
2. Check the log files in the `logs` directory
3. Run KrakenBot from a terminal that won't close on error:
   ```
   python main.py run --mode monitor --log-level DEBUG
   ```

### "ModuleNotFoundError" or "ImportError"

**Symptoms:**
- Error message mentions missing module
- "No module named X" error

**Solutions:**
1. Reinstall dependencies:
   ```
   pip install -r requirements.txt --force-reinstall
   ```
2. Run the fix tool to repair dependencies
3. Check if the virtual environment is activated

### API Connection Errors

**Symptoms:**
- "ConnectionError" or "ConnectionRefusedError"
- "Timeout" errors
- "SSL Certificate" errors

**Solutions:**
1. Check your internet connection
2. Verify your API keys in the `.env` file
3. Try running without API keys (monitoring mode only)
4. Check if Kraken API is down: https://status.kraken.com/

### Configuration Errors

**Symptoms:**
- "KeyError" or "AttributeError"
- "No such file or directory" errors
- Missing configuration values

**Solutions:**
1. Run the fix tool to repair the `.env` file
2. Check if the `.env` file exists and has the required variables
3. Run the setup wizard to reconfigure:
   ```
   python setup_wizard.py
   ```

### Permission Errors

**Symptoms:**
- "PermissionError" or "Access is denied"
- Cannot write to files or directories

**Solutions:**
1. Run the fix tool to repair permissions
2. Run as administrator (Windows) or with sudo (Linux)
3. Check if the data and logs directories exist and are writable

### Health Monitor Crashes

**Symptoms:**
- Error when running health check
- "AttributeError" or "TypeError" in health_monitor.py

**Solutions:**
1. Run the diagnostic tool to identify the specific issue
2. Run the fix tool to repair dependencies
3. Try running without health monitoring:
   ```
   python main.py run --mode monitor
   ```

### Dashboard Errors

**Symptoms:**
- Streamlit errors
- "No module named streamlit"
- Dashboard doesn't start

**Solutions:**
1. Install Streamlit:
   ```
   pip install streamlit
   ```
2. Check if data files exist for the dashboard to read
3. Run in monitor mode first to generate data:
   ```
   python main.py run --mode monitor
   ```

## Advanced Troubleshooting

If the quick fix tools don't resolve your issue, try these advanced troubleshooting steps:

### Check Python Version

KrakenBot requires Python 3.6 or higher:
```
python --version
```

### Check Virtual Environment

Make sure you're using the correct virtual environment:
```
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Check Directory Structure

Ensure all required directories exist:
```
mkdir -p data logs data/reports data/historical
```

### Check File Permissions

Set correct permissions on Linux/Mac:
```
chmod +x *.sh
chmod -R 755 .
```

### Reinstall from Scratch

If all else fails, try a clean installation:
1. Backup your `.env` file
2. Delete the `venv` directory
3. Create a new virtual environment
4. Reinstall dependencies
5. Restore your `.env` file

## Reading Diagnostic Logs

The diagnostic tool creates a log file with detailed information about your system and KrakenBot installation. Here's how to interpret it:

1. **System Information**: Basic info about your OS and Python version
2. **Directory Structure**: Checks if all required directories exist
3. **Core Files**: Verifies all essential files are present
4. **Environment File**: Checks if the `.env` file exists and has required variables
5. **Python Dependencies**: Lists all installed dependencies and their versions
6. **API Connection**: Tests connection to Kraken API
7. **Module Tests**: Tests each core module of KrakenBot
8. **Command Tests**: Tests running basic commands

Look for "FAILED" in the log to identify specific issues.

## Getting Help

If you've tried all the troubleshooting steps and still have issues:

1. Run the diagnostic tool and save the log file
2. Check the KrakenBot documentation
3. Search for similar issues online
4. Reach out to the community for help

## Preventative Measures

To avoid issues in the future:

1. Always use the virtual environment
2. Don't modify core files unless you know what you're doing
3. Keep dependencies up to date
4. Regularly backup your configuration
5. Monitor system resources (CPU, memory, disk space)

## Recovering from Crashes

If KrakenBot crashes during operation:

1. Check the logs to identify the cause
2. Run the diagnostic tool
3. Fix any issues identified
4. Restart KrakenBot

The system is designed to be resilient and will recover from most errors automatically.