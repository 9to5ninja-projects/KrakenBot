# PowerShell script to run the KrakenBot dashboard
# This script activates the virtual environment and starts the Streamlit dashboard

# Get the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Activate virtual environment
$activateScript = Join-Path $projectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript
} else {
    Write-Error "Virtual environment activation script not found: $activateScript"
    Write-Host "Please run install.ps1 first to set up the environment."
    exit 1
}

# Check if Streamlit is installed
try {
    $streamlitVersion = streamlit --version
    Write-Host "Found Streamlit: $streamlitVersion"
} catch {
    Write-Error "Streamlit is not installed. Please install it with: pip install streamlit"
    exit 1
}

# Start the dashboard
$dashboardPath = Join-Path $projectRoot "dashboard.py"
if (Test-Path $dashboardPath) {
    Write-Host "Starting Streamlit dashboard..."
    streamlit run $dashboardPath
} else {
    Write-Error "Dashboard file not found: $dashboardPath"
    exit 1
}