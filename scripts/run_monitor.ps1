# PowerShell script to run the KrakenBot monitor
# This script activates the virtual environment and starts the arbitrage monitor

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

# Parse command line arguments
param(
    [string]$mode = "monitor",
    [double]$amount = 10000,
    [double]$threshold = 5,
    [int]$interval = 10,
    [string]$logLevel = "INFO"
)

# Start the monitor
$mainPath = Join-Path $projectRoot "main.py"
if (Test-Path $mainPath) {
    Write-Host "Starting KrakenBot in $mode mode..."
    python $mainPath --mode $mode --amount $amount --threshold $threshold --interval $interval --log-level $logLevel
} else {
    Write-Error "Main file not found: $mainPath"
    exit 1
}