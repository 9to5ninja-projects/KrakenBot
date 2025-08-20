# PowerShell installation script for KrakenBot
# This script sets up the virtual environment and installs dependencies

# Create directories if they don't exist
$dataDir = Join-Path $PSScriptRoot "..\data"
$logsDir = Join-Path $PSScriptRoot "..\logs"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force
    Write-Host "Created data directory: $dataDir"
}

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force
    Write-Host "Created logs directory: $logsDir"
}

# Create virtual environment if it doesn't exist
$venvDir = Join-Path $PSScriptRoot "..\venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Make sure Python is installed."
        exit 1
    }
    Write-Host "Virtual environment created at: $venvDir"
}

# Activate virtual environment
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript
} else {
    Write-Error "Virtual environment activation script not found: $activateScript"
    exit 1
}

# Install dependencies
$requirementsFile = Join-Path $PSScriptRoot "..\requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing dependencies from $requirementsFile..."
    pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies."
        exit 1
    }
    Write-Host "Dependencies installed successfully."
} else {
    Write-Error "Requirements file not found: $requirementsFile"
    exit 1
}

# Create .env file from example if it doesn't exist
$envFile = Join-Path $PSScriptRoot "..\.env"
$envExampleFile = Join-Path $PSScriptRoot "..\.env.example"

if (-not (Test-Path $envFile) -and (Test-Path $envExampleFile)) {
    Write-Host "Creating .env file from example..."
    Copy-Item -Path $envExampleFile -Destination $envFile
    Write-Host ".env file created. Please edit it with your settings."
}

Write-Host "Installation completed successfully!"
Write-Host "To run the bot, use: python main.py"
Write-Host "For more options, use: python main.py --help"