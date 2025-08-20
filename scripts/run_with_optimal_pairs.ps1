# PowerShell script to run simulation with optimal pairs

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Check if we have a comprehensive analysis file
$analysisFiles = Get-ChildItem -Path "data" -Filter "pairs_analysis_*" -Directory | Sort-Object LastWriteTime -Descending
if ($analysisFiles.Count -eq 0) {
    Write-Host "No pair analysis found. Please run find_optimal_pairs.ps1 first."
    exit 1
}

$latestAnalysis = $analysisFiles[0].FullName
$recommendationsFile = Join-Path $latestAnalysis "pair_recommendations.txt"
$analysisFile = Join-Path $latestAnalysis "comprehensive_analysis.json"

if (-not (Test-Path $recommendationsFile) -or -not (Test-Path $analysisFile)) {
    Write-Host "Analysis files not found in $latestAnalysis"
    exit 1
}

# Read the recommendations
$recommendations = Get-Content $recommendationsFile
Write-Host "`nFound recommendations from previous analysis:"
$recommendations | ForEach-Object { Write-Host "  $_" }

# Extract the best base currency and path from the JSON file
$analysisJson = Get-Content $analysisFile | ConvertFrom-Json
$bestPath = $analysisJson.all_paths[0]
$baseCurrency = $bestPath.base_currency
$path = $bestPath.path
$meanProfit = $bestPath.mean_profit
$maxProfit = $bestPath.max_profit

Write-Host "`nBest path identified:"
Write-Host "  Base Currency: $baseCurrency"
Write-Host "  Path: $path"
Write-Host "  Mean Profit: $meanProfit%"
Write-Host "  Max Profit: $maxProfit%"

# Start the dashboard in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit -Command `"cd 'e:\KrakenBot'; streamlit run dashboard.py`""

# Wait a moment for the dashboard to start
Start-Sleep -Seconds 3

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\optimal_run_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

Write-Host "`nStarting simulation with optimal trading pairs..."
Write-Host "Using base currency: $baseCurrency"
Write-Host "Data will be saved to: $dataDir"

# First, optimize triangular paths with the best base currency
Write-Host "`nOptimizing triangular paths with $baseCurrency as base..."
python main.py optimize --days 90 --base $baseCurrency --count 10

# Run the monitor with optimized settings
Write-Host "`nRunning simulation with optimized settings..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 30 --log-level INFO

# Copy the collected data to our timestamped directory
Copy-Item -Path "data\opportunities.csv" -Destination "$dataDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$dataDir\stats.json" -Force
Copy-Item -Path "data\optimal_triangles.json" -Destination "$dataDir\optimal_triangles.json" -Force

# Create a summary file
$summary = @"
# Optimal Pairs Simulation Summary

- **Run Date:** $timestamp
- **Base Currency:** $baseCurrency
- **Best Path:** $path
- **Mean Profit in Analysis:** $meanProfit%
- **Max Profit in Analysis:** $maxProfit%
- **Mode:** Simulation (no actual trades)
- **Starting Amount:** $100 $baseCurrency
- **Min Profit Threshold:** 0.1%
- **Position Size:** 2% of capital
- **Check Interval:** 30 seconds

## Analysis Notes

This run used the optimal trading pairs identified from our comprehensive analysis across multiple base currencies.

## Recommendations

$($recommendations -join "`n")
"@

$summary | Out-File -FilePath "$dataDir\README.md"

Write-Host "`nOptimal pairs simulation complete!"
Write-Host "Data saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")