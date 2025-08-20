# PowerShell script to run a live simulation with real-time data collection

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this data collection run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\collection_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

# Run the monitor with position sizing and a very low profit threshold to capture more data
Write-Host "`nStarting live data collection..."
Write-Host "Data will be saved to: $dataDir"

# Run for 4 hours (14400 seconds)
$endTime = (Get-Date).AddHours(4)
$runTime = 14400

Write-Host "Collection will run until: $endTime (approximately 4 hours)"
Write-Host "Press Ctrl+C to stop collection early"

# Start the monitor
python main.py run --mode monitor --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 30 --log-level INFO

# Copy the collected data to our timestamped directory
Copy-Item -Path "data\opportunities.csv" -Destination "$dataDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$dataDir\stats.json" -Force

# Create a summary file
$summary = @"
# Data Collection Summary

- **Collection Date:** $timestamp
- **Duration:** 4 hours (or until manually stopped)
- **Mode:** Monitor (no actual trades)
- **Starting Amount:** $100 CAD
- **Min Profit Threshold:** 0.1%
- **Position Size:** 2% of capital
- **Check Interval:** 5 seconds

## Analysis Notes

This data collection run was performed to gather real-time market data for triangular arbitrage analysis.
The very low profit threshold (0.1%) was used to capture more potential opportunities for analysis.

## Next Steps

1. Analyze the collected data to identify patterns
2. Adjust trading parameters based on findings
3. Run a simulation with the optimized parameters
"@

$summary | Out-File -FilePath "$dataDir\README.md"

Write-Host "`nData collection complete!"
Write-Host "Data saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")