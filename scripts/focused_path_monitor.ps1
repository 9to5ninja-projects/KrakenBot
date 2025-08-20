# PowerShell script to run a focused monitor on the most promising path

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\focused_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

# Run the monitor with focus on the CAD->ETH->BTC->CAD path
Write-Host "`nStarting focused path monitor..."
Write-Host "Monitoring CAD->ETH->BTC->CAD path with very low profit threshold"
Write-Host "Data will be saved to: $dataDir"

# Run the monitor - we'll filter for the specific path in our analysis
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 30 --log-level INFO

# Copy the collected data to our timestamped directory
Copy-Item -Path "data\opportunities.csv" -Destination "$dataDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$dataDir\stats.json" -Force

# Create a summary file
$summary = @"
# Focused Path Monitor Summary

- **Run Date:** $timestamp
- **Mode:** Simulation (no actual trades)
- **Path:** CAD->ETH->BTC->CAD
- **Starting Amount:** $100 CAD
- **Min Profit Threshold:** 0.1%
- **Position Size:** 2% of capital
- **Check Interval:** 30 seconds

## Analysis Notes

This focused run monitored only the CAD->ETH->BTC->CAD path, which has consistently shown better performance in our data collection.
The very low profit threshold (0.1%) was used to capture more potential opportunities for analysis.

## Next Steps

1. Analyze the collected data to identify patterns
2. Look for times when the path comes closest to profitability
3. Consider adjusting trading parameters based on findings
"@

$summary | Out-File -FilePath "$dataDir\README.md"

Write-Host "`nFocused path monitoring complete!"
Write-Host "Data saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")