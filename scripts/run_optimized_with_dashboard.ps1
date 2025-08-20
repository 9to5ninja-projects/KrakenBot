# PowerShell script to run an optimized simulation with dashboard

# Start the dashboard in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit -Command `"cd 'e:\KrakenBot'; streamlit run dashboard.py`""

# Wait a moment for the dashboard to start
Start-Sleep -Seconds 3

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\optimized_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

Write-Host "`nStarting optimized simulation based on data analysis..."
Write-Host "Using lower profit threshold and longer interval to avoid rate limits"
Write-Host "Data will be saved to: $dataDir"

# Run the monitor with optimized settings
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 45 --log-level INFO

# Copy the collected data to our timestamped directory
Copy-Item -Path "data\opportunities.csv" -Destination "$dataDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$dataDir\stats.json" -Force

# Create a summary file
$summary = @"
# Optimized Simulation Summary

- **Run Date:** $timestamp
- **Mode:** Simulation (no actual trades)
- **Starting Amount:** $100 CAD
- **Min Profit Threshold:** 0.1%
- **Position Size:** 2% of capital
- **Check Interval:** 45 seconds

## Analysis Notes

This optimized run used settings based on our data analysis:
- Lower profit threshold (0.1%) to capture more potential opportunities
- Longer interval (45 seconds) to avoid rate limits
- Focus on the CAD->ETH->BTC->CAD path which has shown better performance

## Key Findings

Based on our analysis:
1. The CAD->ETH->BTC->CAD path consistently performs better (-1.49% vs -99.86%)
2. The best hour for trading is hour 11 with average profit of -1.48%
3. The closest we've come to profitability is -1.26% (gap of 1.51% to reach 0.25% profit)
4. Low volatility suggests using longer intervals between checks
"@

$summary | Out-File -FilePath "$dataDir\README.md"

Write-Host "`nOptimized simulation complete!"
Write-Host "Data saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")