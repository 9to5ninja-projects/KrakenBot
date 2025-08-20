# PowerShell script to run a comprehensive data collection and analysis

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this analysis run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$analysisDir = "data\analysis_$timestamp"
New-Item -ItemType Directory -Path $analysisDir -Force | Out-Null

Write-Host "`n[Step 1/4] Finding optimal triangular arbitrage paths..."
python main.py optimize --days 90 --base CAD --count 10

Write-Host "`n[Step 2/4] Performing statistical analysis..."
python main.py analyze --symbols "BTC/CAD" "ETH/CAD" "ETH/BTC" "XRP/CAD" "LTC/CAD" --days 90 --plot

Write-Host "`n[Step 3/4] Running simulation with position sizing..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.5 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 3 --log-level INFO

Write-Host "`n[Step 4/4] Collecting and organizing results..."

# Copy the collected data to our analysis directory
Copy-Item -Path "data\opportunities.csv" -Destination "$analysisDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$analysisDir\stats.json" -Force
Copy-Item -Path "data\positions\positions.json" -Destination "$analysisDir\positions.json" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "data\optimal_triangles.json" -Destination "$analysisDir\optimal_triangles.json" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "data\statistical\*.json" -Destination "$analysisDir\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "data\statistical\*.png" -Destination "$analysisDir\" -Force -ErrorAction SilentlyContinue

# Create a summary file
$summary = @"
# Comprehensive Analysis Summary

- **Analysis Date:** $timestamp
- **Data Collection:** Simulation mode
- **Starting Amount:** $100 CAD
- **Min Profit Threshold:** 0.5%
- **Position Size:** 2% of capital
- **Max Positions:** 10
- **Check Interval:** 3 seconds

## Analysis Components

1. **Optimal Triangle Discovery**
   - Analyzed 90 days of historical data
   - Identified most profitable triangular paths
   - Results saved in optimal_triangles.json

2. **Statistical Analysis**
   - Analyzed volatility patterns
   - Identified optimal trading hours
   - Generated trading recommendations
   - Results saved in statistical JSON files and PNG charts

3. **Position Sizing Simulation**
   - Tested 2% position sizing strategy
   - Tracked multiple concurrent positions
   - Results saved in positions.json

## Next Steps

1. Review the optimal triangular paths and focus on the top performers
2. Adjust trading parameters based on statistical findings
3. Run a longer simulation with the optimized parameters
4. Consider running a live data collection session during high volatility periods
"@

$summary | Out-File -FilePath "$analysisDir\README.md"

Write-Host "`nComprehensive analysis complete!"
Write-Host "Results saved to: $analysisDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")