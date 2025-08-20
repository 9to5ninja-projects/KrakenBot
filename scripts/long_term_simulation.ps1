# PowerShell script to run a long-term simulation with position sizing

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this simulation run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$simDir = "data\simulation_$timestamp"
New-Item -ItemType Directory -Path $simDir -Force | Out-Null

# First, optimize triangular paths to find the best ones
Write-Host "`n[Step 1/2] Finding optimal triangular arbitrage paths..."
python main.py optimize --days 90 --base CAD --count 10

# Run the long-term simulation
Write-Host "`n[Step 2/2] Starting long-term simulation..."
Write-Host "This simulation will run for 8 hours. You can stop it at any time with Ctrl+C."
Write-Host "Results will be saved to: $simDir"

# Run the simulation for 8 hours (28800 seconds)
$endTime = (Get-Date).AddHours(8)
Write-Host "Simulation will run until: $endTime"

# Start the simulation
python main.py run --mode simulate --amount 100 --min-profit-pct 0.5 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 5 --log-level INFO

# Copy the simulation results
Copy-Item -Path "data\opportunities.csv" -Destination "$simDir\opportunities.csv" -Force
Copy-Item -Path "data\stats.json" -Destination "$simDir\stats.json" -Force
Copy-Item -Path "data\positions\positions.json" -Destination "$simDir\positions.json" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "data\optimal_triangles.json" -Destination "$simDir\optimal_triangles.json" -Force -ErrorAction SilentlyContinue

# Create a summary file
$summary = @"
# Long-Term Simulation Summary

- **Simulation Date:** $timestamp
- **Duration:** 8 hours (or until manually stopped)
- **Mode:** Simulation (no actual trades)
- **Starting Amount:** $100 CAD
- **Min Profit Threshold:** 0.5%
- **Position Size:** 2% of capital ($2.00)
- **Max Positions:** 10
- **Max Capital:** $100.00
- **Check Interval:** 5 seconds

## Simulation Strategy

This long-term simulation used a 2% position sizing strategy with multiple concurrent positions.
The system was configured to:

1. Monitor optimal triangular arbitrage paths identified from 90 days of historical data
2. Open positions with 2% of total capital ($2.00 per position)
3. Allow up to 10 concurrent positions (maximum $20.00 deployed at once)
4. Track all opportunities and positions over an 8-hour period

## Results Analysis

To analyze the results:
1. Check positions.json for details on all positions opened during the simulation
2. Review opportunities.csv for all arbitrage opportunities detected
3. Examine stats.json for overall performance metrics

## Next Steps

Based on these simulation results, consider:
1. Adjusting the position size percentage
2. Modifying the profit threshold
3. Changing the maximum number of concurrent positions
4. Running a live data collection session during optimal trading hours
"@

$summary | Out-File -FilePath "$simDir\README.md"

Write-Host "`nLong-term simulation complete!"
Write-Host "Results saved to: $simDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")