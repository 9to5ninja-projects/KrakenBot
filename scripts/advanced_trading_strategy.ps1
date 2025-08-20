# PowerShell script to run the advanced trading strategy

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Step 1: Find optimal triangular arbitrage paths
Write-Host "`n[Step 1/3] Finding optimal triangular arbitrage paths..."
python main.py optimize --days 90 --base CAD --count 5

# Step 2: Perform statistical analysis
Write-Host "`n[Step 2/3] Performing statistical analysis..."
python main.py analyze --symbols "BTC/CAD" "ETH/CAD" "ETH/BTC" "XRP/CAD" "LTC/CAD" --days 90 --plot

# Step 3: Run the monitor with optimal settings
Write-Host "`n[Step 3/3] Starting monitor with optimal settings..."
python main.py run --mode monitor --amount 100 --min-profit-pct 1.5 --notifications --health-monitor

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")