# PowerShell script to run an extended simulation with more trading pairs

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# First, optimize triangular paths to find the best ones
Write-Host "`n[Step 1/2] Finding optimal triangular arbitrage paths..."
python main.py optimize --days 90 --base CAD --count 10

# Run the monitor with position sizing and a lower profit threshold
Write-Host "`n[Step 2/2] Starting extended simulation..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.5 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 20 --log-level INFO

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")