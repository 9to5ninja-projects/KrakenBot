# PowerShell script to run an optimized simulation with lower profit threshold

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the monitor with optimized settings
python main.py run --mode simulate --amount 100 --min-profit-pct 0.25 --max-capital 100 --position-size-pct 2 --max-positions 10 --interval 20 --log-level INFO

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")