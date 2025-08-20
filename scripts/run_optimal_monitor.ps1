# PowerShell script to run the monitor with optimal triangles

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the monitor with optimal triangles
python main.py run --mode monitor --amount 100 --min-profit-pct 1.5 --notifications --health-monitor

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")