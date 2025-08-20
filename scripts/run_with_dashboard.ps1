# PowerShell script to run the monitor with position sizing and dashboard

# Start the dashboard in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit -Command `"cd 'e:\KrakenBot'; streamlit run dashboard.py`""

# Wait a moment for the dashboard to start
Start-Sleep -Seconds 3

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the monitor with position sizing
python main.py run --mode simulate --amount 100 --min-profit-pct 0.5 --max-capital 100 --position-size-pct 2 --max-positions 5 --interval 20 --notifications --health-monitor

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")