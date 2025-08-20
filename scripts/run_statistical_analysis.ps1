# PowerShell script to run the statistical analyzer

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the statistical analyzer with default pairs and generate plots
python main.py analyze --symbols "BTC/CAD" "ETH/CAD" "ETH/BTC" "XRP/CAD" "LTC/CAD" --days 90 --plot

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")