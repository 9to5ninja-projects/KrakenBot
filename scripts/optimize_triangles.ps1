# PowerShell script to run the triangle optimizer

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the triangle optimizer
python main.py optimize --days 90 --base CAD --count 5

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")