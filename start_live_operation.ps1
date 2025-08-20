# KrakenBot Live Trading Operation Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   KrakenBot Live Trading Operation" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will start:" -ForegroundColor Green
Write-Host "- Live trading session with real data collection" -ForegroundColor White
Write-Host "- Dashboard on http://localhost:8505" -ForegroundColor White
Write-Host "- Real-time session monitoring and analysis" -ForegroundColor White
Write-Host "- Comprehensive hourly reports" -ForegroundColor White
Write-Host ""

$response = Read-Host "Start live trading operation? (y/N)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Starting live trading operation..." -ForegroundColor Green
    Write-Host ""
    
    # Change to the script directory
    Set-Location $PSScriptRoot
    
    # Start the live trading session in a new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python start_live_trading_session.py" -WindowStyle Normal
    
    # Wait for the session to initialize
    Start-Sleep -Seconds 10
    
    # Start the session monitor in a new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python session_monitor.py" -WindowStyle Normal
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " Live Trading Operation Started!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Dashboard: http://localhost:8505" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Two windows opened:" -ForegroundColor Green
    Write-Host "1. Live Trading Session - Main trading loop" -ForegroundColor White
    Write-Host "2. Session Monitor - Real-time analysis" -ForegroundColor White
    Write-Host ""
    Write-Host "Data will be saved in the 'data/live_session_*' directory" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to close this window..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} else {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
}