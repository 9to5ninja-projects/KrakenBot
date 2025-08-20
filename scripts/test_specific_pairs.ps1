# PowerShell script to test specific trading pairs

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\specific_pairs_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

Write-Host "`nStarting specific pair testing..."
Write-Host "Testing CAD->ETH->BTC->CAD vs USD->ETH->BTC->USD"
Write-Host "Results will be saved to: $dataDir"

# Test CAD base currency
$cadDir = Join-Path $dataDir "CAD"
New-Item -ItemType Directory -Path $cadDir -Force | Out-Null

Write-Host "`nTesting CAD->ETH->BTC->CAD path..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 1 --interval 45 --log-level INFO --pairs "ETH/CAD,ETH/BTC,BTC/CAD" --duration 180

# Copy the simulation results
if (Test-Path "data\opportunities.csv") {
    Copy-Item -Path "data\opportunities.csv" -Destination "$cadDir\opportunities.csv" -Force
}
if (Test-Path "data\stats.json") {
    Copy-Item -Path "data\stats.json" -Destination "$cadDir\stats.json" -Force
}

# Wait to avoid rate limits
Write-Host "Waiting 2 minutes to avoid rate limits..."
Start-Sleep -Seconds 120

# Test USD base currency
$usdDir = Join-Path $dataDir "USD"
New-Item -ItemType Directory -Path $usdDir -Force | Out-Null

Write-Host "`nTesting USD->ETH->BTC->USD path..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 1 --interval 45 --log-level INFO --pairs "ETH/USD,ETH/BTC,BTC/USD" --duration 180

# Copy the simulation results
if (Test-Path "data\opportunities.csv") {
    Copy-Item -Path "data\opportunities.csv" -Destination "$usdDir\opportunities.csv" -Force
}
if (Test-Path "data\stats.json") {
    Copy-Item -Path "data\stats.json" -Destination "$usdDir\stats.json" -Force
}

# Wait to avoid rate limits
Write-Host "Waiting 2 minutes to avoid rate limits..."
Start-Sleep -Seconds 120

# Test EUR base currency
$eurDir = Join-Path $dataDir "EUR"
New-Item -ItemType Directory -Path $eurDir -Force | Out-Null

Write-Host "`nTesting EUR->ETH->BTC->EUR path..."
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 1 --interval 45 --log-level INFO --pairs "ETH/EUR,ETH/BTC,BTC/EUR" --duration 180

# Copy the simulation results
if (Test-Path "data\opportunities.csv") {
    Copy-Item -Path "data\opportunities.csv" -Destination "$eurDir\opportunities.csv" -Force
}
if (Test-Path "data\stats.json") {
    Copy-Item -Path "data\stats.json" -Destination "$eurDir\stats.json" -Force
}

# Now analyze all the collected data
Write-Host "`nAnalyzing results from all currencies..."

# Create a Python script for analysis
$pythonScript = @"
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Base directory for our analysis
base_dir = Path('data/specific_pairs_$timestamp')
results = {}

# Process each currency directory
for currency_dir in base_dir.glob('*'):
    if not currency_dir.is_dir():
        continue
        
    base_currency = currency_dir.name
    print(f'\nAnalyzing {base_currency} paths:')
    
    # Load opportunities if they exist
    opps_file = currency_dir / 'opportunities.csv'
    if opps_file.exists():
        try:
            opps = pd.read_csv(opps_file)
            
            # Group by path and calculate stats
            path_stats = {}
            for path, group in opps.groupby('path'):
                path_stats[path] = {
                    'count': len(group),
                    'mean_profit': group['profit_percentage'].mean(),
                    'max_profit': group['profit_percentage'].max(),
                    'min_profit': group['profit_percentage'].min(),
                    'std_profit': group['profit_percentage'].std()
                }
            
            # Store results
            results[base_currency] = {
                'path_stats': path_stats
            }
            
            # Print summary
            print(f'  Collected {len(opps)} opportunities')
            
            # Show all paths by mean profit
            if path_stats:
                sorted_paths = sorted(path_stats.items(), 
                                     key=lambda x: x[1]['mean_profit'], 
                                     reverse=True)
                
                print('\n  All paths by mean profit:')
                for i, (path, stats) in enumerate(sorted_paths, 1):
                    print(f'  {i}. {path}: {stats["mean_profit"]:.4f}% (max: {stats["max_profit"]:.4f}%)')
            
        except Exception as e:
            print(f'  Error processing opportunities: {e}')
    else:
        print(f'  No opportunities data found')

# Find the overall best paths
all_paths = []
for base_currency, data in results.items():
    if 'path_stats' in data:
        for path, stats in data['path_stats'].items():
            all_paths.append({
                'base_currency': base_currency,
                'path': path,
                'mean_profit': stats['mean_profit'],
                'max_profit': stats['max_profit'],
                'count': stats['count']
            })

# Sort by mean profit
all_paths.sort(key=lambda x: x['mean_profit'], reverse=True)

print('\n=== OVERALL BEST TRIANGULAR ARBITRAGE PATHS ===')
for i, path_data in enumerate(all_paths, 1):
    print(f'{i}. {path_data["base_currency"]} - {path_data["path"]}: {path_data["mean_profit"]:.4f}% (max: {path_data["max_profit"]:.4f}%)')

# Save the comprehensive results
with open(base_dir / 'analysis.json', 'w') as f:
    json.dump({
        'timestamp': pd.Timestamp.now().isoformat(),
        'all_paths': all_paths,
        'detailed_results': results
    }, f, indent=2, default=str)

# Generate recommendations
recommendations = []

if all_paths:
    # Best base currency and path
    best_path = all_paths[0]
    recommendations.append(f'Focus on the {best_path["path"]} path with {best_path["base_currency"]} as base')
    
    # Compare base currencies
    currencies = set(p['base_currency'] for p in all_paths)
    if len(currencies) > 1:
        currency_stats = {}
        for currency in currencies:
            currency_paths = [p for p in all_paths if p['base_currency'] == currency]
            if currency_paths:
                currency_stats[currency] = {
                    'mean_profit': sum(p['mean_profit'] for p in currency_paths) / len(currency_paths),
                    'max_profit': max(p['max_profit'] for p in currency_paths)
                }
        
        # Sort currencies by mean profit
        sorted_currencies = sorted(currency_stats.items(), key=lambda x: x[1]['mean_profit'], reverse=True)
        recommendations.append(f'{sorted_currencies[0][0]} is the best base currency for triangular arbitrage')
        
        # Compare with other currencies
        for i in range(1, len(sorted_currencies)):
            diff = sorted_currencies[0][1]['mean_profit'] - sorted_currencies[i][1]['mean_profit']
            recommendations.append(f'{sorted_currencies[0][0]} outperforms {sorted_currencies[i][0]} by {diff:.4f}%')

# Save recommendations
with open(base_dir / 'recommendations.txt', 'w') as f:
    f.write('# Optimal Trading Pair Recommendations\n\n')
    for i, rec in enumerate(recommendations, 1):
        f.write(f'{i}. {rec}\n')

print('\nRecommendations:')
for i, rec in enumerate(recommendations, 1):
    print(f'{i}. {rec}')

print(f'\nAnalysis saved to {base_dir}/analysis.json')
print(f'Recommendations saved to {base_dir}/recommendations.txt')
"@

# Save the Python script
$pythonScript | Out-File -FilePath "$dataDir\analyze.py" -Force

# Run the Python script
python "$dataDir\analyze.py"

Write-Host "`nSpecific pair testing complete!"
Write-Host "Results saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")