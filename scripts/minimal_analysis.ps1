# PowerShell script for minimal pair analysis with extreme rate limit protection

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\minimal_pairs_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

Write-Host "`nStarting minimal pair analysis..."
Write-Host "Testing only CAD with specific pairs"
Write-Host "Results will be saved to: $dataDir"

# Create a directory for CAD
$baseDir = Join-Path $dataDir "CAD"
New-Item -ItemType Directory -Path $baseDir -Force | Out-Null

# Create a custom triangles file with just the pairs we want to test
$trianglesJson = @"
[
  {
    "base_currency": "CAD",
    "path": "CAD->ETH->BTC->CAD",
    "pairs": ["ETH/CAD", "ETH/BTC", "BTC/CAD"],
    "directions": ["buy", "sell", "sell"],
    "historical_profit": -1.5
  },
  {
    "base_currency": "CAD",
    "path": "CAD->BTC->ETH->CAD",
    "pairs": ["BTC/CAD", "ETH/BTC", "ETH/CAD"],
    "directions": ["buy", "buy", "sell"],
    "historical_profit": -99.8
  },
  {
    "base_currency": "CAD",
    "path": "CAD->ETH->XRP->CAD",
    "pairs": ["ETH/CAD", "XRP/ETH", "XRP/CAD"],
    "directions": ["buy", "buy", "sell"],
    "historical_profit": -2.0
  },
  {
    "base_currency": "CAD",
    "path": "CAD->XRP->ETH->CAD",
    "pairs": ["XRP/CAD", "XRP/ETH", "ETH/CAD"],
    "directions": ["buy", "sell", "sell"],
    "historical_profit": -2.0
  },
  {
    "base_currency": "CAD",
    "path": "CAD->BTC->XRP->CAD",
    "pairs": ["BTC/CAD", "XRP/BTC", "XRP/CAD"],
    "directions": ["buy", "buy", "sell"],
    "historical_profit": -2.0
  }
]
"@

# Save the custom triangles file
$trianglesJson | Out-File -FilePath "data\optimal_triangles.json" -Force
Copy-Item -Path "data\optimal_triangles.json" -Destination "$baseDir\optimal_triangles.json" -Force

Write-Host "`nCreated custom triangles file with 5 paths"
Write-Host "Running quick simulation with these paths..."

# Run a simulation with these paths
python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 5 --interval 45 --log-level INFO

# Copy the simulation results
if (Test-Path "data\opportunities.csv") {
    Copy-Item -Path "data\opportunities.csv" -Destination "$baseDir\opportunities.csv" -Force
}
if (Test-Path "data\stats.json") {
    Copy-Item -Path "data\stats.json" -Destination "$baseDir\stats.json" -Force
}

# Now analyze the collected data
Write-Host "`nAnalyzing results..."

# Create a Python script for analysis
$pythonScript = @"
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Base directory for our analysis
base_dir = Path('data/minimal_pairs_$timestamp')
results = {}

# Process CAD directory
currency_dir = base_dir / 'CAD'
base_currency = 'CAD'
print(f'\nAnalyzing {base_currency} paths:')

# Load optimal triangles
triangles_file = currency_dir / 'optimal_triangles.json'
if triangles_file.exists():
    with open(triangles_file, 'r') as f:
        triangles = json.load(f)
    
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
                'triangles': triangles,
                'path_stats': path_stats
            }
            
            # Print summary
            print(f'  Found {len(triangles)} triangular paths')
            print(f'  Collected {len(opps)} opportunities')
            
            # Show all paths by mean profit
            if path_stats:
                sorted_paths = sorted(path_stats.items(), 
                                     key=lambda x: x[1]['mean_profit'], 
                                     reverse=True)
                
                print('\n  All paths by mean profit:')
                for i, (path, stats) in enumerate(sorted_paths, 1):
                    print(f'  {i}. {path}: {stats["mean_profit"]:.4f}% (max: {stats["max_profit"]:.4f}%)')
            
            # Find the overall best paths
            all_paths = []
            for path, stats in path_stats.items():
                all_paths.append({
                    'base_currency': base_currency,
                    'path': path,
                    'mean_profit': stats['mean_profit'],
                    'max_profit': stats['max_profit'],
                    'count': stats['count']
                })

            # Sort by mean profit
            all_paths.sort(key=lambda x: x['mean_profit'], reverse=True)

            print('\n=== BEST TRIANGULAR ARBITRAGE PATHS ===')
            for i, path_data in enumerate(all_paths, 1):
                print(f'{i}. {path_data["path"]}: {path_data["mean_profit"]:.4f}% (max: {path_data["max_profit"]:.4f}%)')

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
                # Best path
                best_path = all_paths[0]
                recommendations.append(f'Focus on the {best_path["path"]} path with CAD as base')
                
                # If any path is close to profitability
                if any(p['max_profit'] > -1.0 for p in all_paths):
                    close_paths = [p for p in all_paths if p['max_profit'] > -1.0]
                    for p in close_paths[:3]:
                        recommendations.append(f'The {p["path"]} path came within {abs(p["max_profit"]):.2f}% of profitability')
                
                # Add specific recommendations based on the results
                eth_btc_path = next((p for p in all_paths if p['path'] == 'CAD->ETH->BTC->CAD'), None)
                btc_eth_path = next((p for p in all_paths if p['path'] == 'CAD->BTC->ETH->CAD'), None)
                
                if eth_btc_path and btc_eth_path:
                    if eth_btc_path['mean_profit'] > btc_eth_path['mean_profit']:
                        recommendations.append(f'CAD->ETH->BTC->CAD is {eth_btc_path["mean_profit"] - btc_eth_path["mean_profit"]:.2f}% better than CAD->BTC->ETH->CAD')
                    else:
                        recommendations.append(f'CAD->BTC->ETH->CAD is {btc_eth_path["mean_profit"] - eth_btc_path["mean_profit"]:.2f}% better than CAD->ETH->BTC->CAD')

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
            
        except Exception as e:
            print(f'  Error processing opportunities: {e}')
    else:
        print(f'  No opportunities data found')
else:
    print(f'  No optimal triangles found')
"@

# Save the Python script
$pythonScript | Out-File -FilePath "$dataDir\analyze.py" -Force

# Run the Python script
python "$dataDir\analyze.py"

Write-Host "`nMinimal pair analysis complete!"
Write-Host "Results saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")