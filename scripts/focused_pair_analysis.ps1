# PowerShell script to find optimal trading pairs with rate limit protection

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Create a timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dataDir = "data\focused_pairs_$timestamp"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

Write-Host "`nStarting focused pair analysis..."
Write-Host "Testing key base currencies with rate limit protection"
Write-Host "Results will be saved to: $dataDir"

# Focus on just 3 major base currencies
$baseCurrencies = @("CAD", "USD", "EUR")

foreach ($base in $baseCurrencies) {
    Write-Host "`nAnalyzing triangular paths with base currency: $base"
    
    # Create directory for this base currency
    $baseDir = Join-Path $dataDir $base
    New-Item -ItemType Directory -Path $baseDir -Force | Out-Null
    
    # Run the optimizer for this base currency with a smaller count
    python main.py optimize --days 30 --base $base --count 5
    
    # Copy optimization results
    if (Test-Path "data\optimal_triangles.json") {
        Copy-Item -Path "data\optimal_triangles.json" -Destination "$baseDir\optimal_triangles.json" -Force
    }
    
    # Run a quick simulation with the top path (just 2 minutes)
    Write-Host "Running quick simulation with top $base paths..."
    python main.py run --mode simulate --amount 100 --min-profit-pct 0.1 --max-capital 100 --position-size-pct 2 --max-positions 1 --interval 45 --log-level INFO --duration 120
    
    # Copy the simulation results
    if (Test-Path "data\opportunities.csv") {
        Copy-Item -Path "data\opportunities.csv" -Destination "$baseDir\opportunities.csv" -Force
    }
    if (Test-Path "data\stats.json") {
        Copy-Item -Path "data\stats.json" -Destination "$baseDir\stats.json" -Force
    }
    
    # Wait longer between currencies to avoid rate limits
    Write-Host "Waiting 3 minutes to avoid rate limits..."
    Start-Sleep -Seconds 180
}

# Now analyze all the collected data
Write-Host "`nAnalyzing results from all base currencies..."

python -c @'
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Base directory for our analysis
base_dir = Path("data/focused_pairs_' + $timestamp + '")
results = {}

# Process each base currency
for currency_dir in base_dir.glob("*"):
    if not currency_dir.is_dir():
        continue
        
    base_currency = currency_dir.name
    print(f"\nAnalyzing {base_currency} paths:")
    
    # Load optimal triangles
    triangles_file = currency_dir / "optimal_triangles.json"
    if triangles_file.exists():
        with open(triangles_file, "r") as f:
            triangles = json.load(f)
        
        # Load opportunities if they exist
        opps_file = currency_dir / "opportunities.csv"
        if opps_file.exists():
            try:
                opps = pd.read_csv(opps_file)
                
                # Group by path and calculate stats
                path_stats = {}
                for path, group in opps.groupby("path"):
                    path_stats[path] = {
                        "count": len(group),
                        "mean_profit": group["profit_percentage"].mean(),
                        "max_profit": group["profit_percentage"].max(),
                        "min_profit": group["profit_percentage"].min(),
                        "std_profit": group["profit_percentage"].std()
                    }
                
                # Store results
                results[base_currency] = {
                    "triangles": triangles,
                    "path_stats": path_stats
                }
                
                # Print summary
                print(f"  Found {len(triangles)} optimal triangles")
                print(f"  Collected {len(opps)} opportunities")
                
                # Show the top 3 paths by mean profit
                if path_stats:
                    sorted_paths = sorted(path_stats.items(), 
                                         key=lambda x: x[1]["mean_profit"], 
                                         reverse=True)
                    
                    print("\n  Top paths by mean profit:")
                    for i, (path, stats) in enumerate(sorted_paths[:3], 1):
                        print(f"  {i}. {path}: {stats['mean_profit']:.4f}% (max: {stats['max_profit']:.4f}%)")
                
            except Exception as e:
                print(f"  Error processing opportunities: {e}")
        else:
            print(f"  No opportunities data found")
    else:
        print(f"  No optimal triangles found")

# Find the overall best paths
all_paths = []
for base_currency, data in results.items():
    if "path_stats" in data:
        for path, stats in data["path_stats"].items():
            all_paths.append({
                "base_currency": base_currency,
                "path": path,
                "mean_profit": stats["mean_profit"],
                "max_profit": stats["max_profit"],
                "count": stats["count"]
            })

# Sort by mean profit
all_paths.sort(key=lambda x: x["mean_profit"], reverse=True)

print("\n=== OVERALL BEST TRIANGULAR ARBITRAGE PATHS ===")
for i, path_data in enumerate(all_paths[:10], 1):
    print(f"{i}. {path_data['base_currency']} - {path_data['path']}: {path_data['mean_profit']:.4f}% (max: {path_data['max_profit']:.4f}%)")

# Save the comprehensive results
with open(base_dir / "comprehensive_analysis.json", "w") as f:
    json.dump({
        "timestamp": pd.Timestamp.now().isoformat(),
        "all_paths": all_paths,
        "detailed_results": results
    }, f, indent=2, default=str)

# Generate recommendations
recommendations = []

if all_paths:
    # Best base currency
    best_base = max(results.keys(), key=lambda x: max([p["mean_profit"] for p in all_paths if p["base_currency"] == x], default=-100))
    recommendations.append(f"Use {best_base} as your base currency for triangular arbitrage")
    
    # Best path
    best_path = all_paths[0]
    recommendations.append(f"Focus on the {best_path['path']} path with {best_path['base_currency']} as base")
    
    # If any path is close to profitability
    if any(p["max_profit"] > -1.0 for p in all_paths):
        close_paths = [p for p in all_paths if p["max_profit"] > -1.0]
        for p in close_paths[:3]:
            recommendations.append(f"The {p['path']} path with {p['base_currency']} came within {abs(p['max_profit']):.2f}% of profitability")

# Save recommendations
with open(base_dir / "pair_recommendations.txt", "w") as f:
    f.write("# Optimal Trading Pair Recommendations\n\n")
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec}\n")

print("\nRecommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print(f"\nComprehensive analysis saved to {base_dir}/comprehensive_analysis.json")
print(f"Recommendations saved to {base_dir}/pair_recommendations.txt")
'@

Write-Host "`nFocused pair analysis complete!"
Write-Host "Results saved to: $dataDir"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")