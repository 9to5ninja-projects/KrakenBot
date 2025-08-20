# PowerShell script to analyze collected opportunities data

# Navigate to the KrakenBot directory
Set-Location "e:\KrakenBot"

# Run the analysis script
python -c @'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from pathlib import Path

# Load the opportunities data
print("Loading opportunities data...")
df = pd.read_csv("data/opportunities.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Add hour column for time-based analysis
df["hour"] = df["timestamp"].dt.hour

# Filter for the CAD->ETH->BTC->CAD path
eth_btc_path = df[df["path"] == "CAD->ETH->BTC->CAD"]
btc_eth_path = df[df["path"] == "CAD->BTC->ETH->CAD"]

print(f"Total opportunities: {len(df)}")
print(f"CAD->ETH->BTC->CAD opportunities: {len(eth_btc_path)}")
print(f"CAD->BTC->ETH->CAD opportunities: {len(btc_eth_path)}")

# Calculate statistics
eth_btc_mean = eth_btc_path["profit_percentage"].mean()
eth_btc_max = eth_btc_path["profit_percentage"].max()
eth_btc_min = eth_btc_path["profit_percentage"].min()
eth_btc_std = eth_btc_path["profit_percentage"].std()

btc_eth_mean = btc_eth_path["profit_percentage"].mean()
btc_eth_max = btc_eth_path["profit_percentage"].max()
btc_eth_min = btc_eth_path["profit_percentage"].min()
btc_eth_std = btc_eth_path["profit_percentage"].std()

print("\nCAD->ETH->BTC->CAD Statistics:")
print(f"Mean Profit %: {eth_btc_mean:.4f}%")
print(f"Max Profit %: {eth_btc_max:.4f}%")
print(f"Min Profit %: {eth_btc_min:.4f}%")
print(f"Std Dev: {eth_btc_std:.4f}%")

print("\nCAD->BTC->ETH->CAD Statistics:")
print(f"Mean Profit %: {btc_eth_mean:.4f}%")
print(f"Max Profit %: {btc_eth_max:.4f}%")
print(f"Min Profit %: {btc_eth_min:.4f}%")
print(f"Std Dev: {btc_eth_std:.4f}%")

# Analyze by hour
hourly_eth_btc = eth_btc_path.groupby("hour")["profit_percentage"].mean()
hourly_btc_eth = btc_eth_path.groupby("hour")["profit_percentage"].mean()

print("\nHourly Analysis (CAD->ETH->BTC->CAD):")
for hour, profit in hourly_eth_btc.items():
    print(f"Hour {hour}: {profit:.4f}%")

# Find the best hour
best_hour = hourly_eth_btc.idxmax()
best_profit = hourly_eth_btc.max()
print(f"\nBest hour for CAD->ETH->BTC->CAD: Hour {best_hour} with average profit of {best_profit:.4f}%")

# Analyze price correlations
print("\nPrice Correlation Analysis:")
price_corr = df[["btc_cad_price", "eth_cad_price", "eth_btc_price", "profit_percentage"]].corr()
print(price_corr)

# Calculate how close we are to profitability
eth_btc_path["profit_gap"] = 0.25 - eth_btc_path["profit_percentage"]
closest_to_profit = eth_btc_path.nsmallest(5, "profit_gap")

print("\nClosest to Profitability (CAD->ETH->BTC->CAD):")
for idx, row in closest_to_profit.iterrows():
    print(f"Timestamp: {row['timestamp']}")
    print(f"Profit %: {row['profit_percentage']:.4f}%")
    print(f"Gap to 0.25%: {row['profit_gap']:.4f}%")
    print(f"BTC/CAD: {row['btc_cad_price']:.2f}, ETH/CAD: {row['eth_cad_price']:.2f}, ETH/BTC: {row['eth_btc_price']:.5f}")
    print("---")

# Save analysis results
analysis_dir = Path("data/analysis")
analysis_dir.mkdir(exist_ok=True)

analysis_results = {
    "timestamp": datetime.now().isoformat(),
    "total_opportunities": len(df),
    "eth_btc_opportunities": len(eth_btc_path),
    "btc_eth_opportunities": len(btc_eth_path),
    "eth_btc_stats": {
        "mean_profit_pct": float(eth_btc_mean),
        "max_profit_pct": float(eth_btc_max),
        "min_profit_pct": float(eth_btc_min),
        "std_profit_pct": float(eth_btc_std)
    },
    "btc_eth_stats": {
        "mean_profit_pct": float(btc_eth_mean),
        "max_profit_pct": float(btc_eth_max),
        "min_profit_pct": float(btc_eth_min),
        "std_profit_pct": float(btc_eth_std)
    },
    "best_hour": int(best_hour),
    "best_hour_profit": float(best_profit),
    "closest_to_profit": [
        {
            "timestamp": row["timestamp"].isoformat(),
            "profit_percentage": float(row["profit_percentage"]),
            "profit_gap": float(row["profit_gap"]),
            "btc_cad_price": float(row["btc_cad_price"]),
            "eth_cad_price": float(row["eth_cad_price"]),
            "eth_btc_price": float(row["eth_btc_price"])
        }
        for idx, row in closest_to_profit.iterrows()
    ]
}

with open(analysis_dir / "opportunity_analysis.json", "w") as f:
    json.dump(analysis_results, f, indent=2)

print("\nAnalysis results saved to data/analysis/opportunity_analysis.json")

# Generate recommendations
recommendations = []

if eth_btc_mean > btc_eth_mean:
    recommendations.append("Focus on the CAD->ETH->BTC->CAD path as it shows better performance")
else:
    recommendations.append("Focus on the CAD->BTC->ETH->CAD path as it shows better performance")

if best_profit > -1.0:
    recommendations.append(f"Consider scheduling trading during hour {best_hour} when profit is closest to positive")

if eth_btc_max > -1.0:
    recommendations.append("Lower profit threshold to 0.1% to catch opportunities when market conditions improve")

if eth_btc_std > 0.5:
    recommendations.append("High volatility detected - consider more frequent checks during optimal hours")
else:
    recommendations.append("Low volatility detected - consider longer intervals between checks to avoid rate limits")

print("\nRecommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

# Save recommendations
with open(analysis_dir / "recommendations.txt", "w") as f:
    f.write("# Trading Recommendations\n\n")
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec}\n")

print("\nRecommendations saved to data/analysis/recommendations.txt")
'@

# Pause to keep the window open
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")