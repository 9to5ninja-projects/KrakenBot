import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from exchange import KrakenExchange
from arbitrage import ArbitrageCalculator
from monitor import ArbitrageMonitor
from position_manager import PositionManager
from notifications import NotificationManager

def test_path(base_currency, pairs, output_dir, amount=100, min_profit_pct=0.1, interval=45, duration=300):
    """Test a specific triangular arbitrage path."""
    print(f"Testing {base_currency} path with pairs: {', '.join(pairs)}")
    print(f"Results will be saved to: {output_dir}")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the exchange
    exchange = KrakenExchange()
    
    # Initialize the arbitrage calculator
    calculator = ArbitrageCalculator(
        start_amount=amount,
        min_profit_percentage=min_profit_pct,
        fee_percentage=0.5,
        trading_pairs=pairs
    )
    
    # Initialize the notification manager
    notification_manager = NotificationManager(method="console")
    
    # Initialize the position manager
    position_manager = PositionManager(
        max_capital=amount,
        position_size_percentage=2.0,
        max_positions=1
    )
    
    # Initialize the monitor
    monitor = ArbitrageMonitor(
        exchange=exchange,
        calculator=calculator,
        notification_manager=notification_manager,
        position_manager=position_manager,
        mode="simulate",
        check_interval=interval
    )
    
    # Run the monitor for a limited time
    opportunities = []
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < duration:
        # Check for arbitrage opportunities
        opportunity = monitor.check_for_arbitrage()
        if opportunity:
            opportunities.append(opportunity)
        
        # Wait for the next check
        import time
        time.sleep(interval)
    
    # Save the opportunities
    if opportunities:
        df = pd.DataFrame(opportunities)
        df.to_csv(os.path.join(output_dir, "opportunities.csv"), index=False)
        
        # Calculate statistics
        stats = {
            "total_opportunities": len(opportunities),
            "mean_profit": df["profit_percentage"].mean(),
            "max_profit": df["profit_percentage"].max(),
            "min_profit": df["profit_percentage"].min(),
            "std_profit": df["profit_percentage"].std()
        }
        
        with open(os.path.join(output_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Collected {len(opportunities)} opportunities")
        print(f"Mean profit: {stats['mean_profit']:.4f}%")
        print(f"Max profit: {stats['max_profit']:.4f}%")
    else:
        print("No opportunities found")
    
    return opportunities

def main():
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_dir = os.path.join("data", f"pair_test_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Test CAD path
    cad_pairs = ["ETH/CAD", "ETH/BTC", "BTC/CAD"]
    cad_dir = os.path.join(base_dir, "CAD")
    test_path("CAD", cad_pairs, cad_dir, duration=180)
    
    # Wait to avoid rate limits
    print("Waiting 2 minutes to avoid rate limits...")
    import time
    time.sleep(120)
    
    # Test USD path
    usd_pairs = ["ETH/USD", "ETH/BTC", "BTC/USD"]
    usd_dir = os.path.join(base_dir, "USD")
    test_path("USD", usd_pairs, usd_dir, duration=180)
    
    # Wait to avoid rate limits
    print("Waiting 2 minutes to avoid rate limits...")
    time.sleep(120)
    
    # Test EUR path
    eur_pairs = ["ETH/EUR", "ETH/BTC", "BTC/EUR"]
    eur_dir = os.path.join(base_dir, "EUR")
    test_path("EUR", eur_pairs, eur_dir, duration=180)
    
    # Analyze the results
    analyze_results(base_dir)

def analyze_results(base_dir):
    """Analyze the results from all currencies."""
    print("\nAnalyzing results from all currencies...")
    
    results = {}
    
    # Process each currency directory
    for currency_dir in os.listdir(base_dir):
        currency_path = os.path.join(base_dir, currency_dir)
        if not os.path.isdir(currency_path):
            continue
        
        base_currency = currency_dir
        print(f"\nAnalyzing {base_currency} paths:")
        
        # Load opportunities if they exist
        opps_file = os.path.join(currency_path, "opportunities.csv")
        if os.path.exists(opps_file):
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
                    "path_stats": path_stats
                }
                
                # Print summary
                print(f"  Collected {len(opps)} opportunities")
                
                # Show all paths by mean profit
                if path_stats:
                    sorted_paths = sorted(path_stats.items(), 
                                         key=lambda x: x[1]["mean_profit"], 
                                         reverse=True)
                    
                    print("\n  All paths by mean profit:")
                    for i, (path, stats) in enumerate(sorted_paths, 1):
                        print(f"  {i}. {path}: {stats['mean_profit']:.4f}% (max: {stats['max_profit']:.4f}%)")
                
            except Exception as e:
                print(f"  Error processing opportunities: {e}")
        else:
            print(f"  No opportunities data found")
    
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
    for i, path_data in enumerate(all_paths, 1):
        print(f"{i}. {path_data['base_currency']} - {path_data['path']}: {path_data['mean_profit']:.4f}% (max: {path_data['max_profit']:.4f}%)")
    
    # Save the comprehensive results
    with open(os.path.join(base_dir, "analysis.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "all_paths": all_paths,
            "detailed_results": results
        }, f, indent=2, default=str)
    
    # Generate recommendations
    recommendations = []
    
    if all_paths:
        # Best base currency and path
        best_path = all_paths[0]
        recommendations.append(f"Focus on the {best_path['path']} path with {best_path['base_currency']} as base")
        
        # Compare base currencies
        currencies = set(p["base_currency"] for p in all_paths)
        if len(currencies) > 1:
            currency_stats = {}
            for currency in currencies:
                currency_paths = [p for p in all_paths if p["base_currency"] == currency]
                if currency_paths:
                    currency_stats[currency] = {
                        "mean_profit": sum(p["mean_profit"] for p in currency_paths) / len(currency_paths),
                        "max_profit": max(p["max_profit"] for p in currency_paths)
                    }
            
            # Sort currencies by mean profit
            sorted_currencies = sorted(currency_stats.items(), key=lambda x: x[1]["mean_profit"], reverse=True)
            recommendations.append(f"{sorted_currencies[0][0]} is the best base currency for triangular arbitrage")
            
            # Compare with other currencies
            for i in range(1, len(sorted_currencies)):
                diff = sorted_currencies[0][1]["mean_profit"] - sorted_currencies[i][1]["mean_profit"]
                recommendations.append(f"{sorted_currencies[0][0]} outperforms {sorted_currencies[i][0]} by {diff:.4f}%")
    
    # Save recommendations
    with open(os.path.join(base_dir, "recommendations.txt"), "w") as f:
        f.write("# Optimal Trading Pair Recommendations\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\nAnalysis saved to {base_dir}/analysis.json")
    print(f"Recommendations saved to {base_dir}/recommendations.txt")

if __name__ == "__main__":
    main()