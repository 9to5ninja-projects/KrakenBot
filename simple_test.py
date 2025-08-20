"""
Simple script to test specific trading pairs for triangular arbitrage.
"""
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# Import the necessary modules
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator

def test_path(base_currency, pairs, amount=100):
    """Test a specific triangular arbitrage path."""
    print(f"Testing {base_currency} path with pairs: {', '.join(pairs)}")
    
    # Initialize the exchange
    exchange = ExchangeManager(use_api_keys=False)
    
    # Initialize the arbitrage calculator
    calculator = ArbitrageCalculator(exchange)
    
    # Get current prices
    prices = {}
    for pair in pairs:
        try:
            ticker = exchange.fetch_ticker(pair)
            if ticker:
                prices[pair] = ticker['last']  # Use last price instead of bid/ask
                print(f"{pair}: Price: {ticker['last']}")
            else:
                print(f"Could not get ticker for {pair}")
                return None
        except Exception as e:
            print(f"Error getting ticker for {pair}: {e}")
            return None
    
    # Create the path from base currency and pairs
    if base_currency == "CAD":
        if "ETH/CAD" in pairs and "ETH/BTC" in pairs and "BTC/CAD" in pairs:
            path = ['CAD', 'ETH', 'BTC', 'CAD']
        else:
            path = ['CAD', 'BTC', 'ETH', 'CAD']
    elif base_currency == "USD":
        if "ETH/USD" in pairs and "ETH/BTC" in pairs and "BTC/USD" in pairs:
            path = ['USD', 'ETH', 'BTC', 'USD']
        else:
            path = ['USD', 'BTC', 'ETH', 'USD']
    elif base_currency == "EUR":
        if "ETH/EUR" in pairs and "ETH/BTC" in pairs and "BTC/EUR" in pairs:
            path = ['EUR', 'ETH', 'BTC', 'EUR']
        else:
            path = ['EUR', 'BTC', 'ETH', 'EUR']
    else:
        path = [base_currency, 'BTC', 'ETH', base_currency]
    
    # Calculate potential profit using custom path
    result = calculator.calculate_custom_path(path, prices)
    
    if result and result.is_profitable:
        print(f"Path: {' → '.join(result.path)}")
        print(f"Profit: {result.profit_percentage:.4f}%")
        print(f"Final amount: {result.end_amount:.2f} {base_currency}")
        return {
            "path": ' → '.join(result.path),
            "profit_percentage": result.profit_percentage,
            "final_amount": result.end_amount
        }
    else:
        if result:
            print(f"Path: {' → '.join(result.path)}")
            print(f"Profit: {result.profit_percentage:.4f}% (Not profitable)")
            print(f"Final amount: {result.end_amount:.2f} {base_currency}")
        print("No profitable arbitrage opportunity found")
        return None

def main():
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_dir = os.path.join("data", f"simple_test_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    
    results = []
    
    # Test CAD path
    print("\n=== Testing CAD->ETH->BTC->CAD ===")
    cad_pairs = ["ETH/CAD", "ETH/BTC", "BTC/CAD"]
    cad_result = test_path("CAD", cad_pairs)
    if cad_result:
        results.append({
            "base_currency": "CAD",
            "path": cad_result["path"],
            "profit_percentage": cad_result["profit_percentage"],
            "final_amount": cad_result["final_amount"]
        })
    
    # Wait to avoid rate limits
    print("\nWaiting 30 seconds to avoid rate limits...")
    time.sleep(30)
    
    # Test USD path
    print("\n=== Testing USD->ETH->BTC->USD ===")
    usd_pairs = ["ETH/USD", "ETH/BTC", "BTC/USD"]
    usd_result = test_path("USD", usd_pairs)
    if usd_result:
        results.append({
            "base_currency": "USD",
            "path": usd_result["path"],
            "profit_percentage": usd_result["profit_percentage"],
            "final_amount": usd_result["final_amount"]
        })
    
    # Wait to avoid rate limits
    print("\nWaiting 30 seconds to avoid rate limits...")
    time.sleep(30)
    
    # Test EUR path
    print("\n=== Testing EUR->ETH->BTC->EUR ===")
    eur_pairs = ["ETH/EUR", "ETH/BTC", "BTC/EUR"]
    eur_result = test_path("EUR", eur_pairs)
    if eur_result:
        results.append({
            "base_currency": "EUR",
            "path": eur_result["path"],
            "profit_percentage": eur_result["profit_percentage"],
            "final_amount": eur_result["final_amount"]
        })
    
    # Save the results
    if results:
        with open(os.path.join(base_dir, "results.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        # Sort by profit percentage
        results.sort(key=lambda x: x["profit_percentage"], reverse=True)
        
        print("\n=== RESULTS SUMMARY ===")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['base_currency']} - {result['path']}: {result['profit_percentage']:.4f}%")
        
        # Generate recommendations
        recommendations = []
        
        if results:
            # Best path
            best_result = results[0]
            recommendations.append(f"Focus on the {best_result['path']} path with {best_result['base_currency']} as base")
            
            # Compare with other paths
            for i in range(1, len(results)):
                diff = best_result["profit_percentage"] - results[i]["profit_percentage"]
                recommendations.append(f"{best_result['base_currency']} outperforms {results[i]['base_currency']} by {diff:.4f}%")
        
        # Save recommendations
        with open(os.path.join(base_dir, "recommendations.txt"), "w") as f:
            f.write("# Optimal Trading Pair Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\nResults saved to {base_dir}/results.json")
        print(f"Recommendations saved to {base_dir}/recommendations.txt")
    else:
        print("\nNo valid results to save")

if __name__ == "__main__":
    main()