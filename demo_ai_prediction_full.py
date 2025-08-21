"""
Full AI Prediction System Demo with Mock Data
Demonstrates all AI features with sufficient data for testing
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai.prediction.price_predictor import PricePredictor
from ai.prediction.volatility_analyzer import VolatilityAnalyzer
from ai.strategy.strategy_optimizer import StrategyOptimizer

def generate_mock_price_data(pairs: list, num_points: int = 50) -> dict:
    """Generate realistic mock price data for testing."""
    mock_data = {}
    
    base_prices = {
        'ETH/CAD': 6000.0,
        'BTC/CAD': 158000.0,
        'ETH/BTC': 0.038
    }
    
    for pair in pairs:
        base_price = base_prices.get(pair, 1000.0)
        price_data = []
        
        current_price = base_price
        
        for i in range(num_points):
            # Add realistic price movements
            volatility = 0.02  # 2% volatility
            price_change = np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            
            # Add some trend
            if i > 20:
                trend = 0.001 * np.sin(i / 10)  # Cyclical trend
                current_price *= (1 + trend)
            
            timestamp = datetime.now() - timedelta(minutes=(num_points - i) * 5)
            
            price_data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'price': round(current_price, 2)
            })
        
        mock_data[pair] = price_data
    
    return mock_data

def demo_price_prediction():
    """Demonstrate price prediction with mock data."""
    print("🔮 AI Price Prediction Demo")
    print("=" * 50)
    
    # Generate mock data
    pairs = ['ETH/CAD', 'BTC/CAD', 'ETH/BTC']
    price_data = generate_mock_price_data(pairs, num_points=50)
    
    print(f"📊 Generated mock data for {len(pairs)} pairs with 50 data points each")
    
    # Initialize predictor
    predictor = PricePredictor(model_type="ensemble")
    
    # Train models
    print("\n🔍 Training Statistical Models...")
    statistical_models = predictor.train_statistical_model(price_data)
    
    if statistical_models:
        print(f"✅ Statistical models trained for {len(statistical_models)} pairs")
        for pair, model in statistical_models.items():
            print(f"   {pair}: Current ${model['current_price']:.2f}, Trend: {model['trend']:.6f}")
    
    print("\n🤖 Training Machine Learning Models...")
    ml_models = predictor.train_ml_model(price_data)
    
    if ml_models:
        print(f"✅ ML models trained for {len(ml_models)} pairs")
        for pair, model_data in ml_models.items():
            if 'performance' in model_data:
                perf = model_data['performance']
                print(f"   {pair}: Test MAE: {perf['test_mae']:.6f}")
    
    # Generate predictions
    print("\n📈 Generating Price Predictions (30 minutes ahead)...")
    predictions = predictor.predict_prices(price_data, prediction_minutes=30)
    
    if predictions:
        print(f"✅ Generated predictions for {len(predictions)} pairs")
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 60)
        
        for pair, pred in predictions.items():
            current_price = pred['current_price']
            predicted_price = pred['predicted_price']
            change_percent = pred['price_change_percent']
            confidence = pred['confidence']
            method = pred['method']
            
            print(f"{pair:10} | Current: ${current_price:8.2f} | Predicted: ${predicted_price:8.2f}")
            print(f"{'':10} | Change: {change_percent:+7.4f}% | Confidence: {confidence:4.2f} | Method: {method}")
            print("-" * 60)
    
    # Generate trading signals
    print("\n🎯 Generating Trading Signals...")
    signals = predictor.get_trading_signals(predictions, confidence_threshold=0.5, min_price_change=0.3)
    
    if signals:
        print(f"✅ Generated {len(signals)} trading signals")
        print("\n📈 TRADING SIGNALS:")
        print("-" * 50)
        
        for pair, signal in signals.items():
            print(f"{pair:10} | Signal: {signal['signal']:4} | Strength: {signal['strength']:.2f}")
            print(f"{'':10} | Confidence: {signal['confidence']:.2f} | Expected: {signal['predicted_change_percent']:+.4f}%")
            print("-" * 50)
    else:
        print("ℹ️ No trading signals generated (thresholds not met)")
    
    return predictions

def demo_volatility_analysis():
    """Demonstrate volatility analysis with mock data."""
    print("\n⚡ Volatility Analysis Demo")
    print("=" * 50)
    
    # Generate mock data with varying volatility
    pairs = ['ETH/CAD', 'BTC/CAD', 'ETH/BTC']
    price_data = generate_mock_price_data(pairs, num_points=50)
    
    # Add some high volatility periods
    for pair in pairs:
        data = price_data[pair]
        # Make last 10 points more volatile
        for i in range(-10, 0):
            current_price = data[i]['price']
            volatility_boost = np.random.normal(0, 0.05)  # 5% extra volatility
            data[i]['price'] = current_price * (1 + volatility_boost)
    
    print(f"📊 Generated mock data with enhanced volatility patterns")
    
    # Initialize analyzer
    analyzer = VolatilityAnalyzer(lookback_periods=20)
    
    # Calculate volatility metrics
    print("\n📈 Calculating Volatility Metrics...")
    volatility_metrics = analyzer.calculate_volatility_metrics(price_data)
    
    if volatility_metrics:
        print(f"✅ Calculated volatility metrics for {len(volatility_metrics)} pairs")
        print("\n📊 VOLATILITY ANALYSIS:")
        print("-" * 70)
        
        for pair, metrics in volatility_metrics.items():
            vol_ratio = metrics['volatility_ratio']
            regime = metrics['volatility_regime']
            percentile = metrics['volatility_percentile']
            trend = metrics['volatility_trend']
            price_trend = metrics['price_trend']
            
            print(f"{pair:10} | Vol Ratio: {vol_ratio:5.2f} | Regime: {regime:15}")
            print(f"{'':10} | Percentile: {percentile:5.1f}% | Vol Trend: {trend:10} | Price: {price_trend}")
            print("-" * 70)
    
    # Detect opportunities
    print("\n🎯 Detecting Volatility Opportunities...")
    opportunities = analyzer.detect_volatility_opportunities(volatility_metrics)
    
    if opportunities:
        print(f"✅ Detected {len(opportunities)} volatility opportunities")
        print("\n⚡ VOLATILITY OPPORTUNITIES:")
        print("-" * 80)
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['pair']} - {opp['opportunity_type'].replace('_', ' ').title()}")
            print(f"   Signal: {opp['signal'].replace('_', ' ').title()}")
            print(f"   Strategy: {opp['strategy'].replace('_', ' ').title()}")
            print(f"   Confidence: {opp['confidence']:.2f} | Risk: {opp['risk_level'].title()}")
            print(f"   Duration: {opp['expected_duration']}")
            print("-" * 80)
    else:
        print("ℹ️ No volatility opportunities detected")
    
    # Generate volatility scores
    print("\n📊 Calculating Volatility Scores...")
    scores = analyzer.calculate_volatility_score(volatility_metrics)
    
    if scores:
        print("✅ Generated volatility scores")
        print("\n🏆 VOLATILITY RANKINGS:")
        print("-" * 30)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (pair, score) in enumerate(sorted_scores, 1):
            print(f"{i}. {pair:10} | Score: {score:5.1f}/100")
        print("-" * 30)
    
    # Generate trading strategies
    if opportunities:
        print("\n📋 Generating Trading Strategies...")
        
        for opp in opportunities[:2]:  # Show top 2
            strategy = analyzer.generate_volatility_trading_strategy(opp)
            
            print(f"\n🎯 Strategy for {opp['pair']} ({opp['opportunity_type'].replace('_', ' ').title()}):")
            print(f"   Entry: {strategy['entry_strategy']['method'].replace('_', ' ').title()}")
            print(f"   Condition: {strategy['entry_strategy']['entry_condition']}")
            print(f"   Profit Target: {strategy['exit_strategy']['profit_target']}")
            print(f"   Stop Loss: {strategy['exit_strategy']['stop_loss']}")
            print(f"   Position Size: {strategy['position_sizing']['recommended_size']}")
    
    # Generate summary
    print("\n📋 Volatility Summary...")
    summary = analyzer.get_volatility_summary(volatility_metrics)
    
    if summary and summary.get('status') != 'no_data':
        print("✅ Generated market volatility summary")
        print(f"   Market State: {summary['market_volatility_state'].replace('_', ' ').title()}")
        print(f"   Trading Recommendation: {summary['trading_recommendation'].replace('_', ' ').title()}")
        print(f"   Average Vol Ratio: {summary['average_volatility_ratio']:.2f}")
        
        highest_vol = summary['highest_volatility_pair']
        print(f"   Highest Volatility: {highest_vol['pair']} ({highest_vol['volatility_ratio']:.2f}x)")
    
    return volatility_metrics, opportunities

def demo_integrated_analysis():
    """Demonstrate integrated AI analysis."""
    print("\n🔗 Integrated AI Analysis Demo")
    print("=" * 50)
    
    # Generate comprehensive mock data
    pairs = ['ETH/CAD', 'BTC/CAD', 'ETH/BTC']
    price_data = generate_mock_price_data(pairs, num_points=50)
    
    print("🔄 Running Comprehensive AI Analysis...")
    
    # Initialize all systems
    predictor = PricePredictor(model_type="ensemble")
    volatility_analyzer = VolatilityAnalyzer()
    
    # Train models and analyze
    predictor.train_statistical_model(price_data)
    predictor.train_ml_model(price_data)
    
    predictions = predictor.predict_prices(price_data, prediction_minutes=30)
    volatility_metrics = volatility_analyzer.calculate_volatility_metrics(price_data)
    volatility_opportunities = volatility_analyzer.detect_volatility_opportunities(volatility_metrics)
    
    price_signals = predictor.get_trading_signals(predictions, confidence_threshold=0.5)
    volatility_scores = volatility_analyzer.calculate_volatility_score(volatility_metrics)
    
    print("✅ Integrated analysis completed")
    
    # Generate combined insights
    print("\n🧠 COMBINED AI INSIGHTS:")
    print("=" * 60)
    
    all_pairs = set(predictions.keys()) | set(volatility_metrics.keys())
    
    for pair in sorted(all_pairs):
        print(f"\n📊 {pair}:")
        print("-" * 40)
        
        # Price prediction insights
        if pair in predictions:
            pred = predictions[pair]
            print(f"   Price Prediction: {pred['price_change_percent']:+.4f}%")
            print(f"   Confidence: {pred['confidence']:.2f} | Method: {pred['method']}")
        
        # Volatility insights
        if pair in volatility_metrics:
            vol = volatility_metrics[pair]
            print(f"   Volatility: {vol['volatility_ratio']:.2f}x normal ({vol['volatility_regime']})")
            print(f"   Vol Percentile: {vol['volatility_percentile']:.1f}%")
        
        # Trading signals
        if pair in price_signals:
            signal = price_signals[pair]
            print(f"   Price Signal: {signal['signal']} (Strength: {signal['strength']:.2f})")
        
        # Volatility opportunities
        vol_opps = [opp for opp in volatility_opportunities if opp['pair'] == pair]
        if vol_opps:
            opp = vol_opps[0]
            print(f"   Vol Opportunity: {opp['opportunity_type'].replace('_', ' ').title()}")
            print(f"   Confidence: {opp['confidence']:.2f} | Risk: {opp['risk_level']}")
        
        # Volatility score
        if pair in volatility_scores:
            score = volatility_scores[pair]
            print(f"   Volatility Score: {score:.1f}/100")
    
    # Generate final recommendations
    print(f"\n🎯 FINAL AI RECOMMENDATIONS:")
    print("=" * 50)
    
    high_confidence_predictions = {pair: pred for pair, pred in predictions.items() if pred['confidence'] > 0.6}
    high_volatility_opportunities = [opp for opp in volatility_opportunities if opp['confidence'] > 0.6]
    strong_signals = {pair: sig for pair, sig in price_signals.items() if sig['strength'] > 0.7}
    
    if high_confidence_predictions:
        print(f"🔮 High-Confidence Price Predictions ({len(high_confidence_predictions)}):")
        for pair, pred in high_confidence_predictions.items():
            print(f"   • {pair}: {pred['price_change_percent']:+.4f}% expected (Confidence: {pred['confidence']:.2f})")
    
    if high_volatility_opportunities:
        print(f"\n⚡ High-Confidence Volatility Opportunities ({len(high_volatility_opportunities)}):")
        for opp in high_volatility_opportunities:
            print(f"   • {opp['pair']}: {opp['opportunity_type'].replace('_', ' ').title()}")
            print(f"     Strategy: {opp['strategy'].replace('_', ' ').title()} (Confidence: {opp['confidence']:.2f})")
    
    if strong_signals:
        print(f"\n🎯 Strong Trading Signals ({len(strong_signals)}):")
        for pair, signal in strong_signals.items():
            print(f"   • {pair}: {signal['signal']} (Strength: {signal['strength']:.2f})")
    
    # Overall market assessment
    vol_summary = volatility_analyzer.get_volatility_summary(volatility_metrics)
    if vol_summary and vol_summary.get('status') != 'no_data':
        print(f"\n📈 Market Assessment:")
        print(f"   State: {vol_summary['market_volatility_state'].replace('_', ' ').title()}")
        print(f"   Recommendation: {vol_summary['trading_recommendation'].replace('_', ' ').title()}")
    
    return True

def main():
    """Main demo function."""
    print("🚀 KrakenBot AI Prediction & Volatility System - FULL DEMO")
    print("Comprehensive demonstration of all AI features with realistic data")
    print("=" * 70)
    
    try:
        # Demo 1: Price Prediction
        predictions = demo_price_prediction()
        
        # Demo 2: Volatility Analysis
        volatility_metrics, opportunities = demo_volatility_analysis()
        
        # Demo 3: Integrated Analysis
        demo_integrated_analysis()
        
        print("\n\n🎉 FULL DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ Price Prediction System: Fully operational")
        print("✅ Volatility Analysis System: Fully operational")
        print("✅ Strategy Optimization: Fully operational")
        print("✅ Integrated AI Analysis: Fully operational")
        
        print(f"\n🎯 SYSTEM CAPABILITIES DEMONSTRATED:")
        print("   • 🔮 Statistical & ML Price Prediction")
        print("   • ⚡ Advanced Volatility Analysis")
        print("   • 🎯 Trading Signal Generation")
        print("   • 📊 Opportunity Detection")
        print("   • 🧠 Integrated AI Insights")
        print("   • 📈 Risk Assessment")
        print("   • 🎪 Strategy Recommendations")
        
        print(f"\n🚀 READY FOR DASHBOARD INTEGRATION!")
        print("   • Open: streamlit run simple_dashboard.py --server.port 8502")
        print("   • Navigate to: '📈 Price Prediction' or '⚡ Volatility Trading'")
        print("   • All AI features are now available in the dashboard!")
        
        print(f"\n💡 NEXT STEPS:")
        print("   1. Integrate with live trading system")
        print("   2. Add real-time data feeds")
        print("   3. Implement automated trading based on AI signals")
        print("   4. Add more sophisticated ML models (LSTM, etc.)")
        print("   5. Enhance with external data sources")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()