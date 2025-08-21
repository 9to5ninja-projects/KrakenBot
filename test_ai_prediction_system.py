"""
Comprehensive Test Script for AI Prediction and Volatility System
Tests price prediction, volatility analysis, and strategy optimization
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai.prediction.price_predictor import PricePredictor
from ai.prediction.volatility_analyzer import VolatilityAnalyzer
from ai.strategy.strategy_optimizer import StrategyOptimizer
import json

def test_price_prediction():
    """Test the price prediction system."""
    print("🔮 Testing AI Price Prediction System")
    print("=" * 50)
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found. Please run a trading session first.")
        return False
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using session: {latest_session.name}")
    
    # Load price data
    price_file = latest_session / "price_history.json"
    if not price_file.exists():
        print("❌ No price history file found")
        return False
    
    with open(price_file, 'r') as f:
        price_data = json.load(f)
    
    print(f"📊 Loaded price data for {len(price_data)} pairs")
    
    # Initialize predictor
    predictor = PricePredictor(model_type="ensemble")
    
    # Test statistical model training
    print("\n🔍 Training Statistical Models...")
    statistical_models = predictor.train_statistical_model(price_data)
    
    if statistical_models:
        print(f"✅ Statistical models trained for {len(statistical_models)} pairs")
        for pair, model in statistical_models.items():
            print(f"   {pair}: Current price ${model['current_price']:.2f}, Trend: {model['trend']:.6f}")
    else:
        print("❌ Failed to train statistical models")
        return False
    
    # Test ML model training
    print("\n🤖 Training Machine Learning Models...")
    ml_models = predictor.train_ml_model(price_data)
    
    if ml_models:
        print(f"✅ ML models trained for {len(ml_models)} pairs")
        for pair, model_data in ml_models.items():
            if 'performance' in model_data:
                perf = model_data['performance']
                print(f"   {pair}: Test MAE: {perf['test_mae']:.6f}, Samples: {perf['test_samples']}")
    else:
        print("⚠️ ML models not available (using statistical models)")
    
    # Test price predictions
    print("\n📈 Generating Price Predictions...")
    predictions = predictor.predict_prices(price_data, prediction_minutes=30)
    
    if predictions:
        print(f"✅ Generated predictions for {len(predictions)} pairs")
        
        for pair, pred in predictions.items():
            current_price = pred['current_price']
            predicted_price = pred['predicted_price']
            change_percent = pred['price_change_percent']
            confidence = pred['confidence']
            method = pred['method']
            
            print(f"   {pair}:")
            print(f"     Current: ${current_price:.2f}")
            print(f"     Predicted: ${predicted_price:.2f}")
            print(f"     Change: {change_percent:+.4f}%")
            print(f"     Confidence: {confidence:.2f}")
            print(f"     Method: {method}")
    else:
        print("❌ Failed to generate predictions")
        return False
    
    # Test trading signals
    print("\n🎯 Generating Trading Signals...")
    signals = predictor.get_trading_signals(predictions, confidence_threshold=0.6, min_price_change=0.5)
    
    if signals:
        print(f"✅ Generated {len(signals)} trading signals")
        
        for pair, signal in signals.items():
            print(f"   {pair}: {signal['signal']} (Strength: {signal['strength']:.2f}, Confidence: {signal['confidence']:.2f})")
    else:
        print("ℹ️ No trading signals generated (thresholds not met)")
    
    print("\n✅ Price Prediction System Test Completed!")
    return True

def test_volatility_analysis():
    """Test the volatility analysis system."""
    print("\n⚡ Testing Volatility Analysis System")
    print("=" * 50)
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found.")
        return False
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    
    # Load price data
    price_file = latest_session / "price_history.json"
    with open(price_file, 'r') as f:
        price_data = json.load(f)
    
    # Initialize volatility analyzer
    analyzer = VolatilityAnalyzer(lookback_periods=20)
    
    # Test volatility metrics calculation
    print("📊 Calculating Volatility Metrics...")
    volatility_metrics = analyzer.calculate_volatility_metrics(price_data)
    
    if volatility_metrics:
        print(f"✅ Calculated volatility metrics for {len(volatility_metrics)} pairs")
        
        for pair, metrics in volatility_metrics.items():
            vol_ratio = metrics['volatility_ratio']
            regime = metrics['volatility_regime']
            percentile = metrics['volatility_percentile']
            trend = metrics['volatility_trend']
            
            print(f"   {pair}:")
            print(f"     Volatility Ratio: {vol_ratio:.2f}")
            print(f"     Regime: {regime}")
            print(f"     Percentile: {percentile:.1f}%")
            print(f"     Trend: {trend}")
    else:
        print("❌ Failed to calculate volatility metrics")
        return False
    
    # Test opportunity detection
    print("\n🎯 Detecting Volatility Opportunities...")
    opportunities = analyzer.detect_volatility_opportunities(volatility_metrics)
    
    if opportunities:
        print(f"✅ Detected {len(opportunities)} volatility opportunities")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"   {i}. {opp['pair']} - {opp['opportunity_type']}")
            print(f"      Signal: {opp['signal']}")
            print(f"      Strategy: {opp['strategy']}")
            print(f"      Confidence: {opp['confidence']:.2f}")
            print(f"      Risk Level: {opp['risk_level']}")
            print(f"      Duration: {opp['expected_duration']}")
    else:
        print("ℹ️ No volatility opportunities detected")
    
    # Test volatility scores
    print("\n📈 Calculating Volatility Scores...")
    scores = analyzer.calculate_volatility_score(volatility_metrics)
    
    if scores:
        print(f"✅ Calculated volatility scores for {len(scores)} pairs")
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for pair, score in sorted_scores:
            print(f"   {pair}: {score:.1f}/100")
    
    # Test volatility summary
    print("\n📋 Generating Volatility Summary...")
    summary = analyzer.get_volatility_summary(volatility_metrics)
    
    if summary and summary.get('status') != 'no_data':
        print("✅ Generated volatility summary")
        print(f"   Pairs Analyzed: {summary['total_pairs_analyzed']}")
        print(f"   Average Vol Ratio: {summary['average_volatility_ratio']:.2f}")
        print(f"   Market State: {summary['market_volatility_state']}")
        print(f"   Trading Recommendation: {summary['trading_recommendation']}")
        
        highest_vol = summary['highest_volatility_pair']
        print(f"   Highest Volatility: {highest_vol['pair']} ({highest_vol['volatility_ratio']:.2f})")
    
    # Test trading strategy generation
    if opportunities:
        print("\n📋 Generating Trading Strategies...")
        
        for opp in opportunities[:2]:  # Test top 2 opportunities
            strategy = analyzer.generate_volatility_trading_strategy(opp)
            
            print(f"   Strategy for {opp['pair']} ({opp['opportunity_type']}):")
            print(f"     Entry Method: {strategy['entry_strategy']['method']}")
            print(f"     Profit Target: {strategy['exit_strategy']['profit_target']}")
            print(f"     Stop Loss: {strategy['exit_strategy']['stop_loss']}")
            print(f"     Position Size: {strategy['position_sizing']['recommended_size']}")
    
    print("\n✅ Volatility Analysis System Test Completed!")
    return True

def test_strategy_optimization():
    """Test the strategy optimization system."""
    print("\n🎯 Testing Strategy Optimization System")
    print("=" * 50)
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found.")
        return False
    
    # Use the most recent session
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    
    # Load session data
    session_data = {}
    
    files_to_load = {
        'metadata': 'session_metadata.json',
        'portfolio_history': 'portfolio_history.json',
        'price_history': 'price_history.json',
        'trades': 'trades.json',
        'performance': 'performance_summary.json'
    }
    
    for key, filename in files_to_load.items():
        file_path = latest_session / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                session_data[key] = json.load(f)
        else:
            session_data[key] = {} if key in ['metadata', 'performance'] else []
    
    print(f"📁 Loaded session data from {latest_session.name}")
    
    # Initialize optimizer
    optimizer = StrategyOptimizer()
    
    # Test performance analysis
    print("\n📊 Analyzing Current Performance...")
    analysis = optimizer.analyze_current_performance(session_data)
    
    if analysis:
        print("✅ Performance analysis completed")
        
        # Display session info
        session_info = analysis['session_info']
        print(f"   Session ID: {session_info['session_id']}")
        print(f"   Duration: {session_info['duration_hours']:.2f} hours")
        print(f"   Strategy: {session_info['strategy_type']}")
        print(f"   Trading Pairs: {', '.join(session_info['trading_pairs'])}")
        
        # Display performance metrics
        perf = analysis['performance_metrics']
        print(f"   Total Return: {perf['total_return_percent']:.4f}%")
        print(f"   Success Rate: {perf['success_rate_percent']:.2f}%")
        print(f"   Total Trades: {perf['total_trades']}")
        print(f"   Max Drawdown: {perf['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        
        # Display parameter effectiveness
        param_eff = analysis['parameter_effectiveness']
        if 'thresholds' in param_eff:
            thresholds = param_eff['thresholds']
            buy_eff = thresholds['buy_threshold']['effectiveness_score']
            sell_eff = thresholds['sell_threshold']['effectiveness_score']
            print(f"   Buy Threshold Effectiveness: {buy_eff:.2f}")
            print(f"   Sell Threshold Effectiveness: {sell_eff:.2f}")
        
        # Display improvement opportunities
        opportunities = analysis['improvement_opportunities']
        if opportunities:
            print(f"\n🎯 Improvement Opportunities ({len(opportunities)} found):")
            for opp in opportunities:
                print(f"   • {opp['type']} ({opp['priority']} priority)")
                print(f"     {opp['description']}")
        
        # Display optimization recommendations
        recommendations = analysis['optimization_recommendations']
        if recommendations:
            print(f"\n💡 Optimization Recommendations ({len(recommendations)} found):")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   • {rec['parameter']} ({rec['priority']} priority)")
                print(f"     Current: {rec['current_value']}")
                print(f"     Recommended: {rec['recommended_value']}")
                print(f"     Change: {rec['change_percent']:+.1f}%")
                print(f"     Rationale: {rec['rationale']}")
        
        # Test parameter optimization
        print("\n⚙️ Generating Optimized Parameters...")
        current_params = session_data.get('metadata', {}).get('parameters', {})
        
        optimized = optimizer.generate_optimized_parameters(current_params, recommendations)
        
        if optimized:
            print("✅ Generated optimized parameter set")
            
            applied_changes = optimized['applied_changes']
            print(f"   Applied {len(applied_changes)} parameter changes:")
            
            for change in applied_changes:
                print(f"   • {change['parameter']}: {change['old_value']} → {change['new_value']}")
                print(f"     Change: {change['change_percent']:+.1f}%")
                print(f"     Rationale: {change['rationale']}")
            
            # Save optimized parameters
            opt_file = latest_session / "test_optimized_parameters.json"
            optimizer.save_optimization_results(optimized, str(opt_file))
            print(f"   Saved optimized parameters to {opt_file}")
    else:
        print("❌ Failed to analyze performance")
        return False
    
    print("\n✅ Strategy Optimization System Test Completed!")
    return True

def test_integration():
    """Test integration between all AI systems."""
    print("\n🔗 Testing AI System Integration")
    print("=" * 50)
    
    # Find the current live session
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        print("❌ No live sessions found.")
        return False
    
    latest_session = max(live_sessions, key=lambda x: x.stat().st_mtime)
    
    # Load price data
    price_file = latest_session / "price_history.json"
    with open(price_file, 'r') as f:
        price_data = json.load(f)
    
    print("🔄 Running Integrated AI Analysis...")
    
    # Initialize all systems
    predictor = PricePredictor(model_type="ensemble")
    volatility_analyzer = VolatilityAnalyzer()
    
    # Train prediction models
    predictor.train_statistical_model(price_data)
    predictor.train_ml_model(price_data)
    
    # Generate predictions
    predictions = predictor.predict_prices(price_data, prediction_minutes=30)
    
    # Analyze volatility
    volatility_metrics = volatility_analyzer.calculate_volatility_metrics(price_data)
    volatility_opportunities = volatility_analyzer.detect_volatility_opportunities(volatility_metrics)
    
    # Generate trading signals
    price_signals = predictor.get_trading_signals(predictions, confidence_threshold=0.6)
    
    print("✅ Integrated analysis completed")
    
    # Combine insights
    print("\n🧠 Combined AI Insights:")
    
    all_pairs = set(predictions.keys()) | set(volatility_metrics.keys())
    
    for pair in all_pairs:
        print(f"\n   {pair}:")
        
        # Price prediction insights
        if pair in predictions:
            pred = predictions[pair]
            print(f"     Price Prediction: {pred['price_change_percent']:+.4f}% (Confidence: {pred['confidence']:.2f})")
        
        # Volatility insights
        if pair in volatility_metrics:
            vol = volatility_metrics[pair]
            print(f"     Volatility: {vol['volatility_ratio']:.2f}x normal ({vol['volatility_regime']})")
        
        # Trading signals
        if pair in price_signals:
            signal = price_signals[pair]
            print(f"     Signal: {signal['signal']} (Strength: {signal['strength']:.2f})")
        
        # Volatility opportunities
        vol_opps = [opp for opp in volatility_opportunities if opp['pair'] == pair]
        if vol_opps:
            opp = vol_opps[0]
            print(f"     Volatility Opportunity: {opp['opportunity_type']} ({opp['confidence']:.2f})")
    
    # Generate combined recommendations
    print(f"\n🎯 Combined Trading Recommendations:")
    
    high_confidence_predictions = {pair: pred for pair, pred in predictions.items() if pred['confidence'] > 0.7}
    high_volatility_opportunities = [opp for opp in volatility_opportunities if opp['confidence'] > 0.7]
    
    if high_confidence_predictions:
        print(f"   High-Confidence Price Predictions ({len(high_confidence_predictions)}):")
        for pair, pred in high_confidence_predictions.items():
            print(f"     • {pair}: {pred['price_change_percent']:+.4f}% expected")
    
    if high_volatility_opportunities:
        print(f"   High-Confidence Volatility Opportunities ({len(high_volatility_opportunities)}):")
        for opp in high_volatility_opportunities:
            print(f"     • {opp['pair']}: {opp['opportunity_type']} ({opp['strategy']})")
    
    print("\n✅ AI System Integration Test Completed!")
    return True

def main():
    """Main test function."""
    print("🚀 KrakenBot AI Prediction & Volatility System Test Suite")
    print("Testing advanced AI features for price prediction and volatility trading")
    print("=" * 70)
    
    test_results = []
    
    try:
        # Test 1: Price Prediction System
        result1 = test_price_prediction()
        test_results.append(("Price Prediction", result1))
        
        # Test 2: Volatility Analysis System
        result2 = test_volatility_analysis()
        test_results.append(("Volatility Analysis", result2))
        
        # Test 3: Strategy Optimization System
        result3 = test_strategy_optimization()
        test_results.append(("Strategy Optimization", result3))
        
        # Test 4: System Integration
        result4 = test_integration()
        test_results.append(("System Integration", result4))
        
        # Summary
        print("\n\n🎉 TEST SUITE COMPLETED!")
        print("=" * 70)
        
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        
        print(f"📊 Results: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        if passed_tests == total_tests:
            print(f"\n🎯 ALL SYSTEMS OPERATIONAL!")
            print("The AI prediction and volatility system is ready for production use!")
            
            print(f"\n📋 Available Features:")
            print("   • 🔮 Price Prediction (Statistical + ML)")
            print("   • ⚡ Volatility Analysis & Opportunity Detection")
            print("   • 🎯 Strategy Optimization & Parameter Tuning")
            print("   • 📊 Integrated AI Trading Signals")
            print("   • 📈 Dashboard Integration")
            
            print(f"\n🚀 Next Steps:")
            print("   1. Open dashboard: streamlit run simple_dashboard.py --server.port 8502")
            print("   2. Navigate to '📈 Price Prediction' or '⚡ Volatility Trading' tabs")
            print("   3. Use '🎯 Strategy Optimization' for parameter tuning")
            print("   4. Integrate with live trading system")
        else:
            print(f"\n⚠️ Some tests failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test suite error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()