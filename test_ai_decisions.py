#!/usr/bin/env python3
"""
Test AI Decision Making
Quick test to see what the AI is actually analyzing and why it's not trading
"""

import asyncio
import pandas as pd
from datetime import datetime
from exchange import ExchangeManager
from ai_strategy_optimizer import AIStrategyOptimizer
from advanced_technical_indicators import AdvancedTechnicalAnalyzer
from multi_coin_analyzer import MultiCoinAnalyzer

async def test_ai_decisions():
    """Test what the AI is actually seeing and deciding"""
    
    print("🔍 Testing AI Decision Making Process...")
    print("=" * 60)
    
    # Initialize components
    exchange = ExchangeManager()
    ai_optimizer = AIStrategyOptimizer()
    technical_analyzer = AdvancedTechnicalAnalyzer()
    multi_coin_analyzer = MultiCoinAnalyzer()
    
    # Test pairs
    pairs = ['ETH/CAD', 'BTC/CAD', 'SOL/CAD', 'XRP/CAD']
    
    for pair in pairs:
        print(f"\n📊 ANALYZING {pair}:")
        print("-" * 40)
        
        try:
            # Get current market data
            price_data = await multi_coin_analyzer.get_market_data(pair, '1h', 50)
            if price_data is None or price_data.empty or len(price_data) < 10:
                print(f"❌ No data available for {pair}")
                continue
            
            current_price = price_data['close'].iloc[-1]
            price_change = (current_price - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100
            
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Price Change:  {price_change:+.3f}%")
            
            # Get AI analysis
            ai_analysis = ai_optimizer.get_current_market_analysis(pair, price_data)
            ai_rec = ai_analysis.get('recommendation', {})
            ai_signal = ai_rec.get('signal', 'HOLD')
            ai_confidence = ai_rec.get('confidence', 0)
            
            print(f"   AI Signal:     {ai_signal}")
            print(f"   AI Confidence: {ai_confidence:.1f}%")
            
            # Get technical analysis
            technical_analysis = technical_analyzer.analyze_pair(price_data, pair)
            tech_signal = technical_analysis.get('composite_signal', {}).get('signal', 'HOLD')
            tech_strength = technical_analysis.get('composite_signal', {}).get('strength', 0)
            
            print(f"   Tech Signal:   {tech_signal}")
            print(f"   Tech Strength: {tech_strength:.1f}%")
            
            # Test decision criteria
            print(f"\n   DECISION ANALYSIS:")
            
            # Conservative criteria (current system)
            conservative_trade = False
            if ai_signal == 'BUY' and tech_signal == 'BUY':
                conservative_trade = ai_confidence > 60 and tech_strength > 40
                print(f"   Conservative (Both BUY): {conservative_trade} (needs AI>60% & Tech>40%)")
            elif ai_signal == 'SELL' and tech_signal == 'SELL':
                conservative_trade = ai_confidence > 60 and tech_strength > 40
                print(f"   Conservative (Both SELL): {conservative_trade} (needs AI>60% & Tech>40%)")
            elif ai_signal in ['BUY', 'SELL'] and ai_confidence > 80:
                conservative_trade = True
                print(f"   Conservative (Strong AI): {conservative_trade} (AI>80%)")
            elif tech_signal in ['BUY', 'SELL'] and tech_strength > 70:
                conservative_trade = True
                print(f"   Conservative (Strong Tech): {conservative_trade} (Tech>70%)")
            else:
                print(f"   Conservative: {conservative_trade} (no criteria met)")
            
            # Active criteria (proposed system)
            active_trade = False
            if ai_signal == 'BUY' and tech_signal == 'BUY':
                active_trade = ai_confidence > 45 and tech_strength > 30
                print(f"   Active (Both BUY): {active_trade} (needs AI>45% & Tech>30%)")
            elif ai_signal == 'SELL' and tech_signal == 'SELL':
                active_trade = ai_confidence > 45 and tech_strength > 30
                print(f"   Active (Both SELL): {active_trade} (needs AI>45% & Tech>30%)")
            elif ai_signal in ['BUY', 'SELL'] and ai_confidence > 65:
                active_trade = True
                print(f"   Active (Strong AI): {active_trade} (AI>65%)")
            elif tech_signal in ['BUY', 'SELL'] and tech_strength > 55:
                active_trade = True
                print(f"   Active (Strong Tech): {active_trade} (Tech>55%)")
            elif ai_signal == tech_signal and ai_signal in ['BUY', 'SELL']:
                active_trade = ai_confidence > 35 and tech_strength > 25
                print(f"   Active (Aligned): {active_trade} (AI>35% & Tech>25%)")
            else:
                print(f"   Active: {active_trade} (no criteria met)")
            
            print(f"   RESULT: Conservative={conservative_trade}, Active={active_trade}")
            
        except Exception as e:
            print(f"❌ Error analyzing {pair}: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("   If you see mostly 'False' results, the AI thresholds are too high")
    print("   If you see some 'True' results, the AI should be trading")
    print("   Consider using the Active Trading Simulation for more trades")

if __name__ == "__main__":
    asyncio.run(test_ai_decisions())