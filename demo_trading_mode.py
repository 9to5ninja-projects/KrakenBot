#!/usr/bin/env python3
"""
Demo Trading Mode
Guaranteed to show AI trading activity for demonstration purposes
Uses very aggressive parameters and enhanced signal detection
"""

import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from enhanced_simulation_system import EnhancedSimulationSystem, StrategyParameters

class DemoTradingSimulation(EnhancedSimulationSystem):
    """Demo simulation guaranteed to show trading activity"""
    
    def __init__(self, initial_capital: float = 1000.0, session_name: str = None):
        super().__init__(initial_capital, session_name)
        
        # Ultra-aggressive demo parameters
        self.strategy_params = StrategyParameters(
            buy_threshold=-0.0005,  # -0.05% (ultra sensitive)
            sell_threshold=0.0005,  # +0.05% (ultra sensitive)
            lookback_periods=2,     # Very short lookback
            min_trade_amount=20.0,  # Lower minimum trade
            max_position_size=0.12, # 12% max per position
            stop_loss_pct=-0.01,    # -1% stop loss
            take_profit_pct=0.008,  # +0.8% take profit
            volatility_threshold=0.01,   # Very low volatility threshold
            volume_threshold=1.1,   # Very low volume threshold
            rsi_oversold=40,        # Less extreme RSI
            rsi_overbought=60,      # Less extreme RSI
            bollinger_buy_position=0.4,   # Less extreme Bollinger
            bollinger_sell_position=0.6   # Less extreme Bollinger
        )
        
        print(f"🎬 DEMO TRADING MODE ENABLED")
        print(f"   Ultra-aggressive parameters for guaranteed activity")
        print(f"   Buy Threshold: {self.strategy_params.buy_threshold:.4f} ({self.strategy_params.buy_threshold*100:.2f}%)")
        print(f"   Sell Threshold: {self.strategy_params.sell_threshold:.4f} ({self.strategy_params.sell_threshold*100:.2f}%)")
        print(f"   This WILL trade frequently!")
    
    def _should_create_opportunity(self, ai_signal: str, tech_signal: str, 
                                 ai_confidence: float, tech_strength: float) -> bool:
        """Ultra-aggressive opportunity creation for demo purposes"""
        
        # Demo mode: Much lower thresholds
        if ai_signal == 'BUY' and tech_signal == 'BUY':
            return ai_confidence > 35 and tech_strength > 15  # Very low thresholds
        elif ai_signal == 'SELL' and tech_signal == 'SELL':
            return ai_confidence > 35 and tech_strength > 15  # Very low thresholds
        
        # Allow single signals with very low confidence
        elif ai_signal in ['BUY', 'SELL'] and ai_confidence > 50:  # Lowered from 65
            return True
        elif tech_signal in ['BUY', 'SELL'] and tech_strength > 25:  # Lowered from 55
            return True
        
        # Allow weak signals if they align
        elif ai_signal == tech_signal and ai_signal in ['BUY', 'SELL']:
            return ai_confidence > 25 and tech_strength > 10  # Very low threshold
        
        # Demo enhancement: Create opportunities even with HOLD signals if there's any movement
        elif ai_confidence > 45 or tech_strength > 20:
            # Randomly convert some HOLD signals to BUY/SELL for demo
            if random.random() < 0.3:  # 30% chance
                return True
        
        return False
    
    def _enhance_signals_for_demo(self, ai_signal: str, tech_signal: str, 
                                ai_confidence: float, tech_strength: float):
        """Enhance signals for demo purposes"""
        
        # If both are HOLD but have some strength, randomly make them directional
        if ai_signal == 'HOLD' and tech_signal == 'HOLD':
            if ai_confidence > 45 or tech_strength > 15:
                # Randomly assign direction based on slight market bias
                if random.random() < 0.5:
                    ai_signal = 'BUY'
                    ai_confidence = min(ai_confidence + 15, 85)  # Boost confidence
                else:
                    ai_signal = 'SELL'
                    ai_confidence = min(ai_confidence + 15, 85)  # Boost confidence
                
                # Also boost technical signal
                if random.random() < 0.7:  # 70% chance to align
                    tech_signal = ai_signal
                    tech_strength = min(tech_strength + 20, 75)  # Boost strength
        
        return ai_signal, tech_signal, ai_confidence, tech_strength
    
    async def _analyze_trading_opportunities(self):
        """Enhanced opportunity analysis for demo mode"""
        opportunities = []
        
        for pair in self.trading_pairs:
            try:
                # Get market data
                price_data = await self.multi_coin_analyzer.get_market_data(pair, '1h', 100)
                
                if price_data.empty:
                    continue
                
                # Get AI analysis
                ai_analysis = self.ai_optimizer.get_current_market_analysis(pair, price_data)
                ai_rec = ai_analysis.get('recommendation', {})
                ai_signal = ai_rec.get('signal', 'HOLD')
                ai_confidence = ai_rec.get('confidence', 50.0)
                
                # Get technical analysis
                technical_analysis = self.technical_analyzer.analyze_pair(price_data, pair)
                tech_signal = technical_analysis.get('composite_signal', {}).get('signal', 'HOLD')
                tech_strength = technical_analysis.get('composite_signal', {}).get('strength', 0.0)
                
                # DEMO ENHANCEMENT: Boost weak signals
                ai_signal, tech_signal, ai_confidence, tech_strength = self._enhance_signals_for_demo(
                    ai_signal, tech_signal, ai_confidence, tech_strength
                )
                
                # Check if we should create an opportunity
                if self._should_create_opportunity(ai_signal, tech_signal, ai_confidence, tech_strength):
                    
                    # Determine final signal (prefer agreement, but allow single strong signals)
                    final_signal = 'HOLD'
                    final_confidence = 0.0
                    
                    if ai_signal == tech_signal and ai_signal in ['BUY', 'SELL']:
                        final_signal = ai_signal
                        final_confidence = (ai_confidence + tech_strength) / 2
                    elif ai_confidence > 60:
                        final_signal = ai_signal
                        final_confidence = ai_confidence * 0.8  # Slight discount for single signal
                    elif tech_strength > 40:
                        final_signal = tech_signal
                        final_confidence = tech_strength * 0.8  # Slight discount for single signal
                    
                    if final_signal in ['BUY', 'SELL']:
                        opportunity = {
                            'timestamp': datetime.now().isoformat(),
                            'pair': pair,
                            'signal': final_signal,
                            'confidence': final_confidence,
                            'ai_signal': ai_signal,
                            'ai_confidence': ai_confidence,
                            'tech_signal': tech_signal,
                            'tech_strength': tech_strength,
                            'price': price_data['close'].iloc[-1],
                            'reasoning': f"Demo mode: {ai_signal}({ai_confidence:.1f}%) + {tech_signal}({tech_strength:.1f}%)",
                            'executed': False
                        }
                        opportunities.append(opportunity)
                        
                        print(f"🎯 DEMO OPPORTUNITY: {pair} {final_signal} (Confidence: {final_confidence:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error analyzing {pair}: {e}")
        
        return opportunities

async def main():
    """Run demo trading simulation"""
    print("🎬 Starting Demo Trading Simulation...")
    print("   Ultra-aggressive parameters GUARANTEED to show trades")
    print("   This is for demonstration purposes only")
    print("=" * 60)
    
    # Create demo trading simulation
    sim = DemoTradingSimulation(
        initial_capital=1000.0,
        session_name=f"demo_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    try:
        # Run for 2 hours with 20-second intervals (very frequent checks)
        await sim.run_simulation(duration_hours=2.0, check_interval=20)
        
        print("\n🎉 Demo Trading Simulation Complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())