#!/usr/bin/env python3
"""
Active Trading Simulation
More aggressive parameters for demonstration and testing
"""

import asyncio
from datetime import datetime, timedelta
from enhanced_simulation_system import EnhancedSimulationSystem, StrategyParameters

class ActiveTradingSimulation(EnhancedSimulationSystem):
    """Enhanced simulation with more active trading parameters"""
    
    def __init__(self, initial_capital: float = 1000.0, session_name: str = None):
        super().__init__(initial_capital, session_name)
        
        # Override with more active strategy parameters
        self.strategy_params = StrategyParameters(
            buy_threshold=-0.001,   # -0.1% (more sensitive)
            sell_threshold=0.002,   # +0.2% (more sensitive)
            lookback_periods=3,     # Shorter lookback
            min_trade_amount=25.0,  # Lower minimum trade
            max_position_size=0.15, # 15% max per position
            stop_loss_pct=-0.015,   # -1.5% stop loss
            take_profit_pct=0.01,   # +1.0% take profit
            volatility_threshold=0.015,  # Lower volatility threshold
            volume_threshold=1.2,   # Lower volume threshold
            rsi_oversold=35,        # Less extreme RSI
            rsi_overbought=65,      # Less extreme RSI
            bollinger_buy_position=0.3,   # Less extreme Bollinger
            bollinger_sell_position=0.7   # Less extreme Bollinger
        )
        
        print(f"🚀 ACTIVE TRADING MODE ENABLED")
        print(f"   Buy Threshold: {self.strategy_params.buy_threshold:.3f} ({self.strategy_params.buy_threshold*100:.1f}%)")
        print(f"   Sell Threshold: {self.strategy_params.sell_threshold:.3f} ({self.strategy_params.sell_threshold*100:.1f}%)")
        print(f"   Min Trade: ${self.strategy_params.min_trade_amount}")
        print(f"   Max Position: {self.strategy_params.max_position_size:.1%}")
    
    def _should_create_opportunity(self, ai_signal: str, tech_signal: str, 
                                 ai_confidence: float, tech_strength: float) -> bool:
        """More aggressive opportunity creation for active trading"""
        
        # Lower thresholds for demonstration
        if ai_signal == 'BUY' and tech_signal == 'BUY':
            return ai_confidence > 45 and tech_strength > 30  # Lowered from 60/40
        elif ai_signal == 'SELL' and tech_signal == 'SELL':
            return ai_confidence > 45 and tech_strength > 30  # Lowered from 60/40
        
        # Allow single signals with lower confidence
        elif ai_signal in ['BUY', 'SELL'] and ai_confidence > 65:  # Lowered from 80
            return True
        elif tech_signal in ['BUY', 'SELL'] and tech_strength > 55:  # Lowered from 70
            return True
        
        # Allow moderate signals if they align
        elif ai_signal == tech_signal and ai_signal in ['BUY', 'SELL']:
            return ai_confidence > 35 and tech_strength > 25  # Much lower threshold
        
        return False

async def main():
    """Run active trading simulation"""
    print("🚀 Starting Active Trading Simulation...")
    print("   More aggressive parameters for demonstration")
    print("   This will trade more frequently than the conservative version")
    print("=" * 60)
    
    # Create active trading simulation
    sim = ActiveTradingSimulation(
        initial_capital=1000.0,
        session_name=f"active_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    try:
        # Run for 4 hours with 30-second intervals (more frequent checks)
        await sim.run_simulation(duration_hours=4.0, check_interval=30)
        
        print("\n🎉 Active Trading Simulation Complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())