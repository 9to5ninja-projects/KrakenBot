#!/usr/bin/env python3
"""
Working AI Trading System
Uses the newly trained AI models for real trading decisions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
import joblib
from typing import Dict, List, Tuple, Optional

# Import our components
from exchange import ExchangeManager
from technical_indicators import calculate_all_indicators
from enhanced_simulation_system import EnhancedSimulationSystem, StrategyParameters

class WorkingAITradingSystem(EnhancedSimulationSystem):
    """AI Trading System with trained models"""
    
    def __init__(self, initial_capital: float = 1000.0, session_name: str = None):
        super().__init__(initial_capital, session_name)
        
        # Load trained AI models
        self.models_dir = Path("e:/KrakenBot/data/emergency_training/trained_models")
        self.ai_models = {}
        self.scaler = None
        self.feature_columns = []
        
        self._load_trained_models()
        
        # AI-optimized parameters (more aggressive since AI is working)
        self.strategy_params = StrategyParameters(
            buy_threshold=-0.002,   # -0.2% (AI-guided)
            sell_threshold=0.002,   # +0.2% (AI-guided)
            lookback_periods=10,    # AI looks at 10 periods
            min_trade_amount=25.0,  # Minimum trade
            max_position_size=0.15, # 15% max per position
            stop_loss_pct=-0.015,   # -1.5% stop loss
            take_profit_pct=0.012,  # +1.2% take profit
            volatility_threshold=0.015,   # AI handles volatility
            volume_threshold=1.2,   # AI considers volume
            rsi_oversold=35,        # AI-optimized RSI
            rsi_overbought=65,      # AI-optimized RSI
            bollinger_buy_position=0.3,   # AI-optimized Bollinger
            bollinger_sell_position=0.7   # AI-optimized Bollinger
        )
        
        print(f"🤖 WORKING AI TRADING SYSTEM ENABLED")
        print(f"   Using trained AI models for decision making")
        print(f"   AI Classifier Accuracy: {self.model_info.get('classifier_accuracy', 0):.1%}")
        print(f"   AI Regressor R²: {self.model_info.get('regressor_r2', 0):.3f}")
        print(f"   Training Samples: {self.model_info.get('training_samples', 0):,}")
    
    def _load_trained_models(self):
        """Load the trained AI models"""
        
        try:
            # Load model info
            model_info_file = self.models_dir / "model_info.json"
            if model_info_file.exists():
                with open(model_info_file, 'r') as f:
                    self.model_info = json.load(f)
                    self.feature_columns = self.model_info['feature_columns']
            else:
                print("❌ No trained models found! Run emergency_data_collector.py first")
                self.model_info = {}
                return
            
            # Load models
            classifier_file = self.models_dir / "rf_classifier.pkl"
            regressor_file = self.models_dir / "rf_regressor.pkl"
            scaler_file = self.models_dir / "scaler.pkl"
            
            if all(f.exists() for f in [classifier_file, regressor_file, scaler_file]):
                self.ai_models['classifier'] = joblib.load(classifier_file)
                self.ai_models['regressor'] = joblib.load(regressor_file)
                self.scaler = joblib.load(scaler_file)
                print("✅ AI models loaded successfully!")
            else:
                print("❌ Model files missing!")
                
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            self.ai_models = {}
    
    def _get_ai_prediction(self, price_data: pd.DataFrame) -> Tuple[str, float, float]:
        """Get AI prediction for trading decision"""
        
        if not self.ai_models or not self.scaler:
            return 'HOLD', 50.0, 0.0
        
        try:
            # Calculate technical indicators
            data_with_indicators = calculate_all_indicators(price_data)
            
            if data_with_indicators.empty:
                return 'HOLD', 50.0, 0.0
            
            # Extract features
            features = self._extract_features_for_ai(data_with_indicators)
            
            # Prepare for prediction
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            cls_pred = self.ai_models['classifier'].predict(X_scaled)[0]
            cls_proba = self.ai_models['classifier'].predict_proba(X_scaled)[0]
            reg_pred = self.ai_models['regressor'].predict(X_scaled)[0]
            
            # Convert to trading signal
            if cls_pred == 1:
                signal = 'BUY'
            elif cls_pred == -1:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            confidence = max(cls_proba) * 100
            expected_return = reg_pred
            
            return signal, confidence, expected_return
            
        except Exception as e:
            print(f"❌ AI prediction error: {e}")
            return 'HOLD', 50.0, 0.0
    
    def _extract_features_for_ai(self, df: pd.DataFrame) -> Dict:
        """Extract features for AI prediction"""
        
        try:
            latest = df.iloc[-1]
            
            features = {
                'close': latest.get('close', 0),
                'volume': latest.get('volume', 0),
                'rsi': latest.get('rsi', 50),
                'macd': latest.get('macd', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'macd_histogram': latest.get('macd_histogram', 0),
                'bb_upper': latest.get('bb_upper', 0),
                'bb_middle': latest.get('bb_middle', 0),
                'bb_lower': latest.get('bb_lower', 0),
                'bb_position': latest.get('bb_position', 0.5),
                'stoch_k': latest.get('stoch_k', 50),
                'stoch_d': latest.get('stoch_d', 50),
                'williams_r': latest.get('williams_r', -50),
                'cci': latest.get('cci', 0),
                'atr': latest.get('atr', 0),
                'price_change': df['close'].pct_change().iloc[-1] if len(df) > 1 else 0,
                'price_momentum': df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0,
                'volume_ratio': latest.get('volume', 0) / df['volume'].mean() if df['volume'].mean() > 0 else 1,
                'volatility': df['close'].std() if len(df) > 1 else 0,
            }
            
            # Replace any NaN or inf values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    if key == 'bb_position':
                        features[key] = 0.5
                    elif key in ['rsi', 'stoch_k', 'stoch_d']:
                        features[key] = 50
                    elif key == 'williams_r':
                        features[key] = -50
                    elif key == 'volume_ratio':
                        features[key] = 1
                    else:
                        features[key] = 0
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            # Return default features
            return {col: 0 for col in self.feature_columns}
    
    async def _analyze_trading_opportunities(self):
        """AI-powered opportunity analysis"""
        opportunities = []
        
        for pair in self.trading_pairs:
            try:
                # Get market data
                price_data = await self.multi_coin_analyzer.get_market_data(pair, '1h', 100)
                
                if price_data.empty:
                    continue
                
                # Get AI prediction
                ai_signal, ai_confidence, expected_return = self._get_ai_prediction(price_data)
                
                # Get technical analysis for confirmation
                technical_analysis = self.technical_analyzer.analyze_pair(price_data, pair)
                tech_signal = technical_analysis.get('composite_signal', {}).get('signal', 'HOLD')
                tech_strength = technical_analysis.get('composite_signal', {}).get('strength', 0.0)
                
                # AI-driven opportunity creation (AI takes priority)
                should_trade = False
                final_signal = 'HOLD'
                final_confidence = 0.0
                
                # Strong AI signal
                if ai_signal in ['BUY', 'SELL'] and ai_confidence > 60:
                    should_trade = True
                    final_signal = ai_signal
                    final_confidence = ai_confidence
                
                # AI + Technical agreement
                elif ai_signal == tech_signal and ai_signal in ['BUY', 'SELL']:
                    if ai_confidence > 55 and tech_strength > 30:
                        should_trade = True
                        final_signal = ai_signal
                        final_confidence = (ai_confidence + tech_strength) / 2
                
                # Very strong technical with weak AI agreement
                elif tech_signal in ['BUY', 'SELL'] and tech_strength > 70:
                    if ai_confidence > 50:  # AI not strongly opposing
                        should_trade = True
                        final_signal = tech_signal
                        final_confidence = tech_strength * 0.8  # Discount for no AI agreement
                
                if should_trade:
                    opportunity = {
                        'timestamp': datetime.now().isoformat(),
                        'pair': pair,
                        'signal': final_signal,
                        'confidence': final_confidence,
                        'ai_signal': ai_signal,
                        'ai_confidence': ai_confidence,
                        'ai_expected_return': expected_return,
                        'tech_signal': tech_signal,
                        'tech_strength': tech_strength,
                        'price': price_data['close'].iloc[-1],
                        'reasoning': f"AI: {ai_signal}({ai_confidence:.1f}%) + Tech: {tech_signal}({tech_strength:.1f}%)",
                        'executed': False
                    }
                    opportunities.append(opportunity)
                    
                    print(f"🤖 AI OPPORTUNITY: {pair} {final_signal} (Confidence: {final_confidence:.1f}%, Expected: {expected_return:+.4f})")
                
            except Exception as e:
                print(f"❌ Error analyzing {pair}: {e}")
        
        return opportunities

async def main():
    """Run working AI trading simulation"""
    print("🤖 Starting Working AI Trading System...")
    print("   Using trained AI models for intelligent trading")
    print("   This system makes REAL AI-driven decisions!")
    print("=" * 60)
    
    # Create AI trading system
    ai_system = WorkingAITradingSystem(
        initial_capital=1000.0,
        session_name=f"working_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    try:
        # Run for 3 hours with 30-second intervals
        await ai_system.run_simulation(duration_hours=3.0, check_interval=30)
        
        print("\n🎉 AI Trading Simulation Complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ AI simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())