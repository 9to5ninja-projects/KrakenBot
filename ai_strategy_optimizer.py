"""
AI-Driven Strategy Optimizer for KrakenBot
Continuously learns and adapts trading strategies based on market conditions and performance
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from advanced_technical_indicators import AdvancedTechnicalAnalyzer

@dataclass
class StrategyParameters:
    """Trading strategy parameters that can be optimized."""
    buy_threshold: float = -0.003  # -0.3%
    sell_threshold: float = 0.005  # +0.5%
    lookback_periods: int = 5
    min_trade_amount: float = 25.0
    max_position_size: float = 0.25  # 25%
    stop_loss_pct: float = -0.02  # -2%
    take_profit_pct: float = 0.015  # +1.5%
    volatility_threshold: float = 0.02  # 2%
    volume_threshold: float = 1.5  # 1.5x average volume
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bollinger_buy_position: float = 0.2  # Buy when price is in bottom 20% of bands
    bollinger_sell_position: float = 0.8  # Sell when price is in top 80% of bands

@dataclass
class MarketCondition:
    """Represents current market conditions."""
    volatility: float
    trend: str  # 'bullish', 'bearish', 'sideways'
    volume_ratio: float
    rsi_level: float
    bollinger_position: float
    macd_signal: str
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # Day of week (0-6)

@dataclass
class TradeOutcome:
    """Represents the outcome of a trade."""
    entry_price: float
    exit_price: float
    profit_pct: float
    profit_cad: float
    duration_minutes: int
    market_condition: MarketCondition
    strategy_params: StrategyParameters
    success: bool

class AIStrategyOptimizer:
    """AI-powered strategy optimization system."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.models = {}
        self.scalers = {}
        self.trade_history = []
        self.current_params = StrategyParameters()
        self.performance_metrics = {}
        
        # Load existing data
        self._load_historical_data()
        self._initialize_models()
    
    def _load_historical_data(self):
        """Load historical trading data for analysis."""
        print("Loading historical trading data...")
        
        # Find all session directories
        session_dirs = []
        for pattern in ["live_session_*", "optimized_session_*", "aggressive_test_*", 
                       "simple_pair_monitor_*", "extended_validation_*"]:
            session_dirs.extend(self.data_dir.glob(pattern))
        
        for session_dir in session_dirs:
            try:
                self._load_session_data(session_dir)
            except Exception as e:
                print(f"Error loading {session_dir.name}: {e}")
        
        print(f"Loaded {len(self.trade_history)} historical trades")
    
    def _load_session_data(self, session_dir: Path):
        """Load data from a specific session."""
        # Load trades
        trades_file = session_dir / "trades.json"
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                trades = json.load(f)
            
            # Load portfolio history for market conditions
            portfolio_file = session_dir / "portfolio_history.json"
            portfolio_data = []
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
            
            # Process trades and extract outcomes
            for trade in trades:
                try:
                    outcome = self._extract_trade_outcome(trade, portfolio_data)
                    if outcome:
                        self.trade_history.append(outcome)
                except Exception as e:
                    continue
    
    def _extract_trade_outcome(self, trade: Dict, portfolio_data: List[Dict]) -> Optional[TradeOutcome]:
        """Extract trade outcome from trade and portfolio data."""
        try:
            # Find corresponding portfolio data
            trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            
            # Find market condition at trade time
            market_condition = self._extract_market_condition(trade_time, portfolio_data, trade['pair'])
            
            # For now, simulate trade outcomes (in real system, this would be actual data)
            entry_price = trade['price']
            
            # Simulate exit based on typical outcomes
            if trade.get('side', 'buy') == 'buy':
                # Simulate sell after some time
                profit_pct = np.random.normal(0.002, 0.01)  # Average 0.2% profit with 1% std
                exit_price = entry_price * (1 + profit_pct)
                duration_minutes = np.random.randint(30, 240)  # 30 minutes to 4 hours
            else:
                profit_pct = np.random.normal(-0.002, 0.01)  # Reverse for sell
                exit_price = entry_price * (1 + profit_pct)
                duration_minutes = np.random.randint(30, 240)
            
            profit_cad = trade.get('cost', 25) * profit_pct
            success = profit_pct > 0
            
            return TradeOutcome(
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                profit_cad=profit_cad,
                duration_minutes=duration_minutes,
                market_condition=market_condition,
                strategy_params=self.current_params,  # Use current params as default
                success=success
            )
        except Exception as e:
            return None
    
    def _extract_market_condition(self, timestamp: datetime, portfolio_data: List[Dict], pair: str) -> MarketCondition:
        """Extract market conditions at a specific time."""
        # Find closest portfolio data point
        closest_data = None
        min_diff = float('inf')
        
        for data_point in portfolio_data:
            data_time = datetime.fromisoformat(data_point['timestamp'].replace('Z', '+00:00'))
            diff = abs((data_time - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_data = data_point
        
        if not closest_data:
            # Return default market condition
            return MarketCondition(
                volatility=0.02,
                trend='sideways',
                volume_ratio=1.0,
                rsi_level=50,
                bollinger_position=0.5,
                macd_signal='HOLD',
                time_of_day=timestamp.hour,
                day_of_week=timestamp.weekday()
            )
        
        # Extract market metrics
        prices = closest_data.get('prices', {})
        current_price = prices.get(pair, 0)
        
        # Calculate volatility from recent price changes (simplified)
        volatility = 0.02  # Default 2%
        
        # Determine trend (simplified)
        trend = 'sideways'
        
        return MarketCondition(
            volatility=volatility,
            trend=trend,
            volume_ratio=1.0,  # Default
            rsi_level=50,  # Default neutral
            bollinger_position=0.5,  # Default middle
            macd_signal='HOLD',
            time_of_day=timestamp.hour,
            day_of_week=timestamp.weekday()
        )
    
    def _initialize_models(self):
        """Initialize machine learning models."""
        print(" Initializing AI models...")
        
        # Model for predicting trade success probability
        self.models['success_predictor'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Model for predicting profit percentage
        self.models['profit_predictor'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        # Scalers for feature normalization
        self.scalers['features'] = StandardScaler()
        
        # Train models if we have enough data
        if len(self.trade_history) >= 10:
            self._train_models()
    
    def _prepare_features(self, market_condition: MarketCondition, strategy_params: StrategyParameters) -> np.ndarray:
        """Prepare features for machine learning models."""
        features = [
            market_condition.volatility,
            1 if market_condition.trend == 'bullish' else -1 if market_condition.trend == 'bearish' else 0,
            market_condition.volume_ratio,
            market_condition.rsi_level,
            market_condition.bollinger_position,
            1 if market_condition.macd_signal == 'BUY' else -1 if market_condition.macd_signal == 'SELL' else 0,
            market_condition.time_of_day / 24.0,  # Normalize to 0-1
            market_condition.day_of_week / 7.0,   # Normalize to 0-1
            strategy_params.buy_threshold,
            strategy_params.sell_threshold,
            strategy_params.lookback_periods / 20.0,  # Normalize
            strategy_params.max_position_size,
            strategy_params.stop_loss_pct,
            strategy_params.take_profit_pct,
            strategy_params.volatility_threshold,
            strategy_params.rsi_oversold / 100.0,
            strategy_params.rsi_overbought / 100.0,
            strategy_params.bollinger_buy_position,
            strategy_params.bollinger_sell_position
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _train_models(self):
        """Train machine learning models on historical data."""
        if len(self.trade_history) < 10:
            print("WARNING: Not enough data to train models")
            return
        
        print(f" Training AI models on {len(self.trade_history)} trades...")
        
        # Prepare training data
        X = []
        y_success = []
        y_profit = []
        
        for outcome in self.trade_history:
            features = self._prepare_features(outcome.market_condition, outcome.strategy_params)
            X.append(features.flatten())
            y_success.append(1 if outcome.success else 0)
            y_profit.append(outcome.profit_pct)
        
        X = np.array(X)
        y_success = np.array(y_success)
        y_profit = np.array(y_profit)
        
        # Scale features
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Split data
        X_train, X_test, y_success_train, y_success_test = train_test_split(
            X_scaled, y_success, test_size=0.2, random_state=42
        )
        
        X_train_profit, X_test_profit, y_profit_train, y_profit_test = train_test_split(
            X_scaled, y_profit, test_size=0.2, random_state=42
        )
        
        # Train success predictor
        self.models['success_predictor'].fit(X_train, y_success_train)
        success_accuracy = accuracy_score(y_success_test, 
                                        self.models['success_predictor'].predict(X_test))
        
        # Train profit predictor
        self.models['profit_predictor'].fit(X_train_profit, y_profit_train)
        profit_mse = mean_squared_error(y_profit_test, 
                                      self.models['profit_predictor'].predict(X_test_profit))
        
        print(f"SUCCESS: Success Predictor Accuracy: {success_accuracy:.3f}")
        print(f"SUCCESS: Profit Predictor MSE: {profit_mse:.6f}")
        
        # Save models
        self._save_models()
    
    def _save_models(self):
        """Save trained models to disk."""
        models_dir = self.data_dir / "ai_models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            with open(models_dir / f"{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        for name, scaler in self.scalers.items():
            with open(models_dir / f"{name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
    
    def _load_models(self):
        """Load trained models from disk."""
        models_dir = self.data_dir / "ai_models"
        if not models_dir.exists():
            return
        
        for model_file in models_dir.glob("*.pkl"):
            if "scaler" in model_file.name:
                name = model_file.name.replace("_scaler.pkl", "")
                with open(model_file, 'rb') as f:
                    self.scalers[name] = pickle.load(f)
            else:
                name = model_file.name.replace(".pkl", "")
                with open(model_file, 'rb') as f:
                    self.models[name] = pickle.load(f)
    
    def predict_trade_outcome(self, market_condition: MarketCondition, 
                            strategy_params: StrategyParameters) -> Dict:
        """Predict the outcome of a trade given market conditions and strategy."""
        if 'success_predictor' not in self.models or 'features' not in self.scalers:
            return {
                'success_probability': 0.5,
                'expected_profit': 0.0,
                'confidence': 0.0,
                'recommendation': 'INSUFFICIENT_DATA'
            }
        
        # Prepare features
        features = self._prepare_features(market_condition, strategy_params)
        
        # Check if scaler is fitted
        try:
            features_scaled = self.scalers['features'].transform(features)
        except Exception:
            # Scaler not fitted, return default prediction
            return {
                'success_probability': 0.5,
                'expected_profit': 0.0,
                'confidence': 0.0,
                'recommendation': 'INSUFFICIENT_DATA'
            }
        
        # Make predictions
        success_prob = self.models['success_predictor'].predict_proba(features_scaled)[0][1]
        expected_profit = self.models['profit_predictor'].predict(features_scaled)[0]
        
        # Calculate confidence based on model certainty
        confidence = max(success_prob, 1 - success_prob) * 100
        
        # Generate recommendation
        if success_prob > 0.7 and expected_profit > 0.002:
            recommendation = 'STRONG_BUY'
        elif success_prob > 0.6 and expected_profit > 0.001:
            recommendation = 'BUY'
        elif success_prob < 0.4 or expected_profit < -0.001:
            recommendation = 'AVOID'
        else:
            recommendation = 'NEUTRAL'
        
        return {
            'success_probability': success_prob,
            'expected_profit': expected_profit,
            'confidence': confidence,
            'recommendation': recommendation
        }
    
    def optimize_strategy_parameters(self, target_pairs: List[str]) -> StrategyParameters:
        """Optimize strategy parameters using AI predictions."""
        print(" Optimizing strategy parameters with AI...")
        
        if len(self.trade_history) < 20:
            print("WARNING: Not enough data for optimization, using default parameters")
            return self.current_params
        
        # Define parameter ranges to test
        param_ranges = {
            'buy_threshold': np.linspace(-0.005, -0.001, 10),
            'sell_threshold': np.linspace(0.002, 0.008, 10),
            'lookback_periods': [3, 5, 7, 10, 15],
            'max_position_size': [0.15, 0.20, 0.25, 0.30],
            'stop_loss_pct': [-0.03, -0.02, -0.015, -0.01],
            'take_profit_pct': [0.01, 0.015, 0.02, 0.025],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75]
        }
        
        best_params = self.current_params
        best_score = -float('inf')
        
        # Test different parameter combinations (simplified grid search)
        for buy_thresh in param_ranges['buy_threshold'][:5]:  # Limit combinations
            for sell_thresh in param_ranges['sell_threshold'][:5]:
                for max_pos in param_ranges['max_position_size']:
                    test_params = StrategyParameters(
                        buy_threshold=buy_thresh,
                        sell_threshold=sell_thresh,
                        max_position_size=max_pos,
                        lookback_periods=self.current_params.lookback_periods,
                        stop_loss_pct=self.current_params.stop_loss_pct,
                        take_profit_pct=self.current_params.take_profit_pct
                    )
                    
                    # Evaluate parameters
                    score = self._evaluate_parameters(test_params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = test_params
        
        print(f"SUCCESS: Optimization complete. Best score: {best_score:.4f}")
        print(f" Optimized parameters:")
        print(f"   Buy threshold: {best_params.buy_threshold:.4f}")
        print(f"   Sell threshold: {best_params.sell_threshold:.4f}")
        print(f"   Max position: {best_params.max_position_size:.2f}")
        
        return best_params
    
    def _evaluate_parameters(self, params: StrategyParameters) -> float:
        """Evaluate strategy parameters using historical data."""
        total_score = 0
        count = 0
        
        for outcome in self.trade_history[-50:]:  # Use recent data
            prediction = self.predict_trade_outcome(outcome.market_condition, params)
            
            # Score based on prediction accuracy and profitability
            if outcome.success and prediction['success_probability'] > 0.5:
                total_score += prediction['success_probability'] * outcome.profit_pct
            elif not outcome.success and prediction['success_probability'] <= 0.5:
                total_score += (1 - prediction['success_probability']) * abs(outcome.profit_pct)
            else:
                total_score -= abs(outcome.profit_pct)  # Penalty for wrong prediction
            
            count += 1
        
        return total_score / max(count, 1)
    
    def get_current_market_analysis(self, pair: str, price_data: pd.DataFrame) -> Dict:
        """Get comprehensive market analysis for current conditions."""
        # Technical analysis
        technical_analysis = self.technical_analyzer.analyze_pair(price_data, pair)
        
        # Extract market condition
        current_time = datetime.now()
        market_condition = MarketCondition(
            volatility=0.02,  # Would be calculated from recent price data
            trend='sideways',  # Would be determined from technical analysis
            volume_ratio=1.0,
            rsi_level=technical_analysis.get('indicators', {}).get('rsi', {}).get('rsi', 50),
            bollinger_position=technical_analysis.get('indicators', {}).get('bollinger', {}).get('position', 0.5),
            macd_signal=technical_analysis.get('indicators', {}).get('macd', {}).get('signal', 'HOLD'),
            time_of_day=current_time.hour,
            day_of_week=current_time.weekday()
        )
        
        # AI prediction
        ai_prediction = self.predict_trade_outcome(market_condition, self.current_params)
        
        return {
            'pair': pair,
            'timestamp': current_time,
            'technical_analysis': technical_analysis,
            'market_condition': asdict(market_condition),
            'ai_prediction': ai_prediction,
            'current_parameters': asdict(self.current_params),
            'recommendation': self._generate_final_recommendation(technical_analysis, ai_prediction)
        }
    
    def _generate_final_recommendation(self, technical_analysis: Dict, ai_prediction: Dict) -> Dict:
        """Generate final trading recommendation combining technical and AI analysis."""
        # Get technical signal
        tech_signal = technical_analysis.get('composite_signal', {}).get('signal', 'HOLD')
        tech_strength = technical_analysis.get('composite_signal', {}).get('strength', 0)
        
        # Get AI recommendation
        ai_rec = ai_prediction.get('recommendation', 'NEUTRAL')
        ai_confidence = ai_prediction.get('confidence', 0)
        
        # Combine signals
        if tech_signal == 'BUY' and ai_rec in ['STRONG_BUY', 'BUY']:
            final_signal = 'BUY'
            confidence = min(100, (tech_strength + ai_confidence) / 2)
        elif tech_signal == 'SELL' and ai_rec == 'AVOID':
            final_signal = 'SELL'
            confidence = min(100, (tech_strength + ai_confidence) / 2)
        elif tech_signal in ['BUY', 'SELL'] and ai_rec == 'NEUTRAL':
            final_signal = tech_signal
            confidence = tech_strength * 0.7  # Reduced confidence
        elif ai_rec in ['STRONG_BUY', 'BUY'] and tech_signal == 'HOLD':
            final_signal = 'BUY'
            confidence = ai_confidence * 0.8  # Reduced confidence
        else:
            final_signal = 'HOLD'
            confidence = 50
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'technical_signal': tech_signal,
            'ai_recommendation': ai_rec,
            'reasoning': self._explain_recommendation(tech_signal, ai_rec, final_signal)
        }
    
    def _explain_recommendation(self, tech_signal: str, ai_rec: str, final_signal: str) -> str:
        """Explain the reasoning behind the recommendation."""
        if final_signal == 'BUY':
            if tech_signal == 'BUY' and ai_rec in ['STRONG_BUY', 'BUY']:
                return "Both technical indicators and AI model suggest buying opportunity"
            elif tech_signal == 'BUY':
                return "Technical indicators suggest buying, AI model is neutral"
            else:
                return "AI model suggests buying opportunity despite mixed technical signals"
        elif final_signal == 'SELL':
            return "Technical indicators suggest selling and AI model recommends avoiding"
        else:
            return "Mixed signals from technical and AI analysis, recommend holding"

if __name__ == "__main__":
    # Test the AI strategy optimizer
    optimizer = AIStrategyOptimizer()
    
    print(" AI Strategy Optimizer Test")
    print("=" * 50)
    
    # Test parameter optimization
    optimized_params = optimizer.optimize_strategy_parameters(['ETH/CAD', 'BTC/CAD'])
    
    print(f"\n Current Parameters:")
    for key, value in asdict(optimized_params).items():
        print(f"   {key}: {value}")
    
    # Test market analysis
    from advanced_technical_indicators import create_sample_price_data
    sample_data = create_sample_price_data("ETH/CAD")
    analysis = optimizer.get_current_market_analysis("ETH/CAD", sample_data)
    
    print(f"\n Market Analysis for ETH/CAD:")
    print(f"AI Recommendation: {analysis['ai_prediction']['recommendation']}")
    print(f"Success Probability: {analysis['ai_prediction']['success_probability']:.3f}")
    print(f"Expected Profit: {analysis['ai_prediction']['expected_profit']:.4f}")
    print(f"Final Signal: {analysis['recommendation']['signal']}")
    print(f"Confidence: {analysis['recommendation']['confidence']:.1f}%")