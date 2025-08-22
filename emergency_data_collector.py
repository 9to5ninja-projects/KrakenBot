#!/usr/bin/env python3
"""
Emergency Data Collector & AI Trainer
Collects real market data and trains AI models IMMEDIATELY
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Import our components
from exchange import ExchangeManager
from technical_indicators import calculate_all_indicators
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class EmergencyDataCollector:
    """Emergency system to collect data and train AI models immediately"""
    
    def __init__(self):
        self.exchange = ExchangeManager()
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("e:/KrakenBot/data/emergency_training")
        self.data_dir.mkdir(exist_ok=True)
        
        # Trading pairs
        self.pairs = ['ETH/CAD', 'BTC/CAD', 'SOL/CAD', 'XRP/CAD']
        
        # Models
        self.models = {}
        self.scalers = {}
        
    async def collect_massive_historical_data(self):
        """Collect as much historical data as possible RIGHT NOW"""
        
        print("🚨 EMERGENCY DATA COLLECTION STARTING...")
        print("   Collecting maximum historical data from Kraken...")
        
        all_data = {}
        
        for pair in self.pairs:
            print(f"📊 Collecting data for {pair}...")
            
            try:
                # Get multiple timeframes for more data
                timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
                pair_data = {}
                
                for timeframe in timeframes:
                    print(f"   Getting {timeframe} data...")
                    
                    # Get maximum available data (1000 candles per request)
                    ohlcv_data = self.exchange.get_historical_ohlcv(pair, timeframe, limit=1000)
                    
                    # Convert to DataFrame
                    if ohlcv_data:
                        data = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                        data.set_index('timestamp', inplace=True)
                    else:
                        data = pd.DataFrame()
                    
                    if not data.empty:
                        # Add technical indicators
                        data_with_indicators = calculate_all_indicators(data)
                        pair_data[timeframe] = data_with_indicators
                        print(f"   ✅ Got {len(data)} candles for {timeframe}")
                    else:
                        print(f"   ❌ No data for {timeframe}")
                    
                    # Small delay to avoid rate limits
                    import time
                    time.sleep(2)  # Use regular sleep since exchange calls are synchronous
                
                all_data[pair] = pair_data
                
            except Exception as e:
                print(f"❌ Error collecting data for {pair}: {e}")
        
        # Save collected data
        data_file = self.data_dir / "collected_market_data.json"
        
        # Convert DataFrames to JSON-serializable format
        json_data = {}
        for pair, timeframes in all_data.items():
            json_data[pair] = {}
            for tf, df in timeframes.items():
                if not df.empty:
                    json_data[pair][tf] = df.to_dict('records')
        
        with open(data_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Saved all collected data to {data_file}")
        return all_data
    
    def generate_synthetic_training_data(self, real_data: Dict) -> pd.DataFrame:
        """Generate synthetic training data based on real market patterns"""
        
        print("🧬 GENERATING SYNTHETIC TRAINING DATA...")
        
        synthetic_data = []
        
        for pair, timeframes in real_data.items():
            for tf, df in timeframes.items():
                if df.empty:
                    continue
                
                print(f"   Processing {pair} {tf} data ({len(df)} samples)...")
                
                # Extract patterns from real data
                for i in range(len(df) - 10):  # Need lookback window
                    try:
                        current = df.iloc[i:i+10]  # 10-period window
                        future = df.iloc[i+10] if i+10 < len(df) else df.iloc[-1]
                        
                        # Calculate features
                        features = self._extract_features(current)
                        
                        # Calculate target (future price movement)
                        price_change = (future['close'] - current['close'].iloc[-1]) / current['close'].iloc[-1]
                        
                        # Create training sample
                        sample = {
                            'pair': pair,
                            'timeframe': tf,
                            'timestamp': current.index[-1],
                            'price_change_target': price_change,
                            'signal_target': 1 if price_change > 0.002 else (-1 if price_change < -0.002 else 0),
                            **features
                        }
                        
                        synthetic_data.append(sample)
                        
                        # Generate variations of this pattern
                        for _ in range(3):  # 3 variations per real sample
                            variation = self._create_variation(sample)
                            synthetic_data.append(variation)
                            
                    except Exception as e:
                        continue
        
        df_synthetic = pd.DataFrame(synthetic_data)
        
        # Save synthetic data
        synthetic_file = self.data_dir / "synthetic_training_data.csv"
        df_synthetic.to_csv(synthetic_file, index=False)
        
        print(f"✅ Generated {len(df_synthetic)} synthetic training samples")
        print(f"💾 Saved to {synthetic_file}")
        
        return df_synthetic
    
    def _extract_features(self, df: pd.DataFrame) -> Dict:
        """Extract features from price data window"""
        
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
            
            return features
            
        except Exception as e:
            # Return default features if extraction fails
            return {
                'close': 0, 'volume': 0, 'rsi': 50, 'macd': 0, 'macd_signal': 0,
                'macd_histogram': 0, 'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0,
                'bb_position': 0.5, 'stoch_k': 50, 'stoch_d': 50, 'williams_r': -50,
                'cci': 0, 'atr': 0, 'price_change': 0, 'price_momentum': 0,
                'volume_ratio': 1, 'volatility': 0
            }
    
    def _create_variation(self, base_sample: Dict) -> Dict:
        """Create a variation of a training sample"""
        
        variation = base_sample.copy()
        
        # Add small random variations to features
        noise_level = 0.05  # 5% noise
        
        for key, value in variation.items():
            if key in ['pair', 'timeframe', 'timestamp']:
                continue
            
            if isinstance(value, (int, float)) and value != 0:
                # Add random noise
                noise = np.random.normal(0, abs(value) * noise_level)
                variation[key] = value + noise
        
        return variation
    
    def train_ai_models(self, training_data: pd.DataFrame):
        """Train AI models with the collected data"""
        
        print("🤖 TRAINING AI MODELS...")
        
        # Prepare features and targets
        feature_columns = [
            'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
            'stoch_k', 'stoch_d', 'williams_r', 'cci', 'atr',
            'price_change', 'price_momentum', 'volume_ratio', 'volatility'
        ]
        
        # Clean data
        training_data = training_data.dropna()
        training_data = training_data.replace([np.inf, -np.inf], 0)
        
        if len(training_data) < 100:
            print("❌ Not enough training data!")
            return False
        
        X = training_data[feature_columns].fillna(0)
        y_regression = training_data['price_change_target']
        y_classification = training_data['signal_target'].astype(int)  # Ensure integer labels
        
        print(f"   Training with {len(X)} samples...")
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_regression, y_classification, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print("   Training Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_classifier.fit(X_train_scaled, y_cls_train)
        
        # Train Random Forest Regressor
        print("   Training Random Forest Regressor...")
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_regressor.fit(X_train_scaled, y_reg_train)
        
        # Test models
        cls_score = rf_classifier.score(X_test_scaled, y_cls_test)
        reg_score = rf_regressor.score(X_test_scaled, y_reg_test)
        
        print(f"   ✅ Classifier Accuracy: {cls_score:.3f}")
        print(f"   ✅ Regressor R²: {reg_score:.3f}")
        
        # Save models
        models_dir = self.data_dir / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(rf_classifier, models_dir / "rf_classifier.pkl")
        joblib.dump(rf_regressor, models_dir / "rf_regressor.pkl")
        joblib.dump(scaler, models_dir / "scaler.pkl")
        
        # Save model info
        model_info = {
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classifier_accuracy': cls_score,
            'regressor_r2': reg_score,
            'feature_columns': feature_columns
        }
        
        with open(models_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 Models saved to {models_dir}")
        
        self.models['rf_classifier'] = rf_classifier
        self.models['rf_regressor'] = rf_regressor
        self.scalers['main'] = scaler
        
        return True
    
    async def test_trained_models(self):
        """Test the trained models with live data"""
        
        print("🧪 TESTING TRAINED MODELS WITH LIVE DATA...")
        
        if not self.models:
            print("❌ No trained models available!")
            return
        
        for pair in self.pairs[:2]:  # Test with first 2 pairs
            try:
                print(f"   Testing {pair}...")
                
                # Get fresh data
                ohlcv_data = self.exchange.get_historical_ohlcv(pair, '1h', limit=50)
                
                if ohlcv_data:
                    data = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                    data_with_indicators = calculate_all_indicators(data)
                else:
                    data_with_indicators = pd.DataFrame()
                
                if data_with_indicators.empty:
                    continue
                
                # Extract features
                features = self._extract_features(data_with_indicators.tail(10))
                
                # Prepare for prediction
                feature_columns = [
                    'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
                    'stoch_k', 'stoch_d', 'williams_r', 'cci', 'atr',
                    'price_change', 'price_momentum', 'volume_ratio', 'volatility'
                ]
                
                X = np.array([[features.get(col, 0) for col in feature_columns]])
                X_scaled = self.scalers['main'].transform(X)
                
                # Make predictions
                cls_pred = self.models['rf_classifier'].predict(X_scaled)[0]
                cls_proba = self.models['rf_classifier'].predict_proba(X_scaled)[0]
                reg_pred = self.models['rf_regressor'].predict(X_scaled)[0]
                
                signal = "BUY" if cls_pred == 1 else ("SELL" if cls_pred == -1 else "HOLD")
                confidence = max(cls_proba) * 100
                
                print(f"   🎯 {pair}: {signal} (Confidence: {confidence:.1f}%, Expected: {reg_pred:+.4f})")
                
            except Exception as e:
                print(f"   ❌ Error testing {pair}: {e}")

async def main():
    """Emergency data collection and AI training"""
    
    print("🚨 EMERGENCY AI TRAINING SYSTEM")
    print("   Collecting data and training models RIGHT NOW!")
    print("=" * 60)
    
    collector = EmergencyDataCollector()
    
    try:
        # Step 1: Collect massive historical data
        real_data = await collector.collect_massive_historical_data()
        
        # Step 2: Generate synthetic training data
        training_data = collector.generate_synthetic_training_data(real_data)
        
        # Step 3: Train AI models
        success = collector.train_ai_models(training_data)
        
        if success:
            # Step 4: Test trained models
            await collector.test_trained_models()
            
            print("\n🎉 EMERGENCY TRAINING COMPLETE!")
            print("   AI models are now trained and ready to trade!")
            print("   You can now run the trading simulation with working AI.")
        else:
            print("\n❌ Training failed - not enough data collected")
            
    except Exception as e:
        print(f"❌ Emergency training failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())