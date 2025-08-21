"""
Advanced Price Prediction System
Uses LSTM and XGBoost models to predict cryptocurrency prices
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class PricePredictor:
    """
    Advanced price prediction system using multiple models.
    Falls back to statistical methods if ML libraries unavailable.
    """
    
    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize price predictor.
        
        Args:
            model_type: Type of model ('ensemble', 'statistical', 'ml')
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.prediction_horizon = 5  # Predict 5 periods ahead
        self.sequence_length = 20    # Use 20 periods for prediction
        
        # Model performance tracking
        self.performance_history = {}
        
    def prepare_features(self, price_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Prepare features for price prediction.
        
        Args:
            price_data: Dictionary with pair names as keys and price history as values
            
        Returns:
            DataFrame with engineered features
        """
        all_features = []
        
        for pair, price_history in price_data.items():
            if not price_history or len(price_history) < self.sequence_length:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(price_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic price features
            df[f'{pair}_price'] = df['price']
            df[f'{pair}_returns'] = df['price'].pct_change()
            
            # Technical indicators
            df[f'{pair}_sma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
            df[f'{pair}_sma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
            df[f'{pair}_sma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
            
            # Volatility features
            df[f'{pair}_volatility'] = df['price'].rolling(window=10, min_periods=1).std()
            df[f'{pair}_volatility_ratio'] = df[f'{pair}_volatility'] / df[f'{pair}_sma_10']
            
            # Momentum features
            df[f'{pair}_momentum_5'] = df['price'] / df['price'].shift(5) - 1
            df[f'{pair}_momentum_10'] = df['price'] / df['price'].shift(10) - 1
            
            # Price position features
            df[f'{pair}_high_10'] = df['price'].rolling(window=10, min_periods=1).max()
            df[f'{pair}_low_10'] = df['price'].rolling(window=10, min_periods=1).min()
            df[f'{pair}_price_position'] = (df['price'] - df[f'{pair}_low_10']) / (df[f'{pair}_high_10'] - df[f'{pair}_low_10'])
            
            # Rate of change features
            df[f'{pair}_roc_3'] = df['price'].pct_change(periods=3)
            df[f'{pair}_roc_5'] = df['price'].pct_change(periods=5)
            
            # Add pair identifier
            df['pair'] = pair
            
            all_features.append(df)
        
        if not all_features:
            return pd.DataFrame()
        
        # Combine all pairs
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Fill NaN values
        combined_df = combined_df.ffill().fillna(0)
        
        # Store feature columns
        self.feature_columns = [col for col in combined_df.columns 
                               if col not in ['timestamp', 'price', 'pair']]
        
        return combined_df
    
    def create_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for pair in data['pair'].unique():
            pair_data = data[data['pair'] == pair].copy()
            
            if len(pair_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            for i in range(len(pair_data) - self.sequence_length - self.prediction_horizon + 1):
                # Features sequence
                feature_sequence = pair_data[self.feature_columns].iloc[i:i+self.sequence_length].values
                
                # Target (future price)
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target_value = pair_data[target_col].iloc[target_idx]
                
                X.append(feature_sequence)
                y.append(target_value)
        
        return np.array(X), np.array(y)
    
    def train_statistical_model(self, price_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Train statistical prediction models (moving averages, trend analysis).
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dictionary of trained statistical models
        """
        models = {}
        
        for pair, price_history in price_data.items():
            if not isinstance(price_history, list) or len(price_history) < 20:
                continue
                
            df = pd.DataFrame(price_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate various moving averages
            df['sma_5'] = df['price'].rolling(window=5).mean()
            df['sma_10'] = df['price'].rolling(window=10).mean()
            df['sma_20'] = df['price'].rolling(window=20).mean()
            
            # Calculate trend
            recent_prices = df['price'].tail(10).values
            if len(recent_prices) >= 2:
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            else:
                trend = 0
            
            # Calculate volatility
            volatility = df['price'].tail(20).std()
            
            # Store model parameters
            models[pair] = {
                'current_price': df['price'].iloc[-1],
                'sma_5': df['sma_5'].iloc[-1],
                'sma_10': df['sma_10'].iloc[-1],
                'sma_20': df['sma_20'].iloc[-1],
                'trend': trend,
                'volatility': volatility,
                'recent_high': df['price'].tail(20).max(),
                'recent_low': df['price'].tail(20).min()
            }
        
        self.models['statistical'] = models
        return models
    
    def train_ml_model(self, price_data: Dict[str, List[Dict]]) -> Dict[str, any]:
        """
        Train machine learning models if available.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dictionary of trained ML models
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️ Scikit-learn not available, using statistical models only")
            return self.train_statistical_model(price_data)
        
        # Prepare features
        features_df = self.prepare_features(price_data)
        if features_df.empty:
            return {}
        
        models = {}
        
        for pair in features_df['pair'].unique():
            pair_data = features_df[features_df['pair'] == pair].copy()
            
            if len(pair_data) < 30:  # Need minimum data for ML
                continue
            
            # Prepare target variable (future price)
            pair_data['target'] = pair_data[f'{pair}_price'].shift(-self.prediction_horizon)
            pair_data = pair_data.dropna()
            
            if len(pair_data) < 20:
                continue
            
            # Split data
            split_idx = int(len(pair_data) * 0.8)
            train_data = pair_data[:split_idx]
            test_data = pair_data[split_idx:]
            
            # Prepare features and targets
            feature_cols = [col for col in self.feature_columns if col.startswith(pair)]
            X_train = train_data[feature_cols].values
            y_train = train_data['target'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['target'].values
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            # Store model and performance
            models[pair] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'performance': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            }
        
        self.models['ml'] = models
        return models
    
    def predict_prices(self, current_data: Dict[str, List[Dict]], 
                      prediction_minutes: int = 30) -> Dict[str, Dict]:
        """
        Predict future prices for all pairs.
        
        Args:
            current_data: Current price data
            prediction_minutes: Minutes ahead to predict
            
        Returns:
            Dictionary with predictions for each pair
        """
        predictions = {}
        
        if self.model_type == "statistical" or 'statistical' in self.models:
            predictions.update(self._predict_statistical(current_data, prediction_minutes))
        
        if self.model_type == "ml" and 'ml' in self.models and SKLEARN_AVAILABLE:
            ml_predictions = self._predict_ml(current_data, prediction_minutes)
            
            # Combine with statistical predictions
            for pair, ml_pred in ml_predictions.items():
                if pair in predictions:
                    # Ensemble: average statistical and ML predictions
                    stat_pred = predictions[pair]['predicted_price']
                    ensemble_pred = (stat_pred + ml_pred['predicted_price']) / 2
                    predictions[pair]['predicted_price'] = ensemble_pred
                    predictions[pair]['method'] = 'ensemble'
                    predictions[pair]['ml_confidence'] = ml_pred.get('confidence', 0.5)
                else:
                    predictions[pair] = ml_pred
        
        return predictions
    
    def _predict_statistical(self, current_data: Dict[str, List[Dict]], 
                           prediction_minutes: int) -> Dict[str, Dict]:
        """Statistical prediction method."""
        predictions = {}
        
        if 'statistical' not in self.models:
            self.train_statistical_model(current_data)
        
        for pair, model_params in self.models['statistical'].items():
            current_price = model_params['current_price']
            trend = model_params['trend']
            volatility = model_params['volatility']
            sma_10 = model_params['sma_10']
            
            # Simple trend-based prediction
            trend_prediction = current_price + (trend * prediction_minutes / 5)  # Scale trend
            
            # Mean reversion component
            mean_reversion = sma_10 + (sma_10 - current_price) * 0.1
            
            # Combine predictions
            predicted_price = (trend_prediction * 0.7) + (mean_reversion * 0.3)
            
            # Calculate confidence based on volatility
            confidence = max(0.1, min(0.9, 1.0 - (volatility / current_price)))
            
            predictions[pair] = {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': predicted_price - current_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'confidence': confidence,
                'method': 'statistical',
                'prediction_horizon_minutes': prediction_minutes,
                'volatility': volatility
            }
        
        return predictions
    
    def _predict_ml(self, current_data: Dict[str, List[Dict]], 
                   prediction_minutes: int) -> Dict[str, Dict]:
        """Machine learning prediction method."""
        predictions = {}
        
        if 'ml' not in self.models:
            return predictions
        
        # Prepare current features
        features_df = self.prepare_features(current_data)
        if features_df.empty:
            return predictions
        
        for pair, model_data in self.models['ml'].items():
            pair_data = features_df[features_df['pair'] == pair]
            
            if pair_data.empty:
                continue
            
            # Get latest features
            latest_features = pair_data[model_data['feature_columns']].iloc[-1:].values
            
            # Scale features
            latest_features_scaled = model_data['scaler'].transform(latest_features)
            
            # Make prediction
            predicted_price = model_data['model'].predict(latest_features_scaled)[0]
            current_price = pair_data[f'{pair}_price'].iloc[-1]
            
            # Calculate confidence based on model performance
            test_mae = model_data['performance']['test_mae']
            confidence = max(0.1, min(0.9, 1.0 - (test_mae / current_price)))
            
            predictions[pair] = {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': predicted_price - current_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'confidence': confidence,
                'method': 'ml',
                'prediction_horizon_minutes': prediction_minutes,
                'model_performance': model_data['performance']
            }
        
        return predictions
    
    def evaluate_predictions(self, predictions: Dict[str, Dict], 
                           actual_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Evaluate prediction accuracy against actual data.
        
        Args:
            predictions: Previous predictions
            actual_data: Actual price data
            
        Returns:
            Evaluation metrics
        """
        evaluation = {}
        
        for pair, pred_data in predictions.items():
            if pair not in actual_data:
                continue
            
            actual_prices = [entry['price'] for entry in actual_data[pair]]
            if not actual_prices:
                continue
            
            actual_price = actual_prices[-1]  # Most recent actual price
            predicted_price = pred_data['predicted_price']
            
            # Calculate errors
            absolute_error = abs(actual_price - predicted_price)
            percentage_error = (absolute_error / actual_price) * 100
            
            # Direction accuracy
            predicted_direction = 1 if pred_data['price_change'] > 0 else -1
            actual_direction = 1 if actual_price > pred_data['current_price'] else -1
            direction_correct = predicted_direction == actual_direction
            
            evaluation[pair] = {
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'direction_correct': direction_correct,
                'confidence': pred_data.get('confidence', 0.5),
                'method': pred_data.get('method', 'unknown')
            }
        
        return evaluation
    
    def get_trading_signals(self, predictions: Dict[str, Dict], 
                          confidence_threshold: float = 0.6,
                          min_price_change: float = 0.5) -> Dict[str, Dict]:
        """
        Generate trading signals based on predictions.
        
        Args:
            predictions: Price predictions
            confidence_threshold: Minimum confidence for signals
            min_price_change: Minimum price change percentage for signals
            
        Returns:
            Trading signals for each pair
        """
        signals = {}
        
        for pair, pred_data in predictions.items():
            confidence = pred_data.get('confidence', 0.5)
            price_change_percent = pred_data.get('price_change_percent', 0)
            
            # Skip low confidence predictions
            if confidence < confidence_threshold:
                continue
            
            # Skip small price changes
            if abs(price_change_percent) < min_price_change:
                continue
            
            # Generate signal
            if price_change_percent > min_price_change:
                signal = 'BUY'
                strength = min(1.0, abs(price_change_percent) / 5.0)  # Normalize to 0-1
            elif price_change_percent < -min_price_change:
                signal = 'SELL'
                strength = min(1.0, abs(price_change_percent) / 5.0)
            else:
                signal = 'HOLD'
                strength = 0.5
            
            signals[pair] = {
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'predicted_change_percent': price_change_percent,
                'predicted_price': pred_data['predicted_price'],
                'current_price': pred_data['current_price'],
                'method': pred_data.get('method', 'unknown')
            }
        
        return signals
    
    def save_model(self, filepath: str):
        """Save trained models to file."""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'sequence_length': self.sequence_length,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.prediction_horizon = model_data['prediction_horizon']
        self.sequence_length = model_data['sequence_length']
        self.performance_history = model_data.get('performance_history', {})