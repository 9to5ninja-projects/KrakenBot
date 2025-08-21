"""
Advanced Technical Indicators for KrakenBot
Implements Bollinger Bands, MACD, RSI, Stochastic, and other statistical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TechnicalSignal:
    """Represents a technical analysis signal."""
    indicator: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-100
    confidence: float  # 0-100
    timestamp: datetime
    price: float
    details: Dict

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators."""
    
    def __init__(self):
        self.indicators = {}
        self.signals = []
        
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Calculate position within bands
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        # Generate signals
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        current_position = bb_position.iloc[-1]
        
        # Signal logic
        if current_price <= current_lower:
            signal = "BUY"
            strength = min(100, (current_lower - current_price) / current_lower * 100 * 10)
        elif current_price >= current_upper:
            signal = "SELL"
            strength = min(100, (current_price - current_upper) / current_upper * 100 * 10)
        else:
            signal = "HOLD"
            strength = abs(current_position - 0.5) * 100
        
        return {
            'upper_band': current_upper,
            'lower_band': current_lower,
            'sma': current_sma,
            'position': current_position,
            'signal': signal,
            'strength': strength,
            'confidence': min(95, window * 2),  # Higher confidence with more data
            'squeeze': (current_upper - current_lower) / current_sma < 0.1  # Low volatility
        }
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal_line: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line_values = macd_line.ewm(span=signal_line).mean()
        histogram = macd_line - signal_line_values
        
        # Current values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line_values.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Previous values for trend detection
        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        prev_signal = signal_line_values.iloc[-2] if len(signal_line_values) > 1 else current_signal
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else current_histogram
        
        # Signal logic
        if current_macd > current_signal and prev_macd <= prev_signal:
            signal = "BUY"
            strength = min(100, abs(current_macd - current_signal) * 1000)
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signal = "SELL"
            strength = min(100, abs(current_macd - current_signal) * 1000)
        else:
            signal = "HOLD"
            strength = abs(current_histogram) * 100
        
        return {
            'macd': current_macd,
            'signal_line': current_signal,
            'histogram': current_histogram,
            'signal': signal,
            'strength': strength,
            'confidence': min(90, len(prices) / 2),
            'bullish_crossover': current_macd > current_signal and prev_macd <= prev_signal,
            'bearish_crossover': current_macd < current_signal and prev_macd >= prev_signal
        }
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> Dict:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Signal logic
        if current_rsi <= 30:
            signal = "BUY"
            strength = (30 - current_rsi) * 3.33  # Scale to 0-100
        elif current_rsi >= 70:
            signal = "SELL"
            strength = (current_rsi - 70) * 3.33  # Scale to 0-100
        else:
            signal = "HOLD"
            strength = abs(current_rsi - 50) * 2  # Distance from neutral
        
        return {
            'rsi': current_rsi,
            'signal': signal,
            'strength': min(100, strength),
            'confidence': min(85, window * 3),
            'oversold': current_rsi <= 30,
            'overbought': current_rsi >= 70,
            'divergence': self._detect_rsi_divergence(prices, rsi)
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        # Signal logic
        if current_k <= 20 and current_d <= 20:
            signal = "BUY"
            strength = (20 - min(current_k, current_d)) * 5
        elif current_k >= 80 and current_d >= 80:
            signal = "SELL"
            strength = (max(current_k, current_d) - 80) * 5
        else:
            signal = "HOLD"
            strength = abs(current_k - current_d) * 2
        
        return {
            'k_percent': current_k,
            'd_percent': current_d,
            'signal': signal,
            'strength': min(100, strength),
            'confidence': min(80, k_window * 2),
            'oversold': current_k <= 20 and current_d <= 20,
            'overbought': current_k >= 80 and current_d >= 80,
            'bullish_crossover': current_k > current_d,
            'bearish_crossover': current_k < current_d
        }
    
    def calculate_volume_indicators(self, prices: pd.Series, volumes: pd.Series) -> Dict:
        """Calculate volume-based indicators."""
        # Volume Weighted Average Price (VWAP)
        vwap = (prices * volumes).cumsum() / volumes.cumsum()
        
        # On Balance Volume (OBV)
        obv = np.where(prices.diff() > 0, volumes, 
                      np.where(prices.diff() < 0, -volumes, 0)).cumsum()
        
        # Volume Rate of Change
        volume_roc = volumes.pct_change(periods=10) * 100
        
        current_price = prices.iloc[-1]
        current_vwap = vwap.iloc[-1]
        current_obv = obv[-1]
        current_vol_roc = volume_roc.iloc[-1]
        
        # Signal logic based on VWAP
        if current_price > current_vwap * 1.002:  # 0.2% above VWAP
            signal = "SELL"
            strength = min(100, ((current_price - current_vwap) / current_vwap) * 1000)
        elif current_price < current_vwap * 0.998:  # 0.2% below VWAP
            signal = "BUY"
            strength = min(100, ((current_vwap - current_price) / current_vwap) * 1000)
        else:
            signal = "HOLD"
            strength = abs((current_price - current_vwap) / current_vwap) * 500
        
        return {
            'vwap': current_vwap,
            'obv': current_obv,
            'volume_roc': current_vol_roc,
            'signal': signal,
            'strength': strength,
            'confidence': 75,
            'price_above_vwap': current_price > current_vwap,
            'volume_trend': 'increasing' if current_vol_roc > 5 else 'decreasing' if current_vol_roc < -5 else 'stable'
        }
    
    def _detect_rsi_divergence(self, prices: pd.Series, rsi: pd.Series, window: int = 5) -> bool:
        """Detect RSI divergence patterns."""
        if len(prices) < window * 2:
            return False
        
        recent_prices = prices.tail(window)
        recent_rsi = rsi.tail(window)
        
        price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]
        
        return price_trend != rsi_trend  # Divergence detected
    
    def analyze_pair(self, price_data: pd.DataFrame, pair: str) -> Dict:
        """Comprehensive technical analysis for a trading pair."""
        if price_data.empty:
            return {'error': 'No price data available'}
        
        # Ensure we have OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in price_data.columns for col in required_columns):
            # If we only have close prices, create synthetic OHLC
            if 'close' in price_data.columns:
                price_data['open'] = price_data['close'].shift(1)
                price_data['high'] = price_data['close']
                price_data['low'] = price_data['close']
                price_data['volume'] = 1000  # Default volume
            else:
                return {'error': 'Insufficient price data'}
        
        close_prices = price_data['close']
        high_prices = price_data['high']
        low_prices = price_data['low']
        volumes = price_data['volume']
        
        analysis = {
            'pair': pair,
            'timestamp': datetime.now(),
            'current_price': close_prices.iloc[-1],
            'indicators': {}
        }
        
        try:
            # Calculate all indicators
            analysis['indicators']['bollinger'] = self.calculate_bollinger_bands(close_prices)
            analysis['indicators']['macd'] = self.calculate_macd(close_prices)
            analysis['indicators']['rsi'] = self.calculate_rsi(close_prices)
            analysis['indicators']['stochastic'] = self.calculate_stochastic(high_prices, low_prices, close_prices)
            analysis['indicators']['volume'] = self.calculate_volume_indicators(close_prices, volumes)
            
            # Generate composite signal
            analysis['composite_signal'] = self._generate_composite_signal(analysis['indicators'])
            
        except Exception as e:
            analysis['error'] = f"Analysis error: {str(e)}"
        
        return analysis
    
    def _generate_composite_signal(self, indicators: Dict) -> Dict:
        """Generate a composite signal from all indicators."""
        signals = []
        total_strength = 0
        total_confidence = 0
        
        for indicator_name, indicator_data in indicators.items():
            if 'signal' in indicator_data:
                signal = indicator_data['signal']
                strength = indicator_data.get('strength', 0)
                confidence = indicator_data.get('confidence', 0)
                
                # Weight the signal
                if signal == 'BUY':
                    signals.append(-strength * confidence / 100)  # Negative for buy
                elif signal == 'SELL':
                    signals.append(strength * confidence / 100)   # Positive for sell
                else:
                    signals.append(0)  # Neutral for hold
                
                total_strength += strength
                total_confidence += confidence
        
        if not signals:
            return {'signal': 'HOLD', 'strength': 0, 'confidence': 0}
        
        # Calculate composite values
        composite_score = sum(signals) / len(signals)
        avg_strength = total_strength / len(signals)
        avg_confidence = total_confidence / len(signals)
        
        # Determine final signal
        if composite_score < -20:
            final_signal = 'BUY'
            final_strength = min(100, abs(composite_score))
        elif composite_score > 20:
            final_signal = 'SELL'
            final_strength = min(100, abs(composite_score))
        else:
            final_signal = 'HOLD'
            final_strength = abs(composite_score)
        
        return {
            'signal': final_signal,
            'strength': final_strength,
            'confidence': min(100, avg_confidence),
            'composite_score': composite_score,
            'agreement_level': self._calculate_agreement_level(indicators),
            'recommendation': self._generate_recommendation(final_signal, final_strength, avg_confidence)
        }
    
    def _calculate_agreement_level(self, indicators: Dict) -> float:
        """Calculate how much the indicators agree with each other."""
        signals = []
        for indicator_data in indicators.values():
            if 'signal' in indicator_data:
                signal = indicator_data['signal']
                if signal == 'BUY':
                    signals.append(-1)
                elif signal == 'SELL':
                    signals.append(1)
                else:
                    signals.append(0)
        
        if not signals:
            return 0
        
        # Calculate standard deviation (lower = more agreement)
        std_dev = np.std(signals)
        agreement = max(0, 100 - (std_dev * 50))  # Convert to 0-100 scale
        
        return agreement
    
    def _generate_recommendation(self, signal: str, strength: float, confidence: float) -> str:
        """Generate a human-readable recommendation."""
        if signal == 'BUY':
            if strength > 70 and confidence > 80:
                return "Strong Buy - Multiple indicators confirm oversold conditions"
            elif strength > 50:
                return "Buy - Technical indicators suggest good entry point"
            else:
                return "Weak Buy - Some indicators suggest buying opportunity"
        elif signal == 'SELL':
            if strength > 70 and confidence > 80:
                return "Strong Sell - Multiple indicators confirm overbought conditions"
            elif strength > 50:
                return "Sell - Technical indicators suggest good exit point"
            else:
                return "Weak Sell - Some indicators suggest selling opportunity"
        else:
            return "Hold - Mixed signals, wait for clearer direction"

def create_sample_price_data(pair: str, days: int = 30) -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1H')
    
    # Generate realistic price movements
    np.random.seed(42)
    base_price = 6000 if 'ETH' in pair else 160000 if 'BTC' in pair else 1000
    
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    return df

if __name__ == "__main__":
    # Test the technical analysis
    analyzer = AdvancedTechnicalAnalyzer()
    
    # Test with sample data
    sample_data = create_sample_price_data("ETH/CAD")
    analysis = analyzer.analyze_pair(sample_data, "ETH/CAD")
    
    print("🔍 Advanced Technical Analysis Test")
    print("=" * 50)
    print(f"Pair: {analysis['pair']}")
    print(f"Current Price: ${analysis['current_price']:.2f}")
    
    if 'composite_signal' in analysis:
        composite = analysis['composite_signal']
        print(f"\n🎯 COMPOSITE SIGNAL: {composite['signal']}")
        print(f"Strength: {composite['strength']:.1f}/100")
        print(f"Confidence: {composite['confidence']:.1f}/100")
        print(f"Agreement: {composite['agreement_level']:.1f}/100")
        print(f"Recommendation: {composite['recommendation']}")
    
    print(f"\n📊 INDIVIDUAL INDICATORS:")
    for name, indicator in analysis['indicators'].items():
        print(f"{name.upper()}: {indicator['signal']} (Strength: {indicator['strength']:.1f})")