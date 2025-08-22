"""
Simple Technical Indicators for Emergency Data Collection
"""

import pandas as pd
import numpy as np

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators for a DataFrame"""
    
    if df.empty or len(df) < 20:
        return df
    
    try:
        # Make a copy to avoid modifying original
        result = df.copy()
        
        # RSI
        result['rsi'] = calculate_rsi(result['close'])
        
        # MACD
        macd_data = calculate_macd(result['close'])
        result['macd'] = macd_data['macd']
        result['macd_signal'] = macd_data['signal']
        result['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(result['close'])
        result['bb_upper'] = bb_data['upper']
        result['bb_middle'] = bb_data['middle']
        result['bb_lower'] = bb_data['lower']
        result['bb_position'] = bb_data['position']
        
        # Stochastic
        stoch_data = calculate_stochastic(result['high'], result['low'], result['close'])
        result['stoch_k'] = stoch_data['k']
        result['stoch_d'] = stoch_data['d']
        
        # Williams %R
        result['williams_r'] = calculate_williams_r(result['high'], result['low'], result['close'])
        
        # CCI
        result['cci'] = calculate_cci(result['high'], result['low'], result['close'])
        
        # ATR
        result['atr'] = calculate_atr(result['high'], result['low'], result['close'])
        
        return result
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        histogram = macd - macd_signal
        
        return {
            'macd': macd.fillna(0),
            'signal': macd_signal.fillna(0),
            'histogram': histogram.fillna(0)
        }
    except:
        return {
            'macd': pd.Series([0] * len(prices), index=prices.index),
            'signal': pd.Series([0] * len(prices), index=prices.index),
            'histogram': pd.Series([0] * len(prices), index=prices.index)
        }

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> dict:
    """Calculate Bollinger Bands"""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        position = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper.fillna(prices),
            'middle': sma.fillna(prices),
            'lower': lower.fillna(prices),
            'position': position.fillna(0.5)
        }
    except:
        return {
            'upper': prices,
            'middle': prices,
            'lower': prices,
            'position': pd.Series([0.5] * len(prices), index=prices.index)
        }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> dict:
    """Calculate Stochastic Oscillator"""
    try:
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent.fillna(50),
            'd': d_percent.fillna(50)
        }
    except:
        return {
            'k': pd.Series([50] * len(close), index=close.index),
            'd': pd.Series([50] * len(close), index=close.index)
        }

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    try:
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r.fillna(-50)
    except:
        return pd.Series([-50] * len(close), index=close.index)

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index"""
    try:
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci.fillna(0)
    except:
        return pd.Series([0] * len(close), index=close.index)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    try:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(0)
    except:
        return pd.Series([0] * len(close), index=close.index)