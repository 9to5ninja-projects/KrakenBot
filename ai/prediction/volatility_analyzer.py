"""
Volatility Analysis and Capitalization System
Detects volatility patterns and generates trading opportunities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class VolatilityAnalyzer:
    """
    Advanced volatility analysis system for detecting and capitalizing on market volatility.
    """
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize volatility analyzer.
        
        Args:
            lookback_periods: Number of periods to look back for volatility calculation
        """
        self.lookback_periods = lookback_periods
        self.volatility_history = {}
        self.volatility_patterns = {}
        self.opportunity_threshold = 1.5  # Volatility must be 1.5x normal to trigger
        
    def calculate_volatility_metrics(self, price_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Calculate comprehensive volatility metrics for all pairs.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dictionary with volatility metrics for each pair
        """
        volatility_metrics = {}
        
        for pair, price_history in price_data.items():
            if not isinstance(price_history, list) or len(price_history) < self.lookback_periods:
                continue
                
            df = pd.DataFrame(price_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate returns
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            
            # Basic volatility measures
            current_volatility = df['returns'].tail(self.lookback_periods).std()
            historical_volatility = df['returns'].std()
            
            # Rolling volatility
            df['rolling_vol'] = df['returns'].rolling(window=self.lookback_periods, min_periods=5).std()
            
            # Volatility of volatility
            vol_of_vol = df['rolling_vol'].std()
            
            # Price range volatility
            df['high_low_range'] = (df['price'].rolling(window=5).max() - 
                                   df['price'].rolling(window=5).min()) / df['price']
            range_volatility = df['high_low_range'].mean()
            
            # Volatility clustering detection
            df['vol_squared'] = df['returns'] ** 2
            df['vol_ma'] = df['vol_squared'].rolling(window=10).mean()
            clustering_score = df['vol_ma'].tail(10).std()
            
            # Volatility regime detection
            recent_vol = df['rolling_vol'].tail(10).mean()
            long_term_vol = df['rolling_vol'].mean()
            volatility_regime = self._classify_volatility_regime(recent_vol, long_term_vol)
            
            # Volatility breakout detection
            vol_percentile = self._calculate_volatility_percentile(df['rolling_vol'])
            
            # Intraday volatility patterns
            intraday_patterns = self._analyze_intraday_volatility(df)
            
            volatility_metrics[pair] = {
                'current_volatility': current_volatility,
                'historical_volatility': historical_volatility,
                'volatility_ratio': current_volatility / historical_volatility if historical_volatility > 0 else 1,
                'vol_of_vol': vol_of_vol,
                'range_volatility': range_volatility,
                'clustering_score': clustering_score,
                'volatility_regime': volatility_regime,
                'volatility_percentile': vol_percentile,
                'intraday_patterns': intraday_patterns,
                'current_price': df['price'].iloc[-1],
                'price_trend': self._calculate_price_trend(df['price']),
                'volatility_trend': self._calculate_volatility_trend(df['rolling_vol'])
            }
        
        return volatility_metrics
    
    def _classify_volatility_regime(self, recent_vol: float, long_term_vol: float) -> str:
        """Classify current volatility regime."""
        ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1
        
        if ratio > 1.5:
            return "high_volatility"
        elif ratio < 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def _calculate_volatility_percentile(self, volatility_series: pd.Series) -> float:
        """Calculate current volatility percentile."""
        current_vol = volatility_series.iloc[-1]
        return (volatility_series <= current_vol).mean() * 100
    
    def _analyze_intraday_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze intraday volatility patterns."""
        if len(df) < 24:  # Need at least 24 periods
            return {'pattern': 'insufficient_data'}
        
        # Extract hour from timestamp
        df['hour'] = df['timestamp'].dt.hour
        
        # Calculate average volatility by hour
        hourly_vol = df.groupby('hour')['returns'].std().to_dict()
        
        # Find peak volatility hours
        if hourly_vol:
            peak_hour = max(hourly_vol, key=hourly_vol.get)
            low_hour = min(hourly_vol, key=hourly_vol.get)
            
            return {
                'pattern': 'hourly_analysis',
                'peak_volatility_hour': peak_hour,
                'low_volatility_hour': low_hour,
                'hourly_volatility': hourly_vol,
                'volatility_range': max(hourly_vol.values()) - min(hourly_vol.values())
            }
        
        return {'pattern': 'no_pattern'}
    
    def _calculate_price_trend(self, price_series: pd.Series) -> str:
        """Calculate price trend direction."""
        if len(price_series) < 10:
            return "unknown"
        
        recent_prices = price_series.tail(10)
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if trend_slope > price_series.iloc[-1] * 0.001:  # 0.1% threshold
            return "uptrend"
        elif trend_slope < -price_series.iloc[-1] * 0.001:
            return "downtrend"
        else:
            return "sideways"
    
    def _calculate_volatility_trend(self, volatility_series: pd.Series) -> str:
        """Calculate volatility trend direction."""
        if len(volatility_series) < 10:
            return "unknown"
        
        recent_vol = volatility_series.tail(10).dropna()
        if len(recent_vol) < 5:
            return "unknown"
        
        vol_slope = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
        
        if vol_slope > 0:
            return "increasing"
        elif vol_slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def detect_volatility_opportunities(self, volatility_metrics: Dict[str, Dict]) -> List[Dict]:
        """
        Detect trading opportunities based on volatility analysis.
        
        Args:
            volatility_metrics: Volatility metrics from calculate_volatility_metrics
            
        Returns:
            List of volatility-based trading opportunities
        """
        opportunities = []
        
        for pair, metrics in volatility_metrics.items():
            volatility_ratio = metrics['volatility_ratio']
            volatility_regime = metrics['volatility_regime']
            volatility_percentile = metrics['volatility_percentile']
            price_trend = metrics['price_trend']
            volatility_trend = metrics['volatility_trend']
            
            # High volatility breakout opportunity
            if (volatility_ratio > self.opportunity_threshold and 
                volatility_percentile > 80 and 
                volatility_trend == "increasing"):
                
                opportunities.append({
                    'pair': pair,
                    'opportunity_type': 'volatility_breakout',
                    'signal': 'high_volatility_detected',
                    'strategy': 'momentum_following',
                    'confidence': min(0.9, volatility_ratio / 3.0),
                    'expected_duration': '15-30 minutes',
                    'risk_level': 'high',
                    'details': {
                        'volatility_ratio': volatility_ratio,
                        'volatility_percentile': volatility_percentile,
                        'price_trend': price_trend,
                        'current_price': metrics['current_price']
                    }
                })
            
            # Volatility compression opportunity
            elif (volatility_ratio < 0.7 and 
                  volatility_percentile < 20 and 
                  volatility_trend == "decreasing"):
                
                opportunities.append({
                    'pair': pair,
                    'opportunity_type': 'volatility_compression',
                    'signal': 'low_volatility_compression',
                    'strategy': 'breakout_anticipation',
                    'confidence': 0.6,
                    'expected_duration': '30-60 minutes',
                    'risk_level': 'medium',
                    'details': {
                        'volatility_ratio': volatility_ratio,
                        'volatility_percentile': volatility_percentile,
                        'price_trend': price_trend,
                        'current_price': metrics['current_price']
                    }
                })
            
            # Mean reversion opportunity
            elif (volatility_regime == "high_volatility" and 
                  price_trend in ["uptrend", "downtrend"] and
                  volatility_trend == "decreasing"):
                
                opportunities.append({
                    'pair': pair,
                    'opportunity_type': 'mean_reversion',
                    'signal': 'volatility_mean_reversion',
                    'strategy': 'contrarian',
                    'confidence': 0.7,
                    'expected_duration': '20-45 minutes',
                    'risk_level': 'medium',
                    'details': {
                        'volatility_ratio': volatility_ratio,
                        'price_trend': price_trend,
                        'volatility_trend': volatility_trend,
                        'current_price': metrics['current_price']
                    }
                })
            
            # Volatility clustering opportunity
            elif metrics['clustering_score'] > metrics['historical_volatility'] * 2:
                opportunities.append({
                    'pair': pair,
                    'opportunity_type': 'volatility_clustering',
                    'signal': 'volatility_clustering_detected',
                    'strategy': 'volatility_trading',
                    'confidence': 0.65,
                    'expected_duration': '10-25 minutes',
                    'risk_level': 'high',
                    'details': {
                        'clustering_score': metrics['clustering_score'],
                        'volatility_regime': volatility_regime,
                        'current_price': metrics['current_price']
                    }
                })
        
        # Sort opportunities by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    def generate_volatility_trading_strategy(self, opportunity: Dict) -> Dict:
        """
        Generate specific trading strategy for volatility opportunity.
        
        Args:
            opportunity: Volatility opportunity from detect_volatility_opportunities
            
        Returns:
            Detailed trading strategy
        """
        pair = opportunity['pair']
        opp_type = opportunity['opportunity_type']
        current_price = opportunity['details']['current_price']
        
        strategy = {
            'pair': pair,
            'opportunity_type': opp_type,
            'entry_strategy': {},
            'exit_strategy': {},
            'risk_management': {},
            'position_sizing': {}
        }
        
        if opp_type == 'volatility_breakout':
            # High volatility momentum strategy
            strategy['entry_strategy'] = {
                'method': 'momentum_breakout',
                'entry_condition': 'price_movement > 0.5% in 5 minutes',
                'entry_price_range': f"{current_price * 0.995:.2f} - {current_price * 1.005:.2f}",
                'max_wait_time': '10 minutes'
            }
            
            strategy['exit_strategy'] = {
                'profit_target': f"{current_price * 1.015:.2f}",  # 1.5% profit target
                'stop_loss': f"{current_price * 0.992:.2f}",      # 0.8% stop loss
                'time_exit': '30 minutes maximum',
                'volatility_exit': 'exit if volatility drops below 50th percentile'
            }
            
            strategy['risk_management'] = {
                'max_position_size': '25% of portfolio',
                'volatility_stop': 'exit if volatility ratio drops below 1.2',
                'drawdown_limit': '2% of portfolio value'
            }
        
        elif opp_type == 'volatility_compression':
            # Low volatility breakout anticipation
            strategy['entry_strategy'] = {
                'method': 'breakout_anticipation',
                'entry_condition': 'wait for volatility increase or price breakout',
                'entry_price_range': f"{current_price * 0.998:.2f} - {current_price * 1.002:.2f}",
                'max_wait_time': '60 minutes'
            }
            
            strategy['exit_strategy'] = {
                'profit_target': f"{current_price * 1.01:.2f}",   # 1% profit target
                'stop_loss': f"{current_price * 0.995:.2f}",      # 0.5% stop loss
                'time_exit': '45 minutes maximum',
                'breakout_exit': 'exit on strong directional move'
            }
            
            strategy['risk_management'] = {
                'max_position_size': '30% of portfolio',
                'patience_required': 'wait for clear breakout signal',
                'false_breakout_protection': 'confirm with volume if available'
            }
        
        elif opp_type == 'mean_reversion':
            # Mean reversion strategy
            strategy['entry_strategy'] = {
                'method': 'contrarian_entry',
                'entry_condition': 'price reversal signal after high volatility',
                'entry_price_range': f"{current_price * 0.997:.2f} - {current_price * 1.003:.2f}",
                'max_wait_time': '20 minutes'
            }
            
            strategy['exit_strategy'] = {
                'profit_target': f"{current_price * 1.008:.2f}",  # 0.8% profit target
                'stop_loss': f"{current_price * 0.994:.2f}",      # 0.6% stop loss
                'time_exit': '40 minutes maximum',
                'trend_exit': 'exit if trend continues against position'
            }
            
            strategy['risk_management'] = {
                'max_position_size': '20% of portfolio',
                'trend_confirmation': 'ensure trend is weakening',
                'volatility_confirmation': 'confirm volatility is decreasing'
            }
        
        elif opp_type == 'volatility_clustering':
            # Volatility clustering strategy
            strategy['entry_strategy'] = {
                'method': 'volatility_momentum',
                'entry_condition': 'continued high volatility with directional bias',
                'entry_price_range': f"{current_price * 0.996:.2f} - {current_price * 1.004:.2f}",
                'max_wait_time': '15 minutes'
            }
            
            strategy['exit_strategy'] = {
                'profit_target': f"{current_price * 1.012:.2f}",  # 1.2% profit target
                'stop_loss': f"{current_price * 0.993:.2f}",      # 0.7% stop loss
                'time_exit': '25 minutes maximum',
                'clustering_exit': 'exit when volatility clustering ends'
            }
            
            strategy['risk_management'] = {
                'max_position_size': '22% of portfolio',
                'clustering_monitoring': 'monitor for end of clustering pattern',
                'quick_exit': 'be ready for rapid position changes'
            }
        
        # Add general position sizing
        strategy['position_sizing'] = {
            'base_size': '15% of portfolio',
            'confidence_multiplier': opportunity['confidence'],
            'volatility_adjustment': 'reduce size if volatility > 3x normal',
            'recommended_size': f"{15 * opportunity['confidence']:.1f}% of portfolio"
        }
        
        return strategy
    
    def calculate_volatility_score(self, volatility_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate overall volatility score for each pair.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Dictionary with volatility scores (0-100)
        """
        scores = {}
        
        for pair, metrics in volatility_metrics.items():
            # Components of volatility score
            volatility_ratio_score = min(100, metrics['volatility_ratio'] * 50)
            percentile_score = metrics['volatility_percentile']
            clustering_score = min(100, metrics['clustering_score'] * 1000)  # Scale clustering
            
            # Regime bonus
            regime_bonus = {
                'high_volatility': 20,
                'normal_volatility': 0,
                'low_volatility': -10
            }.get(metrics['volatility_regime'], 0)
            
            # Trend bonus
            trend_bonus = 0
            if metrics['volatility_trend'] == 'increasing':
                trend_bonus = 15
            elif metrics['volatility_trend'] == 'decreasing':
                trend_bonus = -5
            
            # Calculate final score
            raw_score = (volatility_ratio_score * 0.3 + 
                        percentile_score * 0.3 + 
                        clustering_score * 0.2 + 
                        regime_bonus + 
                        trend_bonus)
            
            # Normalize to 0-100
            scores[pair] = max(0, min(100, raw_score))
        
        return scores
    
    def get_volatility_summary(self, volatility_metrics: Dict[str, Dict]) -> Dict:
        """
        Generate summary of volatility analysis.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Summary dictionary
        """
        if not volatility_metrics:
            return {'status': 'no_data'}
        
        # Calculate aggregate metrics
        all_ratios = [m['volatility_ratio'] for m in volatility_metrics.values()]
        all_percentiles = [m['volatility_percentile'] for m in volatility_metrics.values()]
        
        # Count regimes
        regimes = [m['volatility_regime'] for m in volatility_metrics.values()]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        
        # Get volatility scores
        scores = self.calculate_volatility_score(volatility_metrics)
        
        # Find highest volatility pair
        highest_vol_pair = max(volatility_metrics.keys(), 
                              key=lambda x: volatility_metrics[x]['volatility_ratio'])
        
        summary = {
            'total_pairs_analyzed': len(volatility_metrics),
            'average_volatility_ratio': np.mean(all_ratios),
            'average_volatility_percentile': np.mean(all_percentiles),
            'volatility_regime_distribution': regime_counts,
            'highest_volatility_pair': {
                'pair': highest_vol_pair,
                'volatility_ratio': volatility_metrics[highest_vol_pair]['volatility_ratio'],
                'regime': volatility_metrics[highest_vol_pair]['volatility_regime']
            },
            'volatility_scores': scores,
            'market_volatility_state': self._assess_market_volatility_state(all_ratios, regime_counts),
            'trading_recommendation': self._generate_volatility_trading_recommendation(scores, regime_counts)
        }
        
        return summary
    
    def _assess_market_volatility_state(self, volatility_ratios: List[float], 
                                       regime_counts: Dict[str, int]) -> str:
        """Assess overall market volatility state."""
        avg_ratio = np.mean(volatility_ratios)
        high_vol_count = regime_counts.get('high_volatility', 0)
        total_pairs = sum(regime_counts.values())
        
        if avg_ratio > 1.3 or high_vol_count / total_pairs > 0.6:
            return "high_volatility_market"
        elif avg_ratio < 0.8 or regime_counts.get('low_volatility', 0) / total_pairs > 0.6:
            return "low_volatility_market"
        else:
            return "mixed_volatility_market"
    
    def _generate_volatility_trading_recommendation(self, scores: Dict[str, float], 
                                                   regime_counts: Dict[str, int]) -> str:
        """Generate overall trading recommendation based on volatility."""
        avg_score = np.mean(list(scores.values())) if scores else 0
        high_vol_pairs = sum(1 for score in scores.values() if score > 70)
        
        if avg_score > 60 and high_vol_pairs >= 2:
            return "active_volatility_trading_recommended"
        elif avg_score > 40:
            return "selective_volatility_trading"
        else:
            return "low_volatility_wait_for_opportunities"