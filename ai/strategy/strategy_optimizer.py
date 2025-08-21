"""
AI-Powered Strategy Optimization System
Analyzes trading performance and optimizes strategy parameters
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import itertools

class StrategyOptimizer:
    """
    Advanced strategy optimization system using AI analysis.
    """
    
    def __init__(self):
        """Initialize strategy optimizer."""
        self.optimization_history = []
        self.current_parameters = {}
        self.performance_metrics = {}
        self.optimization_results = {}
        
    def analyze_current_performance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current strategy performance.
        
        Args:
            session_data: Trading session data
            
        Returns:
            Performance analysis results
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'session_info': {},
            'performance_metrics': {},
            'parameter_effectiveness': {},
            'improvement_opportunities': [],
            'optimization_recommendations': []
        }
        
        # Extract session information
        metadata = session_data.get('metadata', {})
        portfolio_history = session_data.get('portfolio_history', [])
        trades = session_data.get('trades', [])
        price_history = session_data.get('price_history', {})
        
        # Session info
        analysis['session_info'] = {
            'session_id': metadata.get('session_id', 'unknown'),
            'duration_hours': self._calculate_session_duration(metadata),
            'trading_pairs': metadata.get('trading_pairs', []),
            'initial_balance': metadata.get('initial_balance', 0),
            'strategy_type': metadata.get('strategy', 'unknown')
        }
        
        # Performance metrics
        analysis['performance_metrics'] = self._calculate_performance_metrics(
            portfolio_history, trades, metadata.get('initial_balance', 100)
        )
        
        # Parameter effectiveness analysis
        current_params = metadata.get('parameters', {})
        analysis['parameter_effectiveness'] = self._analyze_parameter_effectiveness(
            current_params, trades, price_history, analysis['performance_metrics']
        )
        
        # Identify improvement opportunities
        analysis['improvement_opportunities'] = self._identify_improvement_opportunities(
            analysis['performance_metrics'], analysis['parameter_effectiveness'], trades, price_history
        )
        
        # Generate optimization recommendations
        analysis['optimization_recommendations'] = self._generate_optimization_recommendations(
            analysis['improvement_opportunities'], current_params
        )
        
        return analysis
    
    def _calculate_session_duration(self, metadata: Dict) -> float:
        """Calculate session duration in hours."""
        start_time_str = metadata.get('start_time', '')
        if not start_time_str:
            return 0
        
        try:
            start_time = datetime.fromisoformat(start_time_str)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 3600
        except:
            return 0
    
    def _calculate_performance_metrics(self, portfolio_history: List[Dict], 
                                     trades: List[Dict], initial_balance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not portfolio_history:
            return {
                'total_return_percent': 0,
                'total_trades': len(trades),
                'successful_trades': 0,
                'success_rate_percent': 0,
                'average_trade_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(portfolio_history)
        
        # Basic metrics
        final_value = df['portfolio_value'].iloc[-1] if not df.empty else initial_balance
        total_return = ((final_value - initial_balance) / initial_balance) * 100
        
        # Trade analysis
        successful_trades = len([t for t in trades if t.get('profit', 0) > 0])
        success_rate = (successful_trades / len(trades) * 100) if trades else 0
        
        # Calculate trade returns
        trade_returns = [t.get('profit_percent', 0) for t in trades if 'profit_percent' in t]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        # Drawdown calculation
        if not df.empty:
            df['cumulative_max'] = df['portfolio_value'].cummax()
            df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']
            max_drawdown = df['drawdown'].min() * 100
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if trade_returns and len(trade_returns) > 1:
            returns_std = np.std(trade_returns)
            sharpe_ratio = (avg_trade_return / returns_std) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        if winning_trades and losing_trades:
            profit_factor = sum(winning_trades) / abs(sum(losing_trades))
        else:
            profit_factor = 0
        
        return {
            'total_return_percent': total_return,
            'final_portfolio_value': final_value,
            'total_trades': len(trades),
            'successful_trades': successful_trades,
            'success_rate_percent': success_rate,
            'average_trade_return': avg_trade_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'trade_frequency': len(trades) / max(1, self._calculate_session_duration({'start_time': datetime.now().isoformat()}))
        }
    
    def _analyze_parameter_effectiveness(self, current_params: Dict, trades: List[Dict], 
                                       price_history: Dict, performance: Dict) -> Dict[str, Any]:
        """Analyze effectiveness of current parameters."""
        effectiveness = {}
        
        # Analyze threshold parameters
        buy_threshold = current_params.get('buy_threshold', -0.008)
        sell_threshold = current_params.get('sell_threshold', 0.012)
        
        # Count threshold hits vs actual trades
        threshold_analysis = self._analyze_threshold_effectiveness(
            buy_threshold, sell_threshold, trades, price_history
        )
        
        effectiveness['thresholds'] = {
            'buy_threshold': {
                'current_value': buy_threshold,
                'effectiveness_score': threshold_analysis['buy_effectiveness'],
                'hit_rate': threshold_analysis['buy_hit_rate'],
                'conversion_rate': threshold_analysis['buy_conversion_rate']
            },
            'sell_threshold': {
                'current_value': sell_threshold,
                'effectiveness_score': threshold_analysis['sell_effectiveness'],
                'hit_rate': threshold_analysis['sell_hit_rate'],
                'conversion_rate': threshold_analysis['sell_conversion_rate']
            }
        }
        
        # Analyze timing parameters
        lookback_periods = current_params.get('lookback_periods', 8)
        effectiveness['timing'] = {
            'lookback_periods': {
                'current_value': lookback_periods,
                'effectiveness_score': self._analyze_lookback_effectiveness(lookback_periods, trades, price_history),
                'optimal_range': self._suggest_lookback_range(trades, price_history)
            }
        }
        
        # Analyze position sizing
        max_position_size = current_params.get('max_position_size', 0.3)
        min_trade_amount = current_params.get('min_trade_amount', 25.0)
        
        effectiveness['position_sizing'] = {
            'max_position_size': {
                'current_value': max_position_size,
                'utilization_rate': self._calculate_position_utilization(trades, max_position_size),
                'effectiveness_score': self._analyze_position_size_effectiveness(trades, performance)
            },
            'min_trade_amount': {
                'current_value': min_trade_amount,
                'effectiveness_score': self._analyze_min_trade_effectiveness(trades, min_trade_amount)
            }
        }
        
        return effectiveness
    
    def _analyze_threshold_effectiveness(self, buy_threshold: float, sell_threshold: float,
                                       trades: List[Dict], price_history: Dict) -> Dict[str, float]:
        """Analyze how effective the current thresholds are."""
        analysis = {
            'buy_effectiveness': 0.5,
            'sell_effectiveness': 0.5,
            'buy_hit_rate': 0,
            'sell_hit_rate': 0,
            'buy_conversion_rate': 0,
            'sell_conversion_rate': 0
        }
        
        if not price_history:
            return analysis
        
        # Count threshold hits and conversions
        buy_signals = 0
        sell_signals = 0
        buy_trades = len([t for t in trades if t.get('action') == 'buy'])
        sell_trades = len([t for t in trades if t.get('action') == 'sell'])
        
        # Analyze price movements vs thresholds
        for pair, price_data in price_history.items():
            if len(price_data) < 2:
                continue
            
            for i in range(1, len(price_data)):
                current_price = price_data[i]['price']
                prev_price = price_data[i-1]['price']
                price_change = (current_price - prev_price) / prev_price
                
                if price_change <= buy_threshold:
                    buy_signals += 1
                elif price_change >= sell_threshold:
                    sell_signals += 1
        
        # Calculate rates
        if buy_signals > 0:
            analysis['buy_hit_rate'] = buy_signals
            analysis['buy_conversion_rate'] = buy_trades / buy_signals
            analysis['buy_effectiveness'] = min(1.0, analysis['buy_conversion_rate'] * 2)
        
        if sell_signals > 0:
            analysis['sell_hit_rate'] = sell_signals
            analysis['sell_conversion_rate'] = sell_trades / sell_signals
            analysis['sell_effectiveness'] = min(1.0, analysis['sell_conversion_rate'] * 2)
        
        return analysis
    
    def _analyze_lookback_effectiveness(self, lookback_periods: int, trades: List[Dict], 
                                      price_history: Dict) -> float:
        """Analyze effectiveness of lookback periods."""
        if not trades or not price_history:
            return 0.5
        
        # Simple heuristic: if we have good trade success rate, lookback is probably good
        successful_trades = len([t for t in trades if t.get('profit', 0) > 0])
        success_rate = successful_trades / len(trades) if trades else 0
        
        # Adjust based on trade frequency
        trade_frequency = len(trades)  # Simplified
        
        # Optimal lookback should balance responsiveness with stability
        if lookback_periods < 5:
            responsiveness_score = 0.8  # High responsiveness
            stability_score = 0.3       # Low stability
        elif lookback_periods > 15:
            responsiveness_score = 0.3  # Low responsiveness
            stability_score = 0.8       # High stability
        else:
            responsiveness_score = 0.6  # Medium responsiveness
            stability_score = 0.6       # Medium stability
        
        # Combine factors
        effectiveness = (success_rate * 0.4 + 
                        responsiveness_score * 0.3 + 
                        stability_score * 0.3)
        
        return min(1.0, effectiveness)
    
    def _suggest_lookback_range(self, trades: List[Dict], price_history: Dict) -> Tuple[int, int]:
        """Suggest optimal lookback range."""
        # Simple heuristic based on data frequency and volatility
        total_data_points = sum(len(data) for data in price_history.values())
        avg_data_points = total_data_points / len(price_history) if price_history else 10
        
        if avg_data_points < 20:
            return (3, 8)   # Short lookback for limited data
        elif avg_data_points < 50:
            return (5, 12)  # Medium lookback
        else:
            return (8, 20)  # Longer lookback for more data
    
    def _calculate_position_utilization(self, trades: List[Dict], max_position_size: float) -> float:
        """Calculate how much of the maximum position size is being utilized."""
        if not trades:
            return 0
        
        position_sizes = [t.get('position_size', 0) for t in trades if 'position_size' in t]
        if not position_sizes:
            return 0
        
        avg_position_size = np.mean(position_sizes)
        return avg_position_size / max_position_size
    
    def _analyze_position_size_effectiveness(self, trades: List[Dict], performance: Dict) -> float:
        """Analyze effectiveness of position sizing strategy."""
        if not trades:
            return 0.5
        
        # Analyze relationship between position size and profitability
        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
        
        if not profitable_trades:
            return 0.3  # Low effectiveness if no profitable trades
        
        # Simple effectiveness based on success rate and profit factor
        success_rate = performance.get('success_rate_percent', 0) / 100
        profit_factor = performance.get('profit_factor', 0)
        
        effectiveness = (success_rate * 0.6 + min(1.0, profit_factor / 2.0) * 0.4)
        return effectiveness
    
    def _analyze_min_trade_effectiveness(self, trades: List[Dict], min_trade_amount: float) -> float:
        """Analyze effectiveness of minimum trade amount."""
        if not trades:
            return 0.5
        
        # Count trades that were close to minimum
        near_minimum_trades = [t for t in trades if t.get('amount', 0) < min_trade_amount * 1.5]
        
        if not near_minimum_trades:
            return 0.7  # Good if we're not hitting minimum often
        
        # Analyze profitability of near-minimum trades
        profitable_near_min = [t for t in near_minimum_trades if t.get('profit', 0) > 0]
        
        if near_minimum_trades:
            near_min_success_rate = len(profitable_near_min) / len(near_minimum_trades)
            return near_min_success_rate
        
        return 0.5
    
    def _identify_improvement_opportunities(self, performance: Dict, parameter_effectiveness: Dict,
                                          trades: List[Dict], price_history: Dict) -> List[Dict]:
        """Identify specific areas for improvement."""
        opportunities = []
        
        # Low success rate opportunity
        success_rate = performance.get('success_rate_percent', 0)
        if success_rate < 50:
            opportunities.append({
                'type': 'success_rate_improvement',
                'priority': 'high',
                'current_value': success_rate,
                'target_value': 60,
                'description': 'Success rate is below 50%, indicating threshold adjustments needed',
                'impact_estimate': 'high'
            })
        
        # Low trade frequency opportunity
        trade_frequency = performance.get('trade_frequency', 0)
        if trade_frequency < 0.5:  # Less than 0.5 trades per hour
            opportunities.append({
                'type': 'trade_frequency_improvement',
                'priority': 'medium',
                'current_value': trade_frequency,
                'target_value': 1.0,
                'description': 'Low trading frequency suggests thresholds may be too restrictive',
                'impact_estimate': 'medium'
            })
        
        # Threshold effectiveness opportunities
        buy_effectiveness = parameter_effectiveness.get('thresholds', {}).get('buy_threshold', {}).get('effectiveness_score', 0.5)
        if buy_effectiveness < 0.4:
            opportunities.append({
                'type': 'buy_threshold_optimization',
                'priority': 'high',
                'current_value': buy_effectiveness,
                'target_value': 0.6,
                'description': 'Buy threshold is not effectively triggering profitable trades',
                'impact_estimate': 'high'
            })
        
        sell_effectiveness = parameter_effectiveness.get('thresholds', {}).get('sell_threshold', {}).get('effectiveness_score', 0.5)
        if sell_effectiveness < 0.4:
            opportunities.append({
                'type': 'sell_threshold_optimization',
                'priority': 'high',
                'current_value': sell_effectiveness,
                'target_value': 0.6,
                'description': 'Sell threshold is not effectively triggering profitable trades',
                'impact_estimate': 'high'
            })
        
        # Position sizing opportunity
        position_utilization = parameter_effectiveness.get('position_sizing', {}).get('max_position_size', {}).get('utilization_rate', 0)
        if position_utilization < 0.3:
            opportunities.append({
                'type': 'position_sizing_optimization',
                'priority': 'medium',
                'current_value': position_utilization,
                'target_value': 0.5,
                'description': 'Low position size utilization suggests room for larger positions',
                'impact_estimate': 'medium'
            })
        
        # Drawdown opportunity
        max_drawdown = abs(performance.get('max_drawdown', 0))
        if max_drawdown > 5:  # More than 5% drawdown
            opportunities.append({
                'type': 'risk_management_improvement',
                'priority': 'high',
                'current_value': max_drawdown,
                'target_value': 3,
                'description': 'High drawdown indicates need for better risk management',
                'impact_estimate': 'high'
            })
        
        return opportunities
    
    def _generate_optimization_recommendations(self, opportunities: List[Dict], 
                                             current_params: Dict) -> List[Dict]:
        """Generate specific parameter optimization recommendations."""
        recommendations = []
        
        for opportunity in opportunities:
            opp_type = opportunity['type']
            priority = opportunity['priority']
            
            if opp_type == 'success_rate_improvement':
                # Recommend threshold adjustments
                current_buy = current_params.get('buy_threshold', -0.008)
                current_sell = current_params.get('sell_threshold', 0.012)
                
                recommendations.append({
                    'parameter': 'buy_threshold',
                    'current_value': current_buy,
                    'recommended_value': current_buy * 0.8,  # Less restrictive
                    'change_percent': -20,
                    'rationale': 'Reduce buy threshold to capture more opportunities',
                    'priority': priority,
                    'expected_impact': 'Increase trade frequency and potentially success rate'
                })
                
                recommendations.append({
                    'parameter': 'sell_threshold',
                    'current_value': current_sell,
                    'recommended_value': current_sell * 0.8,  # Less restrictive
                    'change_percent': -20,
                    'rationale': 'Reduce sell threshold to capture more opportunities',
                    'priority': priority,
                    'expected_impact': 'Increase trade frequency and potentially success rate'
                })
            
            elif opp_type == 'trade_frequency_improvement':
                # Recommend more aggressive thresholds
                current_buy = current_params.get('buy_threshold', -0.008)
                current_sell = current_params.get('sell_threshold', 0.012)
                
                recommendations.append({
                    'parameter': 'buy_threshold',
                    'current_value': current_buy,
                    'recommended_value': current_buy * 0.7,  # More aggressive
                    'change_percent': -30,
                    'rationale': 'More aggressive buy threshold to increase trading frequency',
                    'priority': priority,
                    'expected_impact': 'Significantly increase trade frequency'
                })
            
            elif opp_type == 'buy_threshold_optimization':
                current_buy = current_params.get('buy_threshold', -0.008)
                recommendations.append({
                    'parameter': 'buy_threshold',
                    'current_value': current_buy,
                    'recommended_value': current_buy * 1.2,  # More conservative
                    'change_percent': 20,
                    'rationale': 'Optimize buy threshold for better trade quality',
                    'priority': priority,
                    'expected_impact': 'Improve buy trade success rate'
                })
            
            elif opp_type == 'sell_threshold_optimization':
                current_sell = current_params.get('sell_threshold', 0.012)
                recommendations.append({
                    'parameter': 'sell_threshold',
                    'current_value': current_sell,
                    'recommended_value': current_sell * 1.2,  # More conservative
                    'change_percent': 20,
                    'rationale': 'Optimize sell threshold for better trade quality',
                    'priority': priority,
                    'expected_impact': 'Improve sell trade success rate'
                })
            
            elif opp_type == 'position_sizing_optimization':
                current_max_pos = current_params.get('max_position_size', 0.3)
                recommendations.append({
                    'parameter': 'max_position_size',
                    'current_value': current_max_pos,
                    'recommended_value': min(0.5, current_max_pos * 1.3),
                    'change_percent': 30,
                    'rationale': 'Increase position size to better utilize capital',
                    'priority': priority,
                    'expected_impact': 'Increase potential returns per trade'
                })
            
            elif opp_type == 'risk_management_improvement':
                current_lookback = current_params.get('lookback_periods', 8)
                recommendations.append({
                    'parameter': 'lookback_periods',
                    'current_value': current_lookback,
                    'recommended_value': min(20, current_lookback + 3),
                    'change_percent': 37.5,
                    'rationale': 'Increase lookback periods for more stable signals',
                    'priority': priority,
                    'expected_impact': 'Reduce drawdown and improve stability'
                })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return recommendations
    
    def generate_optimized_parameters(self, current_params: Dict, 
                                    optimization_recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Generate optimized parameter set based on recommendations.
        
        Args:
            current_params: Current parameter values
            optimization_recommendations: List of optimization recommendations
            
        Returns:
            Optimized parameter set
        """
        optimized_params = current_params.copy()
        applied_changes = []
        
        # Apply high priority recommendations first
        high_priority_recs = [r for r in optimization_recommendations if r['priority'] == 'high']
        
        for rec in high_priority_recs[:3]:  # Apply top 3 high priority changes
            param_name = rec['parameter']
            recommended_value = rec['recommended_value']
            
            # Apply the change
            optimized_params[param_name] = recommended_value
            applied_changes.append({
                'parameter': param_name,
                'old_value': rec['current_value'],
                'new_value': recommended_value,
                'change_percent': rec['change_percent'],
                'rationale': rec['rationale']
            })
        
        # Apply medium priority if we have room
        if len(applied_changes) < 3:
            medium_priority_recs = [r for r in optimization_recommendations if r['priority'] == 'medium']
            
            for rec in medium_priority_recs[:2]:  # Apply up to 2 medium priority changes
                if len(applied_changes) >= 3:
                    break
                
                param_name = rec['parameter']
                recommended_value = rec['recommended_value']
                
                optimized_params[param_name] = recommended_value
                applied_changes.append({
                    'parameter': param_name,
                    'old_value': rec['current_value'],
                    'new_value': recommended_value,
                    'change_percent': rec['change_percent'],
                    'rationale': rec['rationale']
                })
        
        return {
            'optimized_parameters': optimized_params,
            'applied_changes': applied_changes,
            'optimization_summary': {
                'total_recommendations': len(optimization_recommendations),
                'applied_changes': len(applied_changes),
                'expected_improvements': [change['rationale'] for change in applied_changes]
            }
        }
    
    def save_optimization_results(self, optimization_data: Dict, filepath: str):
        """Save optimization results to file."""
        with open(filepath, 'w') as f:
            json.dump(optimization_data, f, indent=2, default=str)
    
    def load_optimization_history(self, filepath: str) -> List[Dict]:
        """Load optimization history from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []