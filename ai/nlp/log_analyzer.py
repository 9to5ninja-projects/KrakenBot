"""
Trading Log Analyzer using NLP/LLM
Analyzes trading session data and generates intelligent insights
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

class TradingLogAnalyzer:
    """
    Analyzes trading logs and session data to generate intelligent insights
    using Natural Language Processing and Large Language Models.
    """
    
    def __init__(self, llm_provider: str = "local"):
        """
        Initialize the Trading Log Analyzer.
        
        Args:
            llm_provider: LLM provider to use ("openai", "local", "mock")
        """
        self.llm_provider = llm_provider
        self.llm = self._setup_llm(llm_provider)
        
    def _setup_llm(self, provider: str):
        """Setup the LLM provider."""
        if provider == "openai":
            try:
                import openai
                # Will implement OpenAI integration
                return "openai_client"
            except ImportError:
                print("OpenAI not available, falling back to mock")
                return "mock"
        elif provider == "local":
            # For now, use mock - can implement local LLM later
            return "mock"
        else:
            return "mock"
    
    def analyze_session_data(self, session_dir: str) -> Dict[str, Any]:
        """
        Analyze a complete trading session and generate insights.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            Dictionary containing analysis results and insights
        """
        session_path = Path(session_dir)
        if not session_path.exists():
            return {"error": f"Session directory not found: {session_dir}"}
        
        # Load session data
        session_data = self._load_session_data(session_path)
        
        # Perform analysis
        analysis = {
            "session_info": self._analyze_session_info(session_data),
            "trading_performance": self._analyze_trading_performance(session_data),
            "market_conditions": self._analyze_market_conditions(session_data),
            "missed_opportunities": self._identify_missed_opportunities(session_data),
            "strategy_insights": self._generate_strategy_insights(session_data),
            "recommendations": self._generate_recommendations(session_data),
            "ai_summary": self._generate_ai_summary(session_data)
        }
        
        return analysis
    
    def _load_session_data(self, session_path: Path) -> Dict[str, Any]:
        """Load all session data files."""
        data = {}
        
        # Load metadata
        metadata_file = session_path / "session_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data['metadata'] = json.load(f)
        
        # Load trades
        trades_file = session_path / "trades.json"
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                data['trades'] = json.load(f)
        else:
            data['trades'] = []
        
        # Load portfolio history
        portfolio_file = session_path / "portfolio_history.json"
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                data['portfolio_history'] = json.load(f)
        else:
            data['portfolio_history'] = []
        
        # Load price history
        price_file = session_path / "price_history.json"
        if price_file.exists():
            with open(price_file, 'r') as f:
                data['price_history'] = json.load(f)
        else:
            data['price_history'] = []
        
        # Load performance summary
        perf_file = session_path / "performance_summary.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                data['performance'] = json.load(f)
        
        return data
    
    def _analyze_session_info(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze basic session information."""
        metadata = session_data.get('metadata', {})
        portfolio_history = session_data.get('portfolio_history', [])
        
        if not portfolio_history:
            return {"status": "No portfolio data available"}
        
        start_time = metadata.get('start_time', 'Unknown')
        duration_minutes = len(portfolio_history) * 5  # 5-minute intervals
        
        return {
            "session_id": metadata.get('session_id', 'Unknown'),
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "duration_hours": round(duration_minutes / 60, 2),
            "data_points": len(portfolio_history),
            "trading_pairs": metadata.get('trading_pairs', []),
            "initial_amount": metadata.get('start_amount', 0)
        }
    
    def _analyze_trading_performance(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze trading performance metrics."""
        trades = session_data.get('trades', [])
        portfolio_history = session_data.get('portfolio_history', [])
        
        if not portfolio_history:
            return {"status": "No performance data available"}
        
        # Calculate basic metrics
        initial_value = portfolio_history[0].get('total_value', 100) if portfolio_history else 100
        final_value = portfolio_history[-1].get('total_value', 100) if portfolio_history else 100
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Analyze trades
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t.get('action') == 'buy'])
        sell_trades = len([t for t in trades if t.get('action') == 'sell'])
        
        # Calculate trade success rate (simplified)
        successful_trades = 0
        for trade in trades:
            if trade.get('action') == 'sell' and trade.get('profit', 0) > 0:
                successful_trades += 1
        
        success_rate = (successful_trades / max(sell_trades, 1)) * 100 if sell_trades > 0 else 0
        
        return {
            "total_return_percent": round(total_return, 4),
            "initial_value": initial_value,
            "final_value": final_value,
            "absolute_profit": round(final_value - initial_value, 2),
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "success_rate_percent": round(success_rate, 2),
            "avg_trade_frequency": f"{round(total_trades / max(len(portfolio_history) * 5 / 60, 1), 2)} trades/hour"
        }
    
    def _analyze_market_conditions(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze market conditions during the session."""
        price_history = session_data.get('price_history', {})
        
        if not price_history:
            return {"status": "No price data available"}
        
        # Analyze price movements for each pair
        pair_analysis = {}
        
        # Handle the actual structure: {pair: [{'timestamp': ..., 'price': ...}, ...]}
        for pair, price_entries in price_history.items():
            if not isinstance(price_entries, list):
                continue
                
            pair_analysis[pair] = {
                'prices': [],
                'timestamps': []
            }
            
            for entry in price_entries:
                if isinstance(entry, dict):
                    price = entry.get('price')
                    timestamp = entry.get('timestamp')
                    if price is not None and timestamp is not None:
                        pair_analysis[pair]['prices'].append(price)
                        pair_analysis[pair]['timestamps'].append(timestamp)
        
        # Calculate volatility and trends
        market_summary = {}
        for pair, data in pair_analysis.items():
            prices = data['prices']
            if len(prices) < 2:
                continue
                
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            volatility = sum(abs(change) for change in price_changes) / len(price_changes)
            total_change = prices[-1] - prices[0]
            percent_change = (total_change / prices[0]) * 100
            
            market_summary[pair] = {
                'start_price': prices[0],
                'end_price': prices[-1],
                'percent_change': round(percent_change, 4),
                'volatility': round(volatility, 6),
                'trend': 'bullish' if total_change > 0 else 'bearish' if total_change < 0 else 'sideways'
            }
        
        return {
            "pairs_analyzed": len(market_summary),
            "market_data": market_summary,
            "overall_volatility": "high" if any(data['volatility'] > 0.01 for data in market_summary.values()) else "moderate"
        }
    
    def _identify_missed_opportunities(self, session_data: Dict) -> List[Dict[str, Any]]:
        """Identify potential missed trading opportunities."""
        price_history = session_data.get('price_history', {})
        trades = session_data.get('trades', [])
        
        if not price_history:
            return []
        
        missed_opportunities = []
        
        # Analyze each pair separately
        for pair, price_entries in price_history.items():
            if not isinstance(price_entries, list) or len(price_entries) < 2:
                continue
            
            # Look for significant price movements without trades
            for i in range(1, len(price_entries)):
                current_entry = price_entries[i]
                prev_entry = price_entries[i-1]
                
                if not isinstance(current_entry, dict) or not isinstance(prev_entry, dict):
                    continue
                
                current_price = current_entry.get('price')
                prev_price = prev_entry.get('price')
                timestamp = current_entry.get('timestamp')
                
                if current_price is None or prev_price is None or timestamp is None:
                    continue
                
                price_change = (current_price - prev_price) / prev_price
                
                # Check for significant movements (>1% change)
                if abs(price_change) > 0.01:
                    # Check if there was a trade around this time
                    trade_found = any(
                        trade.get('pair') == pair and 
                        abs(self._parse_timestamp(trade.get('timestamp', '')).timestamp() - 
                            self._parse_timestamp(timestamp).timestamp()) < 300  # 5 minutes
                        for trade in trades
                    )
                    
                    if not trade_found:
                        missed_opportunities.append({
                            'timestamp': timestamp,
                            'pair': pair,
                            'price_change_percent': round(price_change * 100, 4),
                            'opportunity_type': 'buy' if price_change < -0.008 else 'sell' if price_change > 0.012 else 'monitor',
                            'reason': f"Significant {'drop' if price_change < 0 else 'rise'} without corresponding trade"
                        })
        
        return missed_opportunities[:10]  # Return top 10 missed opportunities
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        try:
            # Handle different timestamp formats
            if 'T' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp_str)
        except:
            return datetime.now()
    
    def _generate_strategy_insights(self, session_data: Dict) -> List[str]:
        """Generate strategic insights based on session data."""
        insights = []
        
        # Analyze trading performance
        trades = session_data.get('trades', [])
        portfolio_history = session_data.get('portfolio_history', [])
        
        if not trades:
            insights.append("No trades executed during this session - consider adjusting thresholds for more activity")
            return insights
        
        # Analyze trade timing
        trade_hours = {}
        for trade in trades:
            try:
                timestamp = trade.get('timestamp', '')
                hour = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).hour
                trade_hours[hour] = trade_hours.get(hour, 0) + 1
            except:
                continue
        
        if trade_hours:
            most_active_hour = max(trade_hours, key=trade_hours.get)
            insights.append(f"Most trading activity occurred at hour {most_active_hour}:00")
        
        # Analyze pair performance
        pair_trades = {}
        for trade in trades:
            pair = trade.get('pair', 'unknown')
            if pair not in pair_trades:
                pair_trades[pair] = {'count': 0, 'profit': 0}
            pair_trades[pair]['count'] += 1
            pair_trades[pair]['profit'] += trade.get('profit', 0)
        
        if pair_trades:
            best_pair = max(pair_trades, key=lambda x: pair_trades[x]['profit'])
            insights.append(f"Most profitable pair: {best_pair} with {pair_trades[best_pair]['count']} trades")
        
        # Portfolio growth analysis
        if len(portfolio_history) > 1:
            growth_periods = []
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i-1].get('total_value', 0)
                curr_value = portfolio_history[i].get('total_value', 0)
                if curr_value > prev_value:
                    growth_periods.append(i)
            
            if growth_periods:
                insights.append(f"Portfolio showed growth in {len(growth_periods)} out of {len(portfolio_history)-1} periods")
        
        return insights
    
    def _generate_recommendations(self, session_data: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze performance metrics
        performance = self._analyze_trading_performance(session_data)
        market_conditions = self._analyze_market_conditions(session_data)
        missed_opportunities = self._identify_missed_opportunities(session_data)
        
        # Performance-based recommendations
        if performance.get('total_return_percent', 0) < 0:
            recommendations.append("Consider tightening risk management - portfolio showed negative returns")
        
        if performance.get('success_rate_percent', 0) < 50:
            recommendations.append("Trade success rate is below 50% - consider adjusting entry/exit thresholds")
        
        if performance.get('total_trades', 0) == 0:
            recommendations.append("No trades executed - consider lowering thresholds or increasing monitoring frequency")
        
        # Market condition recommendations
        market_data = market_conditions.get('market_data', {})
        high_volatility_pairs = [pair for pair, data in market_data.items() if data.get('volatility', 0) > 0.01]
        
        if high_volatility_pairs:
            recommendations.append(f"High volatility detected in {', '.join(high_volatility_pairs)} - consider tighter stop-losses")
        
        # Missed opportunity recommendations
        if len(missed_opportunities) > 5:
            recommendations.append(f"Identified {len(missed_opportunities)} missed opportunities - consider more aggressive thresholds")
        
        # Default recommendations if none generated
        if not recommendations:
            recommendations.append("Session performed within normal parameters - continue current strategy")
        
        return recommendations
    
    def _generate_ai_summary(self, session_data: Dict) -> str:
        """Generate an AI-powered natural language summary."""
        # For now, create a structured summary - will enhance with actual LLM later
        
        session_info = self._analyze_session_info(session_data)
        performance = self._analyze_trading_performance(session_data)
        market_conditions = self._analyze_market_conditions(session_data)
        missed_opportunities = self._identify_missed_opportunities(session_data)
        
        summary_parts = []
        
        # Session overview
        duration = session_info.get('duration_hours', 0)
        summary_parts.append(f"Trading session ran for {duration} hours with {session_info.get('data_points', 0)} data collection points.")
        
        # Performance summary
        total_return = performance.get('total_return_percent', 0)
        total_trades = performance.get('total_trades', 0)
        
        if total_return > 0:
            summary_parts.append(f"Portfolio achieved a positive return of {total_return:.4f}% with {total_trades} trades executed.")
        elif total_return < 0:
            summary_parts.append(f"Portfolio experienced a {abs(total_return):.4f}% loss with {total_trades} trades executed.")
        else:
            summary_parts.append(f"Portfolio remained stable with {total_trades} trades executed.")
        
        # Market conditions
        market_data = market_conditions.get('market_data', {})
        if market_data:
            volatile_pairs = [pair for pair, data in market_data.items() if data.get('volatility', 0) > 0.01]
            if volatile_pairs:
                summary_parts.append(f"High volatility was observed in {', '.join(volatile_pairs)}, presenting both opportunities and risks.")
        
        # Missed opportunities
        if missed_opportunities:
            summary_parts.append(f"Analysis identified {len(missed_opportunities)} potential missed opportunities, suggesting room for strategy optimization.")
        
        # Overall assessment
        if total_return > 1:
            summary_parts.append("Overall, this was a successful trading session with positive returns.")
        elif total_return > -1:
            summary_parts.append("The session showed modest performance with room for improvement.")
        else:
            summary_parts.append("This session underperformed expectations and requires strategy review.")
        
        return " ".join(summary_parts)

    def generate_daily_summary(self, session_dir: str) -> str:
        """Generate a concise daily summary for quick review."""
        analysis = self.analyze_session_data(session_dir)
        
        if "error" in analysis:
            return f"Error generating summary: {analysis['error']}"
        
        # Extract key metrics
        session_info = analysis.get('session_info', {})
        performance = analysis.get('trading_performance', {})
        recommendations = analysis.get('recommendations', [])
        
        # Create summary
        summary = f"""
📊 DAILY TRADING SUMMARY
Session: {session_info.get('session_id', 'Unknown')}
Duration: {session_info.get('duration_hours', 0)} hours

💰 PERFORMANCE
Return: {performance.get('total_return_percent', 0):.4f}%
Trades: {performance.get('total_trades', 0)} ({performance.get('buy_trades', 0)} buy, {performance.get('sell_trades', 0)} sell)
Success Rate: {performance.get('success_rate_percent', 0):.2f}%

🎯 TOP RECOMMENDATIONS
{chr(10).join(f"• {rec}" for rec in recommendations[:3])}

🤖 AI INSIGHT
{analysis.get('ai_summary', 'No insights available')}
"""
        
        return summary.strip()