"""
AI Report Generator for KrakenBot
Generates comprehensive AI-powered reports from trading data
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from .log_analyzer import TradingLogAnalyzer

class AIReportGenerator:
    """
    Generates comprehensive AI-powered reports from trading session data.
    """
    
    def __init__(self, llm_provider: str = "local"):
        """
        Initialize the AI Report Generator.
        
        Args:
            llm_provider: LLM provider to use for analysis
        """
        self.analyzer = TradingLogAnalyzer(llm_provider)
        
    def generate_session_report(self, session_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a trading session.
        
        Args:
            session_dir: Path to session directory
            output_dir: Optional output directory for report files
            
        Returns:
            Dictionary containing the complete report
        """
        # Analyze session data
        analysis = self.analyzer.analyze_session_data(session_dir)
        
        if "error" in analysis:
            return analysis
        
        # Generate comprehensive report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_directory": session_dir,
                "report_version": "1.0.0",
                "analyzer_version": "1.0.0"
            },
            "executive_summary": self._generate_executive_summary(analysis),
            "detailed_analysis": analysis,
            "performance_insights": self._generate_performance_insights(analysis),
            "market_analysis": self._generate_market_analysis(analysis),
            "risk_assessment": self._generate_risk_assessment(analysis),
            "optimization_suggestions": self._generate_optimization_suggestions(analysis),
            "action_items": self._generate_action_items(analysis)
        }
        
        # Save report if output directory specified
        if output_dir:
            self._save_report(report, output_dir)
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the trading session."""
        session_info = analysis.get('session_info', {})
        performance = analysis.get('trading_performance', {})
        
        # Determine session rating
        total_return = performance.get('total_return_percent', 0)
        success_rate = performance.get('success_rate_percent', 0)
        
        if total_return > 2 and success_rate > 70:
            rating = "Excellent"
        elif total_return > 0.5 and success_rate > 50:
            rating = "Good"
        elif total_return > -0.5 and success_rate > 30:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            "session_rating": rating,
            "key_metrics": {
                "duration_hours": session_info.get('duration_hours', 0),
                "total_return_percent": performance.get('total_return_percent', 0),
                "total_trades": performance.get('total_trades', 0),
                "success_rate_percent": performance.get('success_rate_percent', 0)
            },
            "highlights": self._extract_highlights(analysis),
            "concerns": self._extract_concerns(analysis),
            "overall_assessment": analysis.get('ai_summary', 'No assessment available')
        }
    
    def _extract_highlights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract positive highlights from the analysis."""
        highlights = []
        
        performance = analysis.get('trading_performance', {})
        
        if performance.get('total_return_percent', 0) > 0:
            highlights.append(f"Achieved positive return of {performance.get('total_return_percent', 0):.4f}%")
        
        if performance.get('success_rate_percent', 0) > 60:
            highlights.append(f"High trade success rate of {performance.get('success_rate_percent', 0):.2f}%")
        
        if performance.get('total_trades', 0) > 10:
            highlights.append(f"Active trading with {performance.get('total_trades', 0)} trades executed")
        
        # Market condition highlights
        market_data = analysis.get('market_conditions', {}).get('market_data', {})
        profitable_pairs = [pair for pair, data in market_data.items() if data.get('percent_change', 0) > 1]
        if profitable_pairs:
            highlights.append(f"Favorable market conditions in {', '.join(profitable_pairs)}")
        
        if not highlights:
            highlights.append("Session completed without major issues")
        
        return highlights
    
    def _extract_concerns(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract concerns and issues from the analysis."""
        concerns = []
        
        performance = analysis.get('trading_performance', {})
        
        if performance.get('total_return_percent', 0) < -1:
            concerns.append(f"Significant loss of {abs(performance.get('total_return_percent', 0)):.4f}%")
        
        if performance.get('success_rate_percent', 0) < 40:
            concerns.append(f"Low trade success rate of {performance.get('success_rate_percent', 0):.2f}%")
        
        missed_opportunities = analysis.get('missed_opportunities', [])
        if len(missed_opportunities) > 5:
            concerns.append(f"Multiple missed opportunities ({len(missed_opportunities)}) identified")
        
        # Market volatility concerns
        market_data = analysis.get('market_conditions', {}).get('market_data', {})
        volatile_pairs = [pair for pair, data in market_data.items() if data.get('volatility', 0) > 0.02]
        if volatile_pairs:
            concerns.append(f"High volatility in {', '.join(volatile_pairs)} increased risk")
        
        return concerns
    
    def _generate_performance_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed performance insights."""
        performance = analysis.get('trading_performance', {})
        strategy_insights = analysis.get('strategy_insights', [])
        
        return {
            "profitability_analysis": {
                "total_return": performance.get('total_return_percent', 0),
                "absolute_profit": performance.get('absolute_profit', 0),
                "return_classification": self._classify_return(performance.get('total_return_percent', 0))
            },
            "trading_activity": {
                "total_trades": performance.get('total_trades', 0),
                "trade_frequency": performance.get('avg_trade_frequency', 'Unknown'),
                "buy_sell_ratio": self._calculate_buy_sell_ratio(performance)
            },
            "success_metrics": {
                "success_rate": performance.get('success_rate_percent', 0),
                "success_classification": self._classify_success_rate(performance.get('success_rate_percent', 0))
            },
            "strategic_insights": strategy_insights
        }
    
    def _classify_return(self, return_pct: float) -> str:
        """Classify return performance."""
        if return_pct > 5:
            return "Exceptional"
        elif return_pct > 2:
            return "Excellent"
        elif return_pct > 0.5:
            return "Good"
        elif return_pct > -0.5:
            return "Neutral"
        elif return_pct > -2:
            return "Poor"
        else:
            return "Critical"
    
    def _classify_success_rate(self, success_rate: float) -> str:
        """Classify success rate performance."""
        if success_rate > 80:
            return "Exceptional"
        elif success_rate > 60:
            return "Good"
        elif success_rate > 40:
            return "Average"
        elif success_rate > 20:
            return "Below Average"
        else:
            return "Poor"
    
    def _calculate_buy_sell_ratio(self, performance: Dict[str, Any]) -> str:
        """Calculate buy/sell ratio."""
        buy_trades = performance.get('buy_trades', 0)
        sell_trades = performance.get('sell_trades', 0)
        
        if sell_trades == 0:
            return f"{buy_trades}:0 (Only buys)"
        elif buy_trades == 0:
            return f"0:{sell_trades} (Only sells)"
        else:
            ratio = buy_trades / sell_trades
            return f"{ratio:.2f}:1"
    
    def _generate_market_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market condition analysis."""
        market_conditions = analysis.get('market_conditions', {})
        market_data = market_conditions.get('market_data', {})
        
        # Analyze each pair
        pair_analysis = {}
        for pair, data in market_data.items():
            pair_analysis[pair] = {
                "price_movement": data.get('percent_change', 0),
                "volatility_level": self._classify_volatility(data.get('volatility', 0)),
                "trend_direction": data.get('trend', 'unknown'),
                "trading_recommendation": self._generate_pair_recommendation(data)
            }
        
        # Overall market assessment
        avg_volatility = sum(data.get('volatility', 0) for data in market_data.values()) / max(len(market_data), 1)
        
        return {
            "overall_market_condition": market_conditions.get('overall_volatility', 'unknown'),
            "average_volatility": avg_volatility,
            "pair_analysis": pair_analysis,
            "market_opportunities": self._identify_market_opportunities(market_data),
            "market_risks": self._identify_market_risks(market_data)
        }
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level."""
        if volatility > 0.02:
            return "Very High"
        elif volatility > 0.01:
            return "High"
        elif volatility > 0.005:
            return "Moderate"
        elif volatility > 0.002:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_pair_recommendation(self, pair_data: Dict[str, Any]) -> str:
        """Generate trading recommendation for a pair."""
        volatility = pair_data.get('volatility', 0)
        trend = pair_data.get('trend', 'sideways')
        percent_change = pair_data.get('percent_change', 0)
        
        if volatility > 0.02:
            return "High risk - use tight stops"
        elif trend == 'bullish' and percent_change > 1:
            return "Consider taking profits"
        elif trend == 'bearish' and percent_change < -1:
            return "Potential buying opportunity"
        else:
            return "Monitor for breakout"
    
    def _identify_market_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify market opportunities."""
        opportunities = []
        
        for pair, data in market_data.items():
            if data.get('trend') == 'bullish' and data.get('volatility', 0) < 0.01:
                opportunities.append(f"{pair}: Stable uptrend with low volatility")
            elif data.get('percent_change', 0) < -2 and data.get('volatility', 0) > 0.01:
                opportunities.append(f"{pair}: Potential oversold bounce opportunity")
        
        return opportunities
    
    def _identify_market_risks(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify market risks."""
        risks = []
        
        for pair, data in market_data.items():
            if data.get('volatility', 0) > 0.02:
                risks.append(f"{pair}: Extremely high volatility increases execution risk")
            elif data.get('trend') == 'bearish' and data.get('percent_change', 0) < -3:
                risks.append(f"{pair}: Strong downtrend may continue")
        
        return risks
    
    def _generate_risk_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment."""
        performance = analysis.get('trading_performance', {})
        market_conditions = analysis.get('market_conditions', {})
        missed_opportunities = analysis.get('missed_opportunities', [])
        
        # Calculate risk score (0-100, higher is riskier)
        risk_score = 0
        
        # Performance-based risk
        if performance.get('total_return_percent', 0) < -2:
            risk_score += 30
        elif performance.get('total_return_percent', 0) < 0:
            risk_score += 15
        
        # Success rate risk
        if performance.get('success_rate_percent', 0) < 30:
            risk_score += 25
        elif performance.get('success_rate_percent', 0) < 50:
            risk_score += 10
        
        # Market volatility risk
        if market_conditions.get('overall_volatility') == 'high':
            risk_score += 20
        
        # Missed opportunities risk
        if len(missed_opportunities) > 10:
            risk_score += 15
        elif len(missed_opportunities) > 5:
            risk_score += 10
        
        risk_level = "Low" if risk_score < 25 else "Medium" if risk_score < 50 else "High" if risk_score < 75 else "Critical"
        
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "risk_factors": self._identify_risk_factors(analysis),
            "mitigation_strategies": self._suggest_risk_mitigation(risk_score, analysis)
        }
    
    def _identify_risk_factors(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors."""
        risk_factors = []
        
        performance = analysis.get('trading_performance', {})
        if performance.get('success_rate_percent', 0) < 50:
            risk_factors.append("Low trade success rate indicates strategy issues")
        
        market_data = analysis.get('market_conditions', {}).get('market_data', {})
        volatile_pairs = [pair for pair, data in market_data.items() if data.get('volatility', 0) > 0.015]
        if volatile_pairs:
            risk_factors.append(f"High volatility in {', '.join(volatile_pairs)} increases execution risk")
        
        missed_opportunities = analysis.get('missed_opportunities', [])
        if len(missed_opportunities) > 8:
            risk_factors.append("Multiple missed opportunities suggest overly conservative thresholds")
        
        return risk_factors
    
    def _suggest_risk_mitigation(self, risk_score: int, analysis: Dict[str, Any]) -> List[str]:
        """Suggest risk mitigation strategies."""
        strategies = []
        
        if risk_score > 50:
            strategies.append("Consider reducing position sizes until performance improves")
            strategies.append("Implement stricter stop-loss mechanisms")
        
        performance = analysis.get('trading_performance', {})
        if performance.get('success_rate_percent', 0) < 40:
            strategies.append("Review and optimize entry/exit thresholds")
        
        market_conditions = analysis.get('market_conditions', {})
        if market_conditions.get('overall_volatility') == 'high':
            strategies.append("Increase monitoring frequency during volatile periods")
        
        if not strategies:
            strategies.append("Continue current risk management approach")
        
        return strategies
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Performance optimizations
        performance = analysis.get('trading_performance', {})
        if performance.get('total_trades', 0) == 0:
            suggestions.append({
                "category": "Activity",
                "priority": "High",
                "suggestion": "Lower trading thresholds to increase activity",
                "expected_impact": "More trading opportunities"
            })
        
        if performance.get('success_rate_percent', 0) < 50:
            suggestions.append({
                "category": "Strategy",
                "priority": "High", 
                "suggestion": "Optimize entry/exit criteria to improve success rate",
                "expected_impact": "Higher profitability per trade"
            })
        
        # Market-based optimizations
        missed_opportunities = analysis.get('missed_opportunities', [])
        if len(missed_opportunities) > 5:
            suggestions.append({
                "category": "Thresholds",
                "priority": "Medium",
                "suggestion": "Consider more aggressive thresholds during favorable conditions",
                "expected_impact": "Capture more opportunities"
            })
        
        return suggestions
    
    def _generate_action_items(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific action items."""
        action_items = []
        
        recommendations = analysis.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5]):  # Top 5 recommendations
            action_items.append({
                "id": i + 1,
                "priority": "High" if i < 2 else "Medium",
                "action": rec,
                "timeline": "Next session",
                "owner": "Trading System"
            })
        
        return action_items
    
    def _save_report(self, report: Dict[str, Any], output_dir: str):
        """Save the report to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON report
        with open(output_path / "ai_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save executive summary as text
        exec_summary = report.get('executive_summary', {})
        summary_text = f"""
AI TRADING ANALYSIS REPORT
Generated: {report['report_metadata']['generated_at']}

EXECUTIVE SUMMARY
Session Rating: {exec_summary.get('session_rating', 'Unknown')}

KEY METRICS:
- Duration: {exec_summary.get('key_metrics', {}).get('duration_hours', 0)} hours
- Return: {exec_summary.get('key_metrics', {}).get('total_return_percent', 0):.4f}%
- Trades: {exec_summary.get('key_metrics', {}).get('total_trades', 0)}
- Success Rate: {exec_summary.get('key_metrics', {}).get('success_rate_percent', 0):.2f}%

HIGHLIGHTS:
{chr(10).join(f"• {h}" for h in exec_summary.get('highlights', []))}

CONCERNS:
{chr(10).join(f"• {c}" for c in exec_summary.get('concerns', []))}

OVERALL ASSESSMENT:
{exec_summary.get('overall_assessment', 'No assessment available')}

ACTION ITEMS:
{chr(10).join(f"{i['id']}. [{i['priority']}] {i['action']}" for i in report.get('action_items', []))}
"""
        
        with open(output_path / "executive_summary.txt", 'w') as f:
            f.write(summary_text.strip())
        
        print(f"📊 AI report saved to {output_path}")

    def generate_comparative_report(self, session_dirs: List[str]) -> Dict[str, Any]:
        """Generate a comparative report across multiple sessions."""
        if len(session_dirs) < 2:
            return {"error": "Need at least 2 sessions for comparison"}
        
        # Analyze each session
        session_analyses = {}
        for session_dir in session_dirs:
            session_name = Path(session_dir).name
            session_analyses[session_name] = self.analyzer.analyze_session_data(session_dir)
        
        # Generate comparison
        comparison = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "sessions_compared": len(session_dirs),
                "report_type": "comparative_analysis"
            },
            "performance_comparison": self._compare_performance(session_analyses),
            "trend_analysis": self._analyze_trends(session_analyses),
            "best_practices": self._identify_best_practices(session_analyses),
            "improvement_opportunities": self._identify_improvements(session_analyses)
        }
        
        return comparison
    
    def _compare_performance(self, session_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across sessions."""
        performance_data = {}
        
        for session_name, analysis in session_analyses.items():
            if "error" not in analysis:
                perf = analysis.get('trading_performance', {})
                performance_data[session_name] = {
                    'return_percent': perf.get('total_return_percent', 0),
                    'total_trades': perf.get('total_trades', 0),
                    'success_rate': perf.get('success_rate_percent', 0)
                }
        
        # Find best and worst performers
        if performance_data:
            best_session = max(performance_data, key=lambda x: performance_data[x]['return_percent'])
            worst_session = min(performance_data, key=lambda x: performance_data[x]['return_percent'])
            
            return {
                "session_data": performance_data,
                "best_performer": {
                    "session": best_session,
                    "return": performance_data[best_session]['return_percent']
                },
                "worst_performer": {
                    "session": worst_session,
                    "return": performance_data[worst_session]['return_percent']
                },
                "average_return": sum(data['return_percent'] for data in performance_data.values()) / len(performance_data)
            }
        
        return {"error": "No valid performance data found"}
    
    def _analyze_trends(self, session_analyses: Dict[str, Any]) -> List[str]:
        """Analyze trends across sessions."""
        trends = []
        
        # Extract performance data
        returns = []
        trade_counts = []
        
        for analysis in session_analyses.values():
            if "error" not in analysis:
                perf = analysis.get('trading_performance', {})
                returns.append(perf.get('total_return_percent', 0))
                trade_counts.append(perf.get('total_trades', 0))
        
        if len(returns) >= 2:
            # Analyze return trend
            if returns[-1] > returns[0]:
                trends.append("Performance is improving over time")
            elif returns[-1] < returns[0]:
                trends.append("Performance is declining over time")
            else:
                trends.append("Performance is stable over time")
            
            # Analyze activity trend
            if trade_counts[-1] > trade_counts[0]:
                trends.append("Trading activity is increasing")
            elif trade_counts[-1] < trade_counts[0]:
                trends.append("Trading activity is decreasing")
        
        return trends
    
    def _identify_best_practices(self, session_analyses: Dict[str, Any]) -> List[str]:
        """Identify best practices from successful sessions."""
        best_practices = []
        
        # Find sessions with positive returns
        successful_sessions = []
        for session_name, analysis in session_analyses.items():
            if "error" not in analysis:
                perf = analysis.get('trading_performance', {})
                if perf.get('total_return_percent', 0) > 0:
                    successful_sessions.append((session_name, analysis))
        
        if successful_sessions:
            best_practices.append(f"Identified {len(successful_sessions)} successful sessions for pattern analysis")
            
            # Analyze common characteristics
            avg_trades = sum(analysis.get('trading_performance', {}).get('total_trades', 0) 
                           for _, analysis in successful_sessions) / len(successful_sessions)
            best_practices.append(f"Successful sessions averaged {avg_trades:.1f} trades")
        
        return best_practices
    
    def _identify_improvements(self, session_analyses: Dict[str, Any]) -> List[str]:
        """Identify improvement opportunities."""
        improvements = []
        
        # Analyze unsuccessful sessions
        unsuccessful_sessions = []
        for session_name, analysis in session_analyses.items():
            if "error" not in analysis:
                perf = analysis.get('trading_performance', {})
                if perf.get('total_return_percent', 0) < 0:
                    unsuccessful_sessions.append((session_name, analysis))
        
        if unsuccessful_sessions:
            improvements.append(f"Focus on improving {len(unsuccessful_sessions)} underperforming sessions")
            
            # Common issues
            low_success_rate_sessions = [s for s, a in unsuccessful_sessions 
                                       if a.get('trading_performance', {}).get('success_rate_percent', 0) < 40]
            if low_success_rate_sessions:
                improvements.append("Multiple sessions show low success rates - review entry/exit criteria")
        
        return improvements