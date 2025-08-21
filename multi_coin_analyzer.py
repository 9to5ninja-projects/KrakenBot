"""
Multi-Coin Trading Analysis for KrakenBot
Analyzes multiple cryptocurrency pairs for optimal trading opportunities
"""

import asyncio
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from exchange import ExchangeManager
from advanced_technical_indicators import AdvancedTechnicalAnalyzer
from ai_strategy_optimizer import AIStrategyOptimizer, MarketCondition

@dataclass
class CoinAnalysis:
    """Analysis results for a specific coin pair."""
    pair: str
    current_price: float
    volume_24h: float
    volatility_24h: float
    volatility_7d: float
    liquidity_score: float
    technical_score: float
    ai_score: float
    composite_score: float
    recommendation: str
    confidence: float
    risk_level: str
    opportunity_rating: int  # 1-10 scale

@dataclass
class TradingOpportunity:
    """Represents a trading opportunity."""
    pair: str
    signal: str  # BUY/SELL/HOLD
    entry_price: float
    target_price: float
    stop_loss: float
    expected_profit_pct: float
    risk_reward_ratio: float
    confidence: float
    urgency: str  # LOW/MEDIUM/HIGH
    reasoning: str

class MultiCoinAnalyzer:
    """Analyzes multiple cryptocurrency pairs for trading opportunities."""
    
    def __init__(self):
        self.exchange = ExchangeManager()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_optimizer = AIStrategyOptimizer()
        
        # Define supported pairs with their characteristics
        self.supported_pairs = {
            'ETH/CAD': {
                'min_order_size': 0.001,
                'typical_volatility': 0.05,  # 5% daily
                'liquidity_tier': 1,  # High liquidity
                'trading_hours': 'always'
            },
            'BTC/CAD': {
                'min_order_size': 0.0001,
                'typical_volatility': 0.04,  # 4% daily
                'liquidity_tier': 1,  # High liquidity
                'trading_hours': 'always'
            },
            'SOL/CAD': {
                'min_order_size': 0.01,
                'typical_volatility': 0.08,  # 8% daily
                'liquidity_tier': 2,  # Medium liquidity
                'trading_hours': 'always'
            },
            'XRP/CAD': {
                'min_order_size': 1.0,
                'typical_volatility': 0.06,  # 6% daily
                'liquidity_tier': 2,  # Medium liquidity
                'trading_hours': 'always'
            },
            'DOGE/CAD': {
                'min_order_size': 10.0,
                'typical_volatility': 0.12,  # 12% daily
                'liquidity_tier': 3,  # Lower liquidity
                'trading_hours': 'always'
            },
            'USDT/CAD': {
                'min_order_size': 1.0,
                'typical_volatility': 0.01,  # 1% daily (stablecoin)
                'liquidity_tier': 1,  # High liquidity
                'trading_hours': 'always'
            }
        }
        
        self.analysis_cache = {}
        self.last_update = {}
    
    async def get_market_data(self, pair: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get market data for a specific pair."""
        try:
            # Get OHLCV data from exchange
            ohlcv = self.exchange.get_historical_ohlcv(pair, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            return df
        except Exception as e:
            print(f"❌ Error fetching data for {pair}: {e}")
            return pd.DataFrame()
    
    def calculate_volatility_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Calculate various volatility metrics."""
        if price_data.empty:
            return {'volatility_24h': 0, 'volatility_7d': 0, 'volatility_score': 0}
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # 24-hour volatility (last 24 data points for hourly data)
        vol_24h = returns.tail(24).std() * np.sqrt(24) if len(returns) >= 24 else returns.std() * np.sqrt(len(returns))
        
        # 7-day volatility (last 168 data points for hourly data)
        vol_7d = returns.tail(168).std() * np.sqrt(168) if len(returns) >= 168 else returns.std() * np.sqrt(len(returns))
        
        # Volatility score (0-100, higher = more volatile)
        # Compare to typical crypto volatility (5% daily)
        vol_score = min(100, (vol_24h / 0.05) * 100)
        
        return {
            'volatility_24h': vol_24h,
            'volatility_7d': vol_7d,
            'volatility_score': vol_score,
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
        }
    
    def calculate_liquidity_score(self, pair: str, price_data: pd.DataFrame) -> float:
        """Calculate liquidity score for a pair."""
        if price_data.empty:
            return 0
        
        # Base score from pair characteristics
        pair_info = self.supported_pairs.get(pair, {})
        tier = pair_info.get('liquidity_tier', 3)
        base_score = {1: 90, 2: 70, 3: 50}.get(tier, 30)
        
        # Adjust based on volume
        avg_volume = price_data['volume'].tail(24).mean()
        volume_score = min(20, np.log10(avg_volume + 1) * 5)  # Log scale for volume
        
        # Adjust based on price stability (less volatility = better liquidity)
        vol_metrics = self.calculate_volatility_metrics(price_data)
        stability_score = max(0, 10 - vol_metrics['volatility_score'] / 10)
        
        total_score = base_score + volume_score + stability_score
        return min(100, total_score)
    
    def calculate_technical_score(self, pair: str, price_data: pd.DataFrame) -> Dict:
        """Calculate technical analysis score."""
        if price_data.empty:
            return {'score': 0, 'signals': {}, 'confidence': 0}
        
        # Get technical analysis
        analysis = self.technical_analyzer.analyze_pair(price_data, pair)
        
        if 'error' in analysis:
            return {'score': 0, 'signals': {}, 'confidence': 0}
        
        # Extract composite signal
        composite = analysis.get('composite_signal', {})
        signal = composite.get('signal', 'HOLD')
        strength = composite.get('strength', 0)
        confidence = composite.get('confidence', 0)
        agreement = composite.get('agreement_level', 0)
        
        # Calculate technical score (0-100)
        if signal == 'BUY':
            score = 50 + (strength * 0.5)  # 50-100 for buy signals
        elif signal == 'SELL':
            score = 50 - (strength * 0.5)  # 0-50 for sell signals
        else:
            score = 50  # Neutral
        
        # Adjust for agreement level
        score = score * (agreement / 100)
        
        return {
            'score': score,
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'agreement': agreement,
            'indicators': analysis.get('indicators', {})
        }
    
    def calculate_ai_score(self, pair: str, price_data: pd.DataFrame) -> Dict:
        """Calculate AI-based score."""
        if price_data.empty:
            return {'score': 50, 'prediction': {}, 'confidence': 0}
        
        try:
            # Get AI analysis
            analysis = self.ai_optimizer.get_current_market_analysis(pair, price_data)
            
            ai_prediction = analysis.get('ai_prediction', {})
            recommendation = analysis.get('recommendation', {})
            
            # Convert AI recommendation to score
            ai_rec = ai_prediction.get('recommendation', 'NEUTRAL')
            success_prob = ai_prediction.get('success_probability', 0.5)
            expected_profit = ai_prediction.get('expected_profit', 0)
            
            # Calculate score based on AI recommendation
            if ai_rec == 'STRONG_BUY':
                score = 80 + (success_prob * 20)
            elif ai_rec == 'BUY':
                score = 60 + (success_prob * 20)
            elif ai_rec == 'AVOID':
                score = 20 - (success_prob * 20)
            else:  # NEUTRAL
                score = 50
            
            # Adjust for expected profit
            if expected_profit > 0:
                score += min(10, expected_profit * 1000)
            else:
                score += max(-10, expected_profit * 1000)
            
            return {
                'score': max(0, min(100, score)),
                'recommendation': ai_rec,
                'success_probability': success_prob,
                'expected_profit': expected_profit,
                'confidence': ai_prediction.get('confidence', 0)
            }
        except Exception as e:
            print(f"⚠️ AI analysis error for {pair}: {e}")
            return {'score': 50, 'prediction': {}, 'confidence': 0}
    
    async def analyze_pair(self, pair: str) -> CoinAnalysis:
        """Comprehensive analysis of a trading pair."""
        print(f"Analyzing {pair}...")
        
        # Get market data
        price_data = await self.get_market_data(pair)
        
        if price_data.empty:
            return CoinAnalysis(
                pair=pair,
                current_price=0,
                volume_24h=0,
                volatility_24h=0,
                volatility_7d=0,
                liquidity_score=0,
                technical_score=0,
                ai_score=0,
                composite_score=0,
                recommendation='NO_DATA',
                confidence=0,
                risk_level='UNKNOWN',
                opportunity_rating=0
            )
        
        # Current price and volume
        current_price = price_data['close'].iloc[-1]
        volume_24h = price_data['volume'].tail(24).sum()
        
        # Calculate metrics
        vol_metrics = self.calculate_volatility_metrics(price_data)
        liquidity_score = self.calculate_liquidity_score(pair, price_data)
        technical_analysis = self.calculate_technical_score(pair, price_data)
        ai_analysis = self.calculate_ai_score(pair, price_data)
        
        # Calculate composite score
        # Weight: Technical 40%, AI 35%, Liquidity 15%, Volatility 10%
        technical_score = technical_analysis['score']
        ai_score = ai_analysis['score']
        volatility_bonus = min(20, vol_metrics['volatility_score'] / 5)  # Bonus for good volatility
        
        composite_score = (
            technical_score * 0.40 +
            ai_score * 0.35 +
            liquidity_score * 0.15 +
            volatility_bonus * 0.10
        )
        
        # Generate recommendation
        recommendation, confidence, risk_level = self._generate_recommendation(
            composite_score, technical_analysis, ai_analysis, vol_metrics
        )
        
        # Calculate opportunity rating (1-10)
        opportunity_rating = max(1, min(10, int(composite_score / 10)))
        
        return CoinAnalysis(
            pair=pair,
            current_price=current_price,
            volume_24h=volume_24h,
            volatility_24h=vol_metrics['volatility_24h'],
            volatility_7d=vol_metrics['volatility_7d'],
            liquidity_score=liquidity_score,
            technical_score=technical_score,
            ai_score=ai_score,
            composite_score=composite_score,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            opportunity_rating=opportunity_rating
        )
    
    def _generate_recommendation(self, composite_score: float, technical_analysis: Dict, 
                               ai_analysis: Dict, vol_metrics: Dict) -> Tuple[str, float, str]:
        """Generate trading recommendation based on all analyses."""
        # Determine recommendation
        if composite_score >= 75:
            recommendation = 'STRONG_BUY'
            confidence = min(95, composite_score)
        elif composite_score >= 60:
            recommendation = 'BUY'
            confidence = min(85, composite_score)
        elif composite_score <= 25:
            recommendation = 'STRONG_SELL'
            confidence = min(95, 100 - composite_score)
        elif composite_score <= 40:
            recommendation = 'SELL'
            confidence = min(85, 100 - composite_score)
        else:
            recommendation = 'HOLD'
            confidence = 50
        
        # Determine risk level
        volatility = vol_metrics['volatility_24h']
        if volatility > 0.10:  # >10% daily volatility
            risk_level = 'HIGH'
        elif volatility > 0.06:  # >6% daily volatility
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return recommendation, confidence, risk_level
    
    async def scan_all_pairs(self) -> List[CoinAnalysis]:
        """Scan all supported pairs for trading opportunities."""
        print("🔍 Scanning all supported pairs...")
        
        analyses = []
        for pair in self.supported_pairs.keys():
            try:
                analysis = await self.analyze_pair(pair)
                analyses.append(analysis)
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"❌ Error analyzing {pair}: {e}")
        
        # Sort by composite score (best opportunities first)
        analyses.sort(key=lambda x: x.composite_score, reverse=True)
        
        return analyses
    
    def identify_trading_opportunities(self, analyses: List[CoinAnalysis], 
                                     max_opportunities: int = 3) -> List[TradingOpportunity]:
        """Identify the best trading opportunities from analyses."""
        opportunities = []
        
        for analysis in analyses[:max_opportunities]:
            if analysis.recommendation in ['STRONG_BUY', 'BUY']:
                opportunity = self._create_buy_opportunity(analysis)
                if opportunity:
                    opportunities.append(opportunity)
            elif analysis.recommendation in ['STRONG_SELL', 'SELL']:
                opportunity = self._create_sell_opportunity(analysis)
                if opportunity:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _create_buy_opportunity(self, analysis: CoinAnalysis) -> Optional[TradingOpportunity]:
        """Create a buy trading opportunity."""
        if analysis.current_price <= 0:
            return None
        
        # Calculate targets based on volatility and technical analysis
        volatility = analysis.volatility_24h
        
        # Target profit: 1.5x daily volatility or minimum 0.5%
        target_profit_pct = max(0.005, volatility * 1.5)
        target_price = analysis.current_price * (1 + target_profit_pct)
        
        # Stop loss: 0.5x daily volatility or maximum 2%
        stop_loss_pct = min(0.02, volatility * 0.5)
        stop_loss = analysis.current_price * (1 - stop_loss_pct)
        
        # Risk-reward ratio
        risk_reward_ratio = target_profit_pct / stop_loss_pct
        
        # Urgency based on confidence and volatility
        if analysis.confidence > 80 and analysis.volatility_24h > 0.05:
            urgency = 'HIGH'
        elif analysis.confidence > 60:
            urgency = 'MEDIUM'
        else:
            urgency = 'LOW'
        
        reasoning = f"Technical score: {analysis.technical_score:.1f}, AI score: {analysis.ai_score:.1f}, " \
                   f"Volatility: {analysis.volatility_24h:.3f}, Risk level: {analysis.risk_level}"
        
        return TradingOpportunity(
            pair=analysis.pair,
            signal='BUY',
            entry_price=analysis.current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_pct=target_profit_pct,
            risk_reward_ratio=risk_reward_ratio,
            confidence=analysis.confidence,
            urgency=urgency,
            reasoning=reasoning
        )
    
    def _create_sell_opportunity(self, analysis: CoinAnalysis) -> Optional[TradingOpportunity]:
        """Create a sell trading opportunity (for existing positions)."""
        if analysis.current_price <= 0:
            return None
        
        # For sell opportunities, we're looking to exit positions
        volatility = analysis.volatility_24h
        
        # Target: Sell before further decline
        target_profit_pct = -max(0.005, volatility * 1.0)  # Negative for sell
        target_price = analysis.current_price * (1 + target_profit_pct)
        
        # Stop loss: If price goes up instead
        stop_loss_pct = min(0.015, volatility * 0.3)
        stop_loss = analysis.current_price * (1 + stop_loss_pct)
        
        risk_reward_ratio = abs(target_profit_pct) / stop_loss_pct
        
        urgency = 'HIGH' if analysis.confidence > 70 else 'MEDIUM'
        
        reasoning = f"Sell signal - Technical score: {analysis.technical_score:.1f}, " \
                   f"AI recommends avoiding, Risk level: {analysis.risk_level}"
        
        return TradingOpportunity(
            pair=analysis.pair,
            signal='SELL',
            entry_price=analysis.current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_pct=target_profit_pct,
            risk_reward_ratio=risk_reward_ratio,
            confidence=analysis.confidence,
            urgency=urgency,
            reasoning=reasoning
        )
    
    def generate_trading_report(self, analyses: List[CoinAnalysis], 
                              opportunities: List[TradingOpportunity]) -> Dict:
        """Generate comprehensive trading report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'market_overview': {
                'total_pairs_analyzed': len(analyses),
                'opportunities_found': len(opportunities),
                'avg_volatility': np.mean([a.volatility_24h for a in analyses]),
                'avg_liquidity_score': np.mean([a.liquidity_score for a in analyses]),
                'market_sentiment': self._assess_market_sentiment(analyses)
            },
            'top_opportunities': [asdict(opp) for opp in opportunities[:3]],
            'pair_rankings': [
                {
                    'pair': a.pair,
                    'score': a.composite_score,
                    'recommendation': a.recommendation,
                    'confidence': a.confidence,
                    'risk_level': a.risk_level,
                    'opportunity_rating': a.opportunity_rating
                }
                for a in analyses
            ],
            'risk_assessment': self._assess_overall_risk(analyses),
            'recommendations': self._generate_portfolio_recommendations(analyses, opportunities)
        }
        
        return report
    
    def _assess_market_sentiment(self, analyses: List[CoinAnalysis]) -> str:
        """Assess overall market sentiment."""
        buy_signals = sum(1 for a in analyses if a.recommendation in ['STRONG_BUY', 'BUY'])
        sell_signals = sum(1 for a in analyses if a.recommendation in ['STRONG_SELL', 'SELL'])
        total = len(analyses)
        
        if buy_signals > total * 0.6:
            return 'BULLISH'
        elif sell_signals > total * 0.6:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _assess_overall_risk(self, analyses: List[CoinAnalysis]) -> Dict:
        """Assess overall portfolio risk."""
        high_risk = sum(1 for a in analyses if a.risk_level == 'HIGH')
        medium_risk = sum(1 for a in analyses if a.risk_level == 'MEDIUM')
        low_risk = sum(1 for a in analyses if a.risk_level == 'LOW')
        
        avg_volatility = np.mean([a.volatility_24h for a in analyses])
        
        if high_risk > len(analyses) * 0.5:
            overall_risk = 'HIGH'
        elif medium_risk + high_risk > len(analyses) * 0.7:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'high_risk_pairs': high_risk,
            'medium_risk_pairs': medium_risk,
            'low_risk_pairs': low_risk,
            'average_volatility': avg_volatility,
            'risk_recommendation': 'Diversify across risk levels' if high_risk > 2 else 'Acceptable risk distribution'
        }
    
    def _generate_portfolio_recommendations(self, analyses: List[CoinAnalysis], 
                                          opportunities: List[TradingOpportunity]) -> List[str]:
        """Generate portfolio-level recommendations."""
        recommendations = []
        
        # Diversification recommendation
        high_score_pairs = [a for a in analyses if a.composite_score > 70]
        if len(high_score_pairs) > 3:
            recommendations.append("Consider diversifying across top 3-4 pairs rather than concentrating in one")
        
        # Risk management
        high_risk_opportunities = [o for o in opportunities if o.confidence > 80]
        if len(high_risk_opportunities) > 1:
            recommendations.append("Multiple high-confidence opportunities available - consider position sizing")
        
        # Market timing
        avg_volatility = np.mean([a.volatility_24h for a in analyses])
        if avg_volatility > 0.08:
            recommendations.append("High market volatility detected - consider smaller position sizes")
        elif avg_volatility < 0.03:
            recommendations.append("Low volatility environment - may need more aggressive thresholds")
        
        # Liquidity considerations
        low_liquidity_pairs = [a for a in analyses if a.liquidity_score < 50]
        if low_liquidity_pairs:
            recommendations.append(f"Be cautious with low liquidity pairs: {', '.join([a.pair for a in low_liquidity_pairs])}")
        
        return recommendations

async def main():
    """Test the multi-coin analyzer."""
    analyzer = MultiCoinAnalyzer()
    
    print("🚀 Multi-Coin Trading Analysis")
    print("=" * 60)
    
    # Scan all pairs
    analyses = await analyzer.scan_all_pairs()
    
    # Identify opportunities
    opportunities = analyzer.identify_trading_opportunities(analyses)
    
    # Generate report
    report = analyzer.generate_trading_report(analyses, opportunities)
    
    # Display results
    print(f"\n📊 MARKET OVERVIEW")
    print(f"Pairs analyzed: {report['market_overview']['total_pairs_analyzed']}")
    print(f"Opportunities found: {report['market_overview']['opportunities_found']}")
    print(f"Market sentiment: {report['market_overview']['market_sentiment']}")
    print(f"Average volatility: {report['market_overview']['avg_volatility']:.3f}")
    
    print(f"\n🎯 TOP OPPORTUNITIES")
    for i, opp in enumerate(report['top_opportunities'], 1):
        print(f"{i}. {opp['pair']} - {opp['signal']}")
        print(f"   Entry: ${opp['entry_price']:.2f}")
        print(f"   Target: ${opp['target_price']:.2f} ({opp['expected_profit_pct']:.3f}%)")
        print(f"   Stop Loss: ${opp['stop_loss']:.2f}")
        print(f"   Confidence: {opp['confidence']:.1f}%")
        print(f"   Urgency: {opp['urgency']}")
    
    print(f"\n📈 PAIR RANKINGS")
    for pair_info in report['pair_rankings']:
        print(f"{pair_info['pair']}: {pair_info['score']:.1f} - {pair_info['recommendation']} "
              f"(Rating: {pair_info['opportunity_rating']}/10)")
    
    print(f"\n🛡️ RISK ASSESSMENT")
    risk = report['risk_assessment']
    print(f"Overall risk: {risk['overall_risk']}")
    print(f"High risk pairs: {risk['high_risk_pairs']}")
    print(f"Average volatility: {risk['average_volatility']:.3f}")
    
    print(f"\n💡 RECOMMENDATIONS")
    for rec in report['recommendations']:
        print(f"• {rec}")

if __name__ == "__main__":
    asyncio.run(main())