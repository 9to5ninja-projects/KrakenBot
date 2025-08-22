#!/usr/bin/env python3
"""
🌍 MARKET OPENING INTELLIGENCE SYSTEM 🌍
Tracks 3 major market opens + pre-opening analysis
"""

import datetime
import pytz
import json
import time
from pathlib import Path
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MarketSession:
    name: str
    timezone: str
    open_time: str  # HH:MM format
    pre_analysis_minutes: int = 60

@dataclass
class OpeningAnalysis:
    session: str
    timestamp: str
    pre_signals: Dict
    opening_prediction: str
    rebound_probability: float
    confidence: float
    manipulation_risk: str

class MarketOpeningAI:
    """AI system for market opening analysis"""
    
    def __init__(self):
        self.sessions = {
            'ASIAN': MarketSession('Asian', 'Asia/Tokyo', '09:00', 60),
            'EUROPEAN': MarketSession('European', 'Europe/London', '08:00', 60), 
            'US': MarketSession('US', 'America/New_York', '09:30', 60)
        }
        
        self.data_dir = Path("data/market_openings")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.analysis_history = []
        self.load_history()
    
    def get_next_market_opening(self) -> Optional[tuple]:
        """Get next market opening time"""
        now = datetime.datetime.now(pytz.UTC)
        next_openings = []
        
        for session_name, session in self.sessions.items():
            # Get timezone
            tz = pytz.timezone(session.timezone)
            
            # Today's opening time
            today = now.astimezone(tz).date()
            opening_time = datetime.datetime.combine(
                today, 
                datetime.time.fromisoformat(session.open_time)
            )
            opening_time = tz.localize(opening_time).astimezone(pytz.UTC)
            
            # If already passed, get tomorrow's
            if opening_time <= now:
                tomorrow = today + datetime.timedelta(days=1)
                opening_time = datetime.datetime.combine(
                    tomorrow,
                    datetime.time.fromisoformat(session.open_time)
                )
                opening_time = tz.localize(opening_time).astimezone(pytz.UTC)
            
            next_openings.append((session_name, opening_time, session))
        
        # Sort by time and return next
        next_openings.sort(key=lambda x: x[1])
        return next_openings[0] if next_openings else None
    
    def analyze_pre_opening_signals(self, session_name: str) -> Dict:
        """Analyze signals before market opening"""
        
        # Get current market data (simulated for now)
        current_price = self.get_current_price()
        volume_data = self.get_volume_data()
        
        signals = {
            'volume_buildup': self.detect_volume_accumulation(volume_data),
            'price_positioning': self.analyze_support_resistance(current_price),
            'institutional_flow': self.detect_large_orders(),
            'cross_market_correlation': self.check_other_markets(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return signals
    
    def detect_volume_accumulation(self, volume_data: List) -> Dict:
        """Detect volume building up before opening"""
        if not volume_data or len(volume_data) < 10:
            return {'status': 'insufficient_data', 'score': 0}
        
        recent_avg = sum(volume_data[-5:]) / 5
        historical_avg = sum(volume_data[-20:-5]) / 15
        
        volume_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
        
        if volume_ratio > 1.5:
            return {'status': 'high_accumulation', 'score': 0.8, 'ratio': volume_ratio}
        elif volume_ratio > 1.2:
            return {'status': 'moderate_accumulation', 'score': 0.6, 'ratio': volume_ratio}
        else:
            return {'status': 'normal_volume', 'score': 0.3, 'ratio': volume_ratio}
    
    def analyze_support_resistance(self, current_price: float) -> Dict:
        """Analyze price positioning vs support/resistance"""
        # Simplified support/resistance analysis
        # In real implementation, use historical price levels
        
        support_level = current_price * 0.995  # 0.5% below
        resistance_level = current_price * 1.005  # 0.5% above
        
        distance_to_support = (current_price - support_level) / support_level
        distance_to_resistance = (resistance_level - current_price) / current_price
        
        if distance_to_resistance < 0.002:  # Very close to resistance
            positioning = 'near_resistance'
            score = 0.7
        elif distance_to_support < 0.002:  # Very close to support
            positioning = 'near_support'
            score = 0.7
        else:
            positioning = 'neutral'
            score = 0.5
        
        return {
            'positioning': positioning,
            'score': score,
            'support': support_level,
            'resistance': resistance_level,
            'current': current_price
        }
    
    def detect_large_orders(self) -> Dict:
        """Detect institutional order flow (simulated)"""
        # In real implementation, analyze order book depth
        import random
        
        # Simulate institutional activity detection
        institutional_score = random.uniform(0.2, 0.9)
        
        if institutional_score > 0.7:
            return {'status': 'high_institutional', 'score': institutional_score}
        elif institutional_score > 0.5:
            return {'status': 'moderate_institutional', 'score': institutional_score}
        else:
            return {'status': 'retail_dominated', 'score': institutional_score}
    
    def check_other_markets(self) -> Dict:
        """Check correlation with other markets"""
        # Simplified cross-market analysis
        # In real implementation, check forex, commodities, indices
        
        correlations = {
            'forex_usd': 0.6,  # Simulated correlation
            'btc_correlation': 0.8,
            'stock_futures': 0.4
        }
        
        avg_correlation = sum(correlations.values()) / len(correlations)
        
        return {
            'correlations': correlations,
            'average_correlation': avg_correlation,
            'market_sync': 'high' if avg_correlation > 0.6 else 'moderate' if avg_correlation > 0.4 else 'low'
        }
    
    def predict_opening_move(self, pre_signals: Dict) -> tuple:
        """Predict opening direction and confidence"""
        
        # Combine all signals for prediction
        volume_score = pre_signals['volume_buildup']['score']
        position_score = pre_signals['price_positioning']['score']
        institutional_score = pre_signals['institutional_flow']['score']
        correlation_score = pre_signals['cross_market_correlation']['average_correlation']
        
        # Weighted prediction algorithm
        bullish_signals = 0
        bearish_signals = 0
        
        # Volume analysis
        if pre_signals['volume_buildup']['status'] == 'high_accumulation':
            bullish_signals += 0.3
        
        # Position analysis
        if pre_signals['price_positioning']['positioning'] == 'near_support':
            bullish_signals += 0.25
        elif pre_signals['price_positioning']['positioning'] == 'near_resistance':
            bearish_signals += 0.25
        
        # Institutional flow
        if institutional_score > 0.7:
            bullish_signals += 0.2
        
        # Overall confidence
        total_signals = bullish_signals + bearish_signals
        confidence = min(0.95, total_signals)
        
        if bullish_signals > bearish_signals:
            direction = 'BULLISH'
        elif bearish_signals > bullish_signals:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return direction, confidence
    
    def calculate_rebound_probability(self, opening_prediction: str) -> float:
        """Calculate probability of rebound after opening move"""
        # Your insight: There's always a rebound, sometimes twice
        
        base_rebound_prob = 0.85  # 85% chance of rebound (your observation)
        
        # Adjust based on prediction confidence
        if opening_prediction == 'NEUTRAL':
            return 0.95  # Neutral moves almost always rebound
        else:
            return base_rebound_prob
    
    def detect_manipulation_risk(self, pre_signals: Dict) -> str:
        """Detect potential manipulation/fake reversal risk"""
        
        # Your insights on corporate/bot manipulation
        risk_factors = 0
        
        # Perfect technical setups are suspicious
        if pre_signals['price_positioning']['score'] > 0.9:
            risk_factors += 1
        
        # Unusual volume patterns
        if pre_signals['volume_buildup']['status'] == 'high_accumulation':
            volume_ratio = pre_signals['volume_buildup']['ratio']
            if volume_ratio > 2.0:  # Extremely high volume = suspicious
                risk_factors += 1
        
        # Low correlation = potential manipulation
        if pre_signals['cross_market_correlation']['average_correlation'] < 0.3:
            risk_factors += 1
        
        if risk_factors >= 2:
            return 'HIGH'
        elif risk_factors == 1:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def perform_opening_analysis(self, session_name: str) -> OpeningAnalysis:
        """Complete opening analysis"""
        
        # Get pre-opening signals
        pre_signals = self.analyze_pre_opening_signals(session_name)
        
        # Predict opening move
        direction, confidence = self.predict_opening_move(pre_signals)
        
        # Calculate rebound probability
        rebound_prob = self.calculate_rebound_probability(direction)
        
        # Assess manipulation risk
        manipulation_risk = self.detect_manipulation_risk(pre_signals)
        
        analysis = OpeningAnalysis(
            session=session_name,
            timestamp=datetime.datetime.now().isoformat(),
            pre_signals=pre_signals,
            opening_prediction=direction,
            rebound_probability=rebound_prob,
            confidence=confidence,
            manipulation_risk=manipulation_risk
        )
        
        # Save analysis
        self.save_analysis(analysis)
        
        return analysis
    
    def get_current_price(self) -> float:
        """Get current ETH/CAD price (simulated)"""
        # In real implementation, get from Kraken API
        import random
        return 3200 + random.uniform(-50, 50)
    
    def get_volume_data(self) -> List[float]:
        """Get recent volume data (simulated)"""
        # In real implementation, get from exchange
        import random
        return [random.uniform(100, 1000) for _ in range(20)]
    
    def save_analysis(self, analysis: OpeningAnalysis):
        """Save analysis to file"""
        self.analysis_history.append(analysis)
        
        # Save to file
        filename = f"opening_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.data_dir / filename
        
        # Convert to dict for JSON serialization
        analysis_dict = {
            'session': analysis.session,
            'timestamp': analysis.timestamp,
            'pre_signals': analysis.pre_signals,
            'opening_prediction': analysis.opening_prediction,
            'rebound_probability': analysis.rebound_probability,
            'confidence': analysis.confidence,
            'manipulation_risk': analysis.manipulation_risk
        }
        
        # Load existing data
        existing_data = []
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Add new analysis
        existing_data.append(analysis_dict)
        
        # Save back
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def load_history(self):
        """Load analysis history"""
        try:
            today_file = f"opening_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.json"
            filepath = self.data_dir / today_file
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.analysis_history = data
        except:
            self.analysis_history = []
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of today's analyses"""
        if not self.analysis_history:
            return {'status': 'no_data'}
        
        total_analyses = len(self.analysis_history)
        
        # Handle both dict and object types
        confidences = []
        predictions = []
        
        for a in self.analysis_history:
            if hasattr(a, 'confidence'):
                confidences.append(a.confidence)
                predictions.append(a.opening_prediction)
            else:
                confidences.append(a.get('confidence', 0))
                predictions.append(a.get('opening_prediction', 'NEUTRAL'))
        
        avg_confidence = sum(confidences) / total_analyses if confidences else 0
        bullish_count = predictions.count('BULLISH')
        bearish_count = predictions.count('BEARISH')
        neutral_count = predictions.count('NEUTRAL')
        
        return {
            'total_analyses': total_analyses,
            'average_confidence': avg_confidence,
            'predictions': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'latest_analysis': self.analysis_history[-1] if self.analysis_history else None
        }

def main():
    """Test the market opening AI"""
    print("🌍 Market Opening Intelligence System")
    print("=" * 50)
    
    ai = MarketOpeningAI()
    
    # Get next market opening
    next_opening = ai.get_next_market_opening()
    if next_opening:
        session_name, opening_time, session = next_opening
        print(f"Next Market Opening: {session_name}")
        print(f"Time: {opening_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Pre-analysis starts: {(opening_time - datetime.timedelta(minutes=session.pre_analysis_minutes)).strftime('%H:%M:%S UTC')}")
        print()
    
    # Perform analysis for demonstration
    print("Performing sample analysis...")
    analysis = ai.perform_opening_analysis('US')
    
    print(f"Session: {analysis.session}")
    print(f"Prediction: {analysis.opening_prediction}")
    print(f"Confidence: {analysis.confidence:.2%}")
    print(f"Rebound Probability: {analysis.rebound_probability:.2%}")
    print(f"Manipulation Risk: {analysis.manipulation_risk}")
    print()
    
    # Show summary
    summary = ai.get_analysis_summary()
    print("Analysis Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()