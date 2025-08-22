"""
Enhanced Simulation System for KrakenBot
Integrates AI-driven strategy optimization, advanced technical analysis, and multi-coin support
"""

import asyncio
import json
import pandas as pd
import numpy as np
import pytz
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse

from exchange import ExchangeManager
from advanced_technical_indicators import AdvancedTechnicalAnalyzer
from ai_strategy_optimizer import AIStrategyOptimizer, StrategyParameters
from multi_coin_analyzer import MultiCoinAnalyzer, TradingOpportunity
from enhanced_nlp_analyzer import EnhancedNLPAnalyzer
from market_opening_ai import MarketOpeningAI

@dataclass
class EnhancedTradeExecution:
    """Enhanced trade execution with AI insights."""
    timestamp: datetime
    pair: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    cost: float
    fee: float
    portfolio_value_before: float
    portfolio_value_after: float
    technical_signals: Dict
    ai_prediction: Dict
    confidence_score: float
    risk_level: str
    reasoning: str

@dataclass
class PortfolioPosition:
    """Enhanced portfolio position tracking."""
    pair: str
    base_amount: float
    quote_amount: float
    avg_buy_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_timestamp: datetime
    technical_score: float
    ai_score: float
    stop_loss_price: float
    take_profit_price: float

class EnhancedSimulationSystem:
    """Enhanced simulation system with AI-driven trading."""
    
    def __init__(self, initial_capital: float = 1000.0, session_name: str = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.session_name = session_name or f"enhanced_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.exchange = ExchangeManager()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_optimizer = AIStrategyOptimizer()
        self.multi_coin_analyzer = MultiCoinAnalyzer()
        self.nlp_analyzer = EnhancedNLPAnalyzer()
        self.market_opening_ai = MarketOpeningAI()
        
        # Market opening tracking
        self.local_timezone = pytz.timezone('America/Toronto')  # Adjust to your timezone
        self.market_observation_window = 4  # 2 hours before + 2 hours after
        self.in_market_opening_window = False
        self.current_market_session = None
        self.last_opening_check = datetime.min
        
        # Portfolio tracking
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trade_history: List[EnhancedTradeExecution] = []
        self.portfolio_history: List[Dict] = []
        self.performance_metrics: Dict = {}
        
        # Enhanced strategy parameters (AI-optimized)
        self.strategy_params = StrategyParameters(
            buy_threshold=-0.003,  # -0.3%
            sell_threshold=0.005,  # +0.5%
            lookback_periods=5,
            min_trade_amount=50.0,  # Higher minimum for $1000 capital
            max_position_size=0.20,  # 20% max per position
            stop_loss_pct=-0.02,    # -2% stop loss
            take_profit_pct=0.015,  # +1.5% take profit
            volatility_threshold=0.02,
            volume_threshold=1.5,
            rsi_oversold=30,
            rsi_overbought=70,
            bollinger_buy_position=0.2,
            bollinger_sell_position=0.8
        )
        
        # Supported pairs with enhanced analysis (actual Kraken CAD pairs)
        self.trading_pairs = ['ETH/CAD', 'BTC/CAD', 'SOL/CAD', 'XRP/CAD']
        
        # Data storage
        self.data_dir = Path("data") / self.session_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(hours=1)  # Re-optimize every hour
        
        print(f" Enhanced Simulation System Initialized")
        print(f" Initial Capital: ${self.initial_capital:,.2f}")
        print(f" Trading Pairs: {', '.join(self.trading_pairs)}")
        print(f" Session: {self.session_name}")
    
    async def run_simulation(self, duration_hours: float = 4.0, check_interval: int = 60):
        """Run the enhanced simulation with AI-driven trading."""
        print(f"\n Starting Enhanced Simulation")
        print(f"Duration: {duration_hours} hours")
        print(f"Check Interval: {check_interval} seconds")
        print("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        step_count = 0
        self.simulation_start_time = start_time  # Store for timer display
        
        # Initial AI optimization
        await self._optimize_strategy_parameters()
        
        while datetime.now() < end_time:
            step_count += 1
            current_time = datetime.now()
            
            print(f"\n Step {step_count} - {current_time.strftime('%H:%M:%S')}")
            
            try:
                # 0. Check market opening windows (YOUR HUMAN INSIGHT INTEGRATION)
                market_opening_context = await self._check_market_opening_windows(current_time)
                
                # 1. Analyze all trading pairs (enhanced with opening context)
                market_analyses = await self._analyze_all_pairs(market_opening_context)
                
                # 2. Identify trading opportunities (adjusted for market openings)
                opportunities = await self._identify_opportunities(market_analyses, market_opening_context)
                
                # 3. Execute trades based on AI recommendations (with opening intelligence)
                executed_trades = await self._execute_ai_driven_trades(opportunities, market_opening_context)
                
                # 4. Update portfolio and positions
                await self._update_portfolio_state(market_analyses)
                
                # 5. Check for stop-loss and take-profit triggers
                await self._check_exit_conditions(market_analyses)
                
                # 6. Generate real-time insights (including opening analysis)
                insights = await self._generate_real_time_insights(market_analyses, executed_trades, market_opening_context)
                
                # 7. Periodic strategy optimization
                if current_time - self.last_optimization > self.optimization_interval:
                    await self._optimize_strategy_parameters()
                    self.last_optimization = current_time
                
                # 8. Save data (including opening intelligence)
                await self._save_step_data(step_count, market_analyses, opportunities, executed_trades, insights, market_opening_context)
                
                # 9. Display progress (with opening status and timer)
                self._display_progress(step_count, executed_trades, insights, market_opening_context, end_time)
                
            except Exception as e:
                print(f"ERROR: Error in step {step_count}: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait for next iteration
            await asyncio.sleep(check_interval)
        
        # Generate final analysis
        await self._generate_final_analysis()
        
        print(f"\n Simulation Complete!")
        print(f" Final Results saved to: {self.data_dir}")
    
    async def _analyze_all_pairs(self, market_opening_context: Dict = None) -> Dict[str, Dict]:
        """Analyze all trading pairs with technical and AI analysis."""
        analyses = {}
        
        for pair in self.trading_pairs:
            try:
                # Get market data
                price_data = await self.multi_coin_analyzer.get_market_data(pair, '1h', 100)
                
                if price_data.empty:
                    continue
                
                # Technical analysis
                technical_analysis = self.technical_analyzer.analyze_pair(price_data, pair)
                
                # AI analysis
                ai_analysis = self.ai_optimizer.get_current_market_analysis(pair, price_data)
                
                # Multi-coin analysis
                coin_analysis = await self.multi_coin_analyzer.analyze_pair(pair)
                
                analyses[pair] = {
                    'price_data': price_data,
                    'technical_analysis': technical_analysis,
                    'ai_analysis': ai_analysis,
                    'coin_analysis': coin_analysis,
                    'current_price': price_data['close'].iloc[-1],
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                print(f"WARNING: Error analyzing {pair}: {e}")
        
        return analyses
    
    async def _identify_opportunities(self, market_analyses: Dict, market_opening_context: Dict = None) -> List[TradingOpportunity]:
        """Identify trading opportunities using AI and technical analysis."""
        opportunities = []
        
        for pair, analysis in market_analyses.items():
            try:
                # Get AI recommendation
                ai_analysis = analysis['ai_analysis']
                ai_rec = ai_analysis.get('recommendation', {})
                final_signal = ai_rec.get('signal', 'HOLD')
                confidence = ai_rec.get('confidence', 0)
                
                # Get technical signal
                technical_analysis = analysis['technical_analysis']
                tech_signal = technical_analysis.get('composite_signal', {}).get('signal', 'HOLD')
                tech_strength = technical_analysis.get('composite_signal', {}).get('strength', 0)
                
                # Get coin analysis
                coin_analysis = analysis['coin_analysis']
                
                # Create opportunity if signals align
                if self._should_create_opportunity(final_signal, tech_signal, confidence, tech_strength):
                    opportunity = self._create_trading_opportunity(
                        pair, analysis, final_signal, confidence
                    )
                    if opportunity:
                        opportunities.append(opportunity)
                        
            except Exception as e:
                print(f"WARNING: Error identifying opportunities for {pair}: {e}")
        
        # Sort by confidence and expected profit
        opportunities.sort(key=lambda x: x.confidence * abs(x.expected_profit_pct), reverse=True)
        
        return opportunities[:3]  # Limit to top 3 opportunities
    
    def _should_create_opportunity(self, ai_signal: str, tech_signal: str, 
                                 ai_confidence: float, tech_strength: float) -> bool:
        """Determine if an opportunity should be created based on signal alignment."""
        # Require both AI and technical signals to agree for high confidence
        if ai_signal == 'BUY' and tech_signal == 'BUY':
            return ai_confidence > 60 and tech_strength > 40
        elif ai_signal == 'SELL' and tech_signal == 'SELL':
            return ai_confidence > 60 and tech_strength > 40
        # Allow single strong signal if very confident
        elif ai_signal in ['BUY', 'SELL'] and ai_confidence > 80:
            return True
        elif tech_signal in ['BUY', 'SELL'] and tech_strength > 70:
            return True
        
        return False
    
    def _create_trading_opportunity(self, pair: str, analysis: Dict, 
                                  signal: str, confidence: float) -> Optional[TradingOpportunity]:
        """Create a trading opportunity from analysis."""
        try:
            current_price = analysis['current_price']
            coin_analysis = analysis['coin_analysis']
            
            # Calculate position size based on confidence and risk
            base_position_size = self.strategy_params.max_position_size
            confidence_multiplier = confidence / 100
            position_size = base_position_size * confidence_multiplier
            
            # Calculate trade amount
            available_capital = self._get_available_capital()
            trade_amount = min(available_capital * position_size, 
                             available_capital * self.strategy_params.max_position_size)
            
            if trade_amount < self.strategy_params.min_trade_amount:
                return None
            
            # Calculate targets
            volatility = coin_analysis.volatility_24h
            
            if signal == 'BUY':
                # Target profit based on volatility and AI prediction
                target_profit_pct = max(self.strategy_params.sell_threshold, volatility * 1.5)
                target_price = current_price * (1 + target_profit_pct)
                stop_loss = current_price * (1 + self.strategy_params.stop_loss_pct)
                
                return TradingOpportunity(
                    pair=pair,
                    signal='BUY',
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_profit_pct=target_profit_pct,
                    risk_reward_ratio=target_profit_pct / abs(self.strategy_params.stop_loss_pct),
                    confidence=confidence,
                    urgency='HIGH' if confidence > 80 else 'MEDIUM',
                    reasoning=f"AI confidence: {confidence:.1f}%, Volatility: {volatility:.3f}"
                )
            
            elif signal == 'SELL' and pair in self.positions:
                # Only create sell opportunities for existing positions
                position = self.positions[pair]
                target_profit_pct = -max(0.005, volatility * 1.0)  # Negative for sell
                target_price = current_price * (1 + target_profit_pct)
                stop_loss = current_price * (1.015)  # 1.5% stop for sell
                
                return TradingOpportunity(
                    pair=pair,
                    signal='SELL',
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_profit_pct=target_profit_pct,
                    risk_reward_ratio=abs(target_profit_pct) / 0.015,
                    confidence=confidence,
                    urgency='HIGH',
                    reasoning=f"Exit signal - AI confidence: {confidence:.1f}%"
                )
            
        except Exception as e:
            print(f"WARNING: Error creating opportunity for {pair}: {e}")
        
        return None
    
    async def _execute_ai_driven_trades(self, opportunities: List[TradingOpportunity], market_opening_context: Dict = None) -> List[EnhancedTradeExecution]:
        """Execute trades based on AI-driven opportunities."""
        executed_trades = []
        
        # Apply market opening strategy adjustments
        if market_opening_context and market_opening_context.get('in_opening_window'):
            adjustments = market_opening_context.get('strategy_adjustments', {})
            
            # Skip trades if high manipulation risk
            if adjustments.get('avoid_trades'):
                print("🚨 Skipping trades due to high manipulation risk")
                return executed_trades
            
            # Wait for rebound if expected
            if adjustments.get('wait_for_rebound'):
                print("🔄 Waiting for rebound pattern before trading")
        
        for opportunity in opportunities:
            try:
                if opportunity.signal == 'BUY':
                    trade = await self._execute_buy_order(opportunity)
                elif opportunity.signal == 'SELL':
                    trade = await self._execute_sell_order(opportunity)
                else:
                    continue
                
                if trade:
                    executed_trades.append(trade)
                    
            except Exception as e:
                print(f"WARNING: Error executing trade for {opportunity.pair}: {e}")
        
        return executed_trades
    
    async def _execute_buy_order(self, opportunity: TradingOpportunity) -> Optional[EnhancedTradeExecution]:
        """Execute a buy order with enhanced tracking."""
        pair = opportunity.pair
        current_price = opportunity.entry_price
        
        # Calculate trade amount
        available_capital = self._get_available_capital()
        max_trade_amount = available_capital * self.strategy_params.max_position_size
        trade_amount = min(max_trade_amount, available_capital * 0.25)  # Max 25% per trade
        
        if trade_amount < self.strategy_params.min_trade_amount:
            return None
        
        # Calculate amount to buy
        amount = trade_amount / current_price
        fee = trade_amount * 0.004  # 0.4% fee
        total_cost = trade_amount + fee
        
        if total_cost > available_capital:
            return None
        
        # Update capital and create position
        portfolio_value_before = self._calculate_portfolio_value()
        self.current_capital -= total_cost
        
        # Create or update position
        if pair in self.positions:
            # Average down
            existing_pos = self.positions[pair]
            total_amount = existing_pos.base_amount + amount
            total_cost_basis = (existing_pos.avg_buy_price * existing_pos.base_amount) + trade_amount
            new_avg_price = total_cost_basis / total_amount
            
            self.positions[pair].base_amount = total_amount
            self.positions[pair].avg_buy_price = new_avg_price
        else:
            # New position
            self.positions[pair] = PortfolioPosition(
                pair=pair,
                base_amount=amount,
                quote_amount=trade_amount,
                avg_buy_price=current_price,
                current_price=current_price,
                unrealized_pnl=0,
                realized_pnl=0,
                entry_timestamp=datetime.now(),
                technical_score=0,  # Will be updated
                ai_score=0,  # Will be updated
                stop_loss_price=opportunity.stop_loss,
                take_profit_price=opportunity.target_price
            )
        
        portfolio_value_after = self._calculate_portfolio_value()
        
        # Create trade execution record
        trade = EnhancedTradeExecution(
            timestamp=datetime.now(),
            pair=pair,
            side='buy',
            amount=amount,
            price=current_price,
            cost=trade_amount,
            fee=fee,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            technical_signals={},  # Will be populated
            ai_prediction={},  # Will be populated
            confidence_score=opportunity.confidence,
            risk_level='MEDIUM',  # Will be calculated
            reasoning=opportunity.reasoning
        )
        
        self.trade_history.append(trade)
        
        print(f"SUCCESS: BUY {pair}: {amount:.6f} @ ${current_price:.2f} (${trade_amount:.2f})")
        
        return trade
    
    async def _execute_sell_order(self, opportunity: TradingOpportunity) -> Optional[EnhancedTradeExecution]:
        """Execute a sell order with enhanced tracking."""
        pair = opportunity.pair
        
        if pair not in self.positions:
            return None
        
        position = self.positions[pair]
        current_price = opportunity.entry_price
        
        # Sell entire position
        amount = position.base_amount
        trade_value = amount * current_price
        fee = trade_value * 0.004  # 0.4% fee
        net_proceeds = trade_value - fee
        
        # Calculate profit/loss
        cost_basis = position.base_amount * position.avg_buy_price
        realized_pnl = net_proceeds - cost_basis
        
        # Update capital
        portfolio_value_before = self._calculate_portfolio_value()
        self.current_capital += net_proceeds
        
        # Remove position
        del self.positions[pair]
        
        portfolio_value_after = self._calculate_portfolio_value()
        
        # Create trade execution record
        trade = EnhancedTradeExecution(
            timestamp=datetime.now(),
            pair=pair,
            side='sell',
            amount=amount,
            price=current_price,
            cost=trade_value,
            fee=fee,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            technical_signals={},
            ai_prediction={},
            confidence_score=opportunity.confidence,
            risk_level='MEDIUM',
            reasoning=f"{opportunity.reasoning}, Realized P&L: ${realized_pnl:.2f}"
        )
        
        self.trade_history.append(trade)
        
        print(f"SUCCESS: SELL {pair}: {amount:.6f} @ ${current_price:.2f} (${trade_value:.2f}, P&L: ${realized_pnl:.2f})")
        
        return trade
    
    async def _update_portfolio_state(self, market_analyses: Dict):
        """Update portfolio positions with current market data."""
        for pair, position in self.positions.items():
            if pair in market_analyses:
                current_price = market_analyses[pair]['current_price']
                position.current_price = current_price
                
                # Calculate unrealized P&L
                cost_basis = position.base_amount * position.avg_buy_price
                current_value = position.base_amount * current_price
                position.unrealized_pnl = current_value - cost_basis
                
                # Update technical and AI scores
                tech_analysis = market_analyses[pair]['technical_analysis']
                ai_analysis = market_analyses[pair]['ai_analysis']
                
                position.technical_score = tech_analysis.get('composite_signal', {}).get('strength', 0)
                position.ai_score = ai_analysis.get('ai_prediction', {}).get('confidence', 0)
    
    async def _check_exit_conditions(self, market_analyses: Dict):
        """Check for stop-loss and take-profit conditions."""
        positions_to_close = []
        
        for pair, position in self.positions.items():
            if pair not in market_analyses:
                continue
            
            current_price = position.current_price
            
            # Check stop-loss
            if current_price <= position.stop_loss_price:
                positions_to_close.append((pair, 'STOP_LOSS'))
            
            # Check take-profit
            elif current_price >= position.take_profit_price:
                positions_to_close.append((pair, 'TAKE_PROFIT'))
            
            # Check time-based exit (if position is old)
            position_age = datetime.now() - position.entry_timestamp
            if position_age > timedelta(hours=4):  # Max 4 hour hold
                positions_to_close.append((pair, 'TIME_EXIT'))
        
        # Execute exit orders
        for pair, reason in positions_to_close:
            await self._execute_exit_order(pair, reason)
    
    async def _execute_exit_order(self, pair: str, reason: str):
        """Execute an exit order for a position."""
        if pair not in self.positions:
            return
        
        position = self.positions[pair]
        current_price = position.current_price
        
        # Create exit opportunity
        exit_opportunity = TradingOpportunity(
            pair=pair,
            signal='SELL',
            entry_price=current_price,
            target_price=current_price,
            stop_loss=current_price,
            expected_profit_pct=0,
            risk_reward_ratio=1,
            confidence=100,
            urgency='HIGH',
            reasoning=f"Exit trigger: {reason}"
        )
        
        # Execute sell order
        trade = await self._execute_sell_order(exit_opportunity)
        
        if trade:
            print(f"EXIT: EXIT {pair}: {reason} @ ${current_price:.2f}")
    
    async def _optimize_strategy_parameters(self):
        """Optimize strategy parameters using AI."""
        print(" Optimizing strategy parameters with AI...")
        
        try:
            # Use AI optimizer to get new parameters
            optimized_params = self.ai_optimizer.optimize_strategy_parameters(self.trading_pairs)
            
            # Update strategy parameters
            self.strategy_params = optimized_params
            
            print(f"SUCCESS: Strategy parameters optimized:")
            print(f"   Buy threshold: {self.strategy_params.buy_threshold:.4f}")
            print(f"   Sell threshold: {self.strategy_params.sell_threshold:.4f}")
            print(f"   Max position: {self.strategy_params.max_position_size:.2f}")
            
        except Exception as e:
            print(f"WARNING: Error optimizing parameters: {e}")
    
    async def _generate_real_time_insights(self, market_analyses: Dict, 
                                         executed_trades: List, market_opening_context: Dict = None) -> List[str]:
        """Generate real-time trading insights."""
        insights = []
        
        # Portfolio performance insight
        current_value = self._calculate_portfolio_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        insights.append(f"Portfolio: ${current_value:.2f} ({total_return:+.3f}%)")
        
        # Active positions insight
        if self.positions:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            insights.append(f"Open positions: {len(self.positions)}, Unrealized P&L: ${total_unrealized:+.2f}")
        
        # Recent trades insight
        if executed_trades:
            insights.append(f"Executed {len(executed_trades)} trades this step")
        
        # Market conditions insight
        if market_analyses:
            avg_volatility = np.mean([
                analysis['coin_analysis'].volatility_24h 
                for analysis in market_analyses.values()
            ])
            insights.append(f"Market volatility: {avg_volatility:.3f}")
        
        # Market opening insights (YOUR HUMAN INTELLIGENCE INTEGRATION)
        if market_opening_context and market_opening_context.get('in_opening_window'):
            session = market_opening_context.get('active_session')
            if market_opening_context.get('time_to_opening'):
                hours_to_opening = market_opening_context['time_to_opening'] / 3600
                insights.append(f"🌍 {session} opens in {hours_to_opening:.1f}h - Pre-opening analysis active")
            elif market_opening_context.get('time_since_opening'):
                hours_since_opening = market_opening_context['time_since_opening'] / 3600
                insights.append(f"👀 {session} opened {hours_since_opening:.1f}h ago - Monitoring rebounds")
            
            if market_opening_context.get('rebound_expected'):
                insights.append("🔄 High rebound probability detected")
            
            if market_opening_context.get('manipulation_risk') == 'HIGH':
                insights.append("⚠️ High manipulation risk - Trading cautiously")
        
        return insights
    
    def _get_available_capital(self) -> float:
        """Get available capital for trading."""
        return max(0, self.current_capital)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.current_capital
        
        for position in self.positions.values():
            position_value = position.base_amount * position.current_price
            total_value += position_value
        
        return total_value
    
    async def _save_step_data(self, step: int, market_analyses: Dict, 
                            opportunities: List, executed_trades: List, insights: List, market_opening_context: Dict = None):
        """Save step data to files."""
        # Portfolio snapshot
        portfolio_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'portfolio_value': self._calculate_portfolio_value(),
            'cash_balance': self.current_capital,
            'total_return': ((self._calculate_portfolio_value() - self.initial_capital) / self.initial_capital) * 100,
            'positions': {
                pair: {
                    'amount': pos.base_amount,
                    'avg_price': pos.avg_buy_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'technical_score': pos.technical_score,
                    'ai_score': pos.ai_score
                }
                for pair, pos in self.positions.items()
            },
            'insights': insights,
            'market_opening_context': market_opening_context  # Save opening intelligence
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        # Save to files
        with open(self.data_dir / "portfolio_history.json", 'w') as f:
            json.dump(self.portfolio_history, f, indent=2)
        
        with open(self.data_dir / "trades.json", 'w') as f:
            trades_data = [asdict(trade) for trade in self.trade_history]
            # Convert datetime objects to strings
            for trade in trades_data:
                trade['timestamp'] = trade['timestamp'].isoformat()
            json.dump(trades_data, f, indent=2)
        
        # Save opportunities
        with open(self.data_dir / "opportunities.json", 'w') as f:
            opportunities_data = [asdict(opp) for opp in opportunities]
            json.dump(opportunities_data, f, indent=2)
    
    def _display_progress(self, step: int, executed_trades: List, insights: List, market_opening_context: Dict = None, end_time: datetime = None):
        """Display current progress with timer."""
        current_value = self._calculate_portfolio_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        # Timer display
        current_time = datetime.now()
        if hasattr(self, 'simulation_start_time') and end_time:
            elapsed = current_time - self.simulation_start_time
            remaining = end_time - current_time
            elapsed_hours = elapsed.total_seconds() / 3600
            remaining_hours = remaining.total_seconds() / 3600
            
            print(f" ⏱️  Timer: {elapsed_hours:.1f}h elapsed | {remaining_hours:.1f}h remaining")
        
        print(f" 💰 Portfolio: ${current_value:.2f} ({total_return:+.3f}%)")
        print(f" 💵 Cash: ${self.current_capital:.2f}")
        print(f" 📊 Positions: {len(self.positions)}")
        print(f" 🔄 Trades: {len(self.trade_history)}")
        
        # Market opening status display (YOUR TIMEZONE-AWARE MONITORING)
        if market_opening_context and market_opening_context.get('in_opening_window'):
            session = market_opening_context.get('active_session')
            print(f" 🌍 Market Window: {session}")
            
            if market_opening_context.get('time_to_opening'):
                hours = market_opening_context['time_to_opening'] / 3600
                print(f" ⏰ Opens in: {hours:.1f}h")
            elif market_opening_context.get('time_since_opening'):
                hours = market_opening_context['time_since_opening'] / 3600
                print(f" ⏰ Opened: {hours:.1f}h ago")
        
        if insights:
            print(f" Insights: {', '.join(insights[:2])}")
    
    async def _generate_final_analysis(self):
        """Generate comprehensive final analysis."""
        print(f"\n Generating Final Analysis...")
        
        # Use NLP analyzer for comprehensive analysis
        analysis = self.nlp_analyzer.analyze_trading_session(str(self.data_dir))
        
        # Save analysis
        with open(self.data_dir / "final_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Display key results
        if 'error' not in analysis:
            print(f"\n FINAL RESULTS")
            print("=" * 50)
            print(analysis['session_summary'])
            
            perf = analysis['performance_analysis']
            print(f"\n Performance:")
            print(f"   Total Return: {perf['total_return']:.3f}%")
            print(f"   Win Rate: {perf['win_rate']:.1f}%")
            print(f"   Max Drawdown: {perf['max_drawdown']:.2f}%")
            
            print(f"\n Key Insights:")
            for insight in analysis['key_insights'][:3]:
                print(f"   • {insight}")
            
            print(f"\n Recommendations:")
            for rec in analysis['strategic_recommendations'][:3]:
                print(f"   • {rec}")
    
    async def _check_market_opening_windows(self, current_time: datetime) -> Dict:
        """Check if we're in a market opening observation window (2h before + 2h after)"""
        
        # Convert current time to UTC for comparison
        current_utc = current_time.replace(tzinfo=pytz.UTC) if current_time.tzinfo is None else current_time.astimezone(pytz.UTC)
        
        market_context = {
            'in_opening_window': False,
            'active_session': None,
            'time_to_opening': None,
            'time_since_opening': None,
            'opening_analysis': None,
            'rebound_expected': False,
            'manipulation_risk': 'LOW',
            'strategy_adjustments': {}
        }
        
        # Check each market session
        for session_name, session in self.market_opening_ai.sessions.items():
            opening_time = self._get_next_opening_time(session, current_utc)
            time_until_opening = (opening_time - current_utc).total_seconds()
            
            # Check if we're in the 4-hour observation window (2h before + 2h after)
            observation_window_seconds = self.market_observation_window * 3600  # 4 hours = 14400 seconds
            pre_opening_seconds = 2 * 3600  # 2 hours before
            
            if -observation_window_seconds/2 <= time_until_opening <= observation_window_seconds/2:
                market_context['in_opening_window'] = True
                market_context['active_session'] = session_name
                
                if time_until_opening > 0:
                    market_context['time_to_opening'] = time_until_opening
                    
                    # Perform pre-opening analysis if within 2 hours
                    if time_until_opening <= pre_opening_seconds:
                        if (current_time - self.last_opening_check).total_seconds() > 300:  # Every 5 minutes
                            analysis = self.market_opening_ai.perform_opening_analysis(session_name)
                            market_context['opening_analysis'] = analysis
                            market_context['rebound_expected'] = analysis.rebound_probability > 0.8
                            market_context['manipulation_risk'] = analysis.manipulation_risk
                            
                            # Your human insights: Adjust strategy for opening
                            market_context['strategy_adjustments'] = self._get_opening_strategy_adjustments(analysis)
                            
                            self.last_opening_check = current_time
                            
                            print(f"\n🌍 MARKET OPENING ALERT: {session_name}")
                            print(f"⏰ Opening in {time_until_opening/3600:.1f} hours")
                            print(f"📊 Prediction: {analysis.opening_prediction}")
                            print(f"🔄 Rebound Expected: {analysis.rebound_probability:.1%}")
                            print(f"⚠️  Manipulation Risk: {analysis.manipulation_risk}")
                else:
                    market_context['time_since_opening'] = abs(time_until_opening)
                    
                    # We're in post-opening window - monitor for rebounds
                    if abs(time_until_opening) <= 2 * 3600:  # Within 2 hours after opening
                        print(f"\n👀 POST-OPENING MONITORING: {session_name}")
                        print(f"⏰ {abs(time_until_opening)/3600:.1f} hours since opening")
                        print("🔄 Monitoring for rebound patterns...")
                
                break  # Only track one active session at a time
        
        return market_context
    
    def _get_next_opening_time(self, session, current_utc):
        """Get next opening time for a session"""
        tz = pytz.timezone(session.timezone)
        
        from datetime import time
        
        # Today's opening time
        today = current_utc.astimezone(tz).date()
        opening_time = datetime.combine(
            today, 
            time.fromisoformat(session.open_time)
        )
        opening_time = tz.localize(opening_time).astimezone(pytz.UTC)
        
        # If already passed, get tomorrow's
        if opening_time <= current_utc:
            tomorrow = today + timedelta(days=1)
            opening_time = datetime.combine(
                tomorrow,
                time.fromisoformat(session.open_time)
            )
            opening_time = tz.localize(opening_time).astimezone(pytz.UTC)
        
        return opening_time
    
    def _get_opening_strategy_adjustments(self, analysis) -> Dict:
        """Get strategy adjustments based on opening analysis (YOUR HUMAN INSIGHTS)"""
        
        adjustments = {
            'wait_for_rebound': False,
            'avoid_trades': False,
            'increase_confidence_threshold': False,
            'reduce_position_size': False,
            'enable_scalping': False
        }
        
        # Your insight: Always expect rebound
        if analysis.rebound_probability > 0.8:
            adjustments['wait_for_rebound'] = True
            adjustments['enable_scalping'] = True  # Quick trades on rebounds
        
        # Your insight: Avoid manipulation
        if analysis.manipulation_risk == 'HIGH':
            adjustments['avoid_trades'] = True
            print("🚨 HIGH MANIPULATION RISK - Avoiding trades until genuine move")
        
        # Low confidence = be more careful
        if analysis.confidence < 0.6:
            adjustments['increase_confidence_threshold'] = True
            adjustments['reduce_position_size'] = True
        
        return adjustments

async def main():
    """Main function to run enhanced simulation."""
    parser = argparse.ArgumentParser(description='Enhanced KrakenBot Simulation')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital (default: 1000)')
    parser.add_argument('--duration', type=float, default=4.0, help='Duration in hours (default: 4)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--session-name', type=str, help='Custom session name')
    
    args = parser.parse_args()
    
    # Create and run simulation
    simulation = EnhancedSimulationSystem(
        initial_capital=args.capital,
        session_name=args.session_name
    )
    
    await simulation.run_simulation(
        duration_hours=args.duration,
        check_interval=args.interval
    )

if __name__ == "__main__":
    asyncio.run(main())