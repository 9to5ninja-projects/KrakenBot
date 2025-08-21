"""
Enhanced Simple Pair Trader with Advanced Risk Management
Includes trailing stops, take profits, position timing, and dynamic thresholds
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

import config
from exchange import KrakenExchange

@dataclass
class Position:
    """Represents an open trading position."""
    pair: str
    entry_price: float
    entry_time: datetime
    amount: float
    side: str  # 'buy' or 'sell'
    stop_loss: float
    take_profit: float
    trailing_stop: float
    max_profit: float = 0.0
    
class EnhancedSimplePairTrader:
    """Enhanced simple pair trader with advanced risk management."""
    
    def __init__(self, initial_balance: float = 100.0):
        """Initialize the enhanced trader."""
        self.exchange = KrakenExchange()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Trading pairs
        self.pairs = [
            config.TRADING_PAIRS.get('PAIR_2', 'ETH/CAD'),  # Primary pair
            config.TRADING_PAIRS.get('PAIR_1', 'BTC/CAD'),  # Secondary pair
        ]
        
        # Enhanced trading parameters
        self.min_trade_amount = getattr(config, 'MIN_TRADE_AMOUNT', 25.0)
        self.max_position_size = getattr(config, 'MAX_POSITION_SIZE', 0.2)
        self.buy_threshold = getattr(config, 'BUY_THRESHOLD', -0.003)
        self.sell_threshold = getattr(config, 'SELL_THRESHOLD', 0.005)
        self.lookback_periods = getattr(config, 'LOOKBACK_PERIODS', 5)
        
        # Risk management parameters
        self.stop_loss_pct = getattr(config, 'STOP_LOSS_PCT', 0.002)  # 0.2%
        self.take_profit_pct = getattr(config, 'TAKE_PROFIT_PCT', 0.006)  # 0.6%
        self.max_hold_time = getattr(config, 'MAX_HOLD_TIME', 1800)  # 30 minutes
        self.trailing_stop_enabled = getattr(config, 'TRAILING_STOP', True)
        self.dynamic_thresholds = getattr(config, 'DYNAMIC_THRESHOLDS', True)
        
        # Fee structure
        self.maker_fee = config.MAKER_FEE
        self.taker_fee = config.TAKER_FEE
        
        # Position tracking
        self.open_positions: Dict[str, Position] = {}
        self.price_history = {pair: [] for pair in self.pairs}
        self.trades = []
        self.portfolio_history = []
        
        # Volatility tracking for dynamic thresholds
        self.volatility_history = {pair: [] for pair in self.pairs}
        
        logger.info(f"Enhanced simple pair trader initialized:")
        logger.info(f"  Pairs: {', '.join(self.pairs)}")
        logger.info(f"  Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"  Buy threshold: {self.buy_threshold*100:.2f}%")
        logger.info(f"  Sell threshold: {self.sell_threshold*100:.2f}%")
        logger.info(f"  Stop loss: {self.stop_loss_pct*100:.2f}%")
        logger.info(f"  Take profit: {self.take_profit_pct*100:.2f}%")
        logger.info(f"  Max hold time: {self.max_hold_time}s")
        logger.info(f"  Trailing stop: {self.trailing_stop_enabled}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all pairs."""
        prices = {}
        for pair in self.pairs:
            try:
                ticker = self.exchange.fetch_ticker(pair)
                prices[pair] = ticker['last']
            except Exception as e:
                logger.error(f"Error fetching price for {pair}: {e}")
                prices[pair] = 0.0
        return prices
    
    def calculate_volatility(self, pair: str, periods: int = 20) -> float:
        """Calculate recent volatility for dynamic threshold adjustment."""
        if len(self.price_history[pair]) < periods:
            return 0.02  # Default volatility
        
        recent_prices = [p['price'] for p in self.price_history[pair][-periods:]]
        if len(recent_prices) < 2:
            return 0.02
        
        # Calculate returns
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)
        
        # Standard deviation of returns
        if len(returns) < 2:
            return 0.02
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return max(0.005, min(0.05, volatility))  # Clamp between 0.5% and 5%
    
    def adjust_thresholds_for_volatility(self, pair: str) -> Tuple[float, float]:
        """Dynamically adjust thresholds based on current volatility."""
        if not self.dynamic_thresholds:
            return self.buy_threshold, self.sell_threshold
        
        volatility = self.calculate_volatility(pair)
        base_vol = 0.02  # Base volatility assumption
        
        # Adjust thresholds based on volatility
        vol_multiplier = volatility / base_vol
        
        # More volatile = wider thresholds, less volatile = tighter thresholds
        adjusted_buy = self.buy_threshold * vol_multiplier
        adjusted_sell = self.sell_threshold * vol_multiplier
        
        # Ensure minimum aggressiveness
        adjusted_buy = max(adjusted_buy, -0.001)  # At least 0.1% drop
        adjusted_sell = min(adjusted_sell, 0.008)  # At most 0.8% gain
        
        return adjusted_buy, adjusted_sell
    
    def should_buy(self, pair: str, current_price: float) -> bool:
        """Determine if we should buy based on price action."""
        if pair in self.open_positions:
            return False  # Already have position
        
        if len(self.price_history[pair]) < self.lookback_periods:
            return False
        
        # Get recent high
        recent_prices = [p['price'] for p in self.price_history[pair][-self.lookback_periods:]]
        recent_high = max(recent_prices)
        
        # Calculate price change from recent high
        price_change = (current_price - recent_high) / recent_high
        
        # Get dynamic thresholds
        buy_threshold, _ = self.adjust_thresholds_for_volatility(pair)
        
        # Buy if price dropped enough from recent high
        should_buy = price_change <= buy_threshold
        
        if should_buy:
            logger.info(f"Buy signal for {pair}: {price_change*100:.3f}% drop (threshold: {buy_threshold*100:.3f}%)")
        
        return should_buy
    
    def should_sell(self, pair: str, current_price: float) -> bool:
        """Determine if we should sell based on price action."""
        if pair not in self.open_positions:
            return False  # No position to sell
        
        position = self.open_positions[pair]
        
        # Calculate current profit/loss
        price_change = (current_price - position.entry_price) / position.entry_price
        
        # Check stop loss
        if current_price <= position.stop_loss:
            logger.info(f"Stop loss triggered for {pair}: {price_change*100:.3f}%")
            return True
        
        # Check take profit
        if current_price >= position.take_profit:
            logger.info(f"Take profit triggered for {pair}: {price_change*100:.3f}%")
            return True
        
        # Check max hold time
        hold_time = (datetime.now() - position.entry_time).total_seconds()
        if hold_time >= self.max_hold_time:
            logger.info(f"Max hold time reached for {pair}: {hold_time:.0f}s")
            return True
        
        # Update trailing stop
        if self.trailing_stop_enabled and price_change > 0:
            # Update max profit seen
            position.max_profit = max(position.max_profit, price_change)
            
            # Calculate trailing stop (0.2% below max profit)
            trailing_stop_price = position.entry_price * (1 + position.max_profit - self.stop_loss_pct)
            
            # Update stop loss if trailing stop is higher
            if trailing_stop_price > position.stop_loss:
                position.stop_loss = trailing_stop_price
                logger.debug(f"Updated trailing stop for {pair}: ${trailing_stop_price:.2f}")
        
        # Check regular sell threshold
        if len(self.price_history[pair]) >= self.lookback_periods:
            recent_prices = [p['price'] for p in self.price_history[pair][-self.lookback_periods:]]
            recent_low = min(recent_prices)
            change_from_low = (current_price - recent_low) / recent_low
            
            _, sell_threshold = self.adjust_thresholds_for_volatility(pair)
            
            if change_from_low >= sell_threshold:
                logger.info(f"Sell signal for {pair}: {change_from_low*100:.3f}% gain (threshold: {sell_threshold*100:.3f}%)")
                return True
        
        return False
    
    def execute_buy(self, pair: str, current_price: float) -> bool:
        """Execute a buy order."""
        try:
            # Calculate position size
            available_balance = self.current_balance
            max_position_value = available_balance * self.max_position_size
            position_value = min(max_position_value, available_balance - self.min_trade_amount)
            
            if position_value < self.min_trade_amount:
                logger.warning(f"Insufficient balance for {pair} buy: ${available_balance:.2f}")
                return False
            
            # Calculate amount to buy (accounting for fees)
            effective_price = current_price * (1 + self.taker_fee)
            amount = position_value / effective_price
            
            # Create position
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            take_profit_price = current_price * (1 + self.take_profit_pct)
            
            position = Position(
                pair=pair,
                entry_price=current_price,
                entry_time=datetime.now(),
                amount=amount,
                side='buy',
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                trailing_stop=stop_loss_price
            )
            
            self.open_positions[pair] = position
            
            # Update balance
            total_cost = position_value + (position_value * self.taker_fee)
            self.current_balance -= total_cost
            
            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'action': 'buy',
                'price': current_price,
                'amount': amount,
                'value': position_value,
                'fee': position_value * self.taker_fee,
                'balance_after': self.current_balance,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
            self.trades.append(trade)
            
            logger.info(f"✅ BUY {pair}: {amount:.6f} @ ${current_price:.2f} (${position_value:.2f})")
            logger.info(f"   Stop Loss: ${stop_loss_price:.2f} | Take Profit: ${take_profit_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing buy for {pair}: {e}")
            return False
    
    def execute_sell(self, pair: str, current_price: float) -> bool:
        """Execute a sell order."""
        try:
            if pair not in self.open_positions:
                return False
            
            position = self.open_positions[pair]
            
            # Calculate proceeds (accounting for fees)
            gross_proceeds = position.amount * current_price
            net_proceeds = gross_proceeds * (1 - self.taker_fee)
            
            # Calculate profit/loss
            original_cost = position.amount * position.entry_price
            profit_loss = net_proceeds - original_cost
            profit_pct = (profit_loss / original_cost) * 100
            
            # Update balance
            self.current_balance += net_proceeds
            
            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'action': 'sell',
                'price': current_price,
                'amount': position.amount,
                'value': gross_proceeds,
                'fee': gross_proceeds * self.taker_fee,
                'profit': profit_loss,
                'profit_pct': profit_pct,
                'hold_time': (datetime.now() - position.entry_time).total_seconds(),
                'balance_after': self.current_balance
            }
            
            self.trades.append(trade)
            
            # Remove position
            del self.open_positions[pair]
            
            profit_emoji = "📈" if profit_loss > 0 else "📉"
            logger.info(f"✅ SELL {pair}: {position.amount:.6f} @ ${current_price:.2f}")
            logger.info(f"   {profit_emoji} Profit: ${profit_loss:.2f} ({profit_pct:+.3f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing sell for {pair}: {e}")
            return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value including open positions."""
        total_value = self.current_balance
        
        # Add value of open positions
        for pair, position in self.open_positions.items():
            if pair in current_prices:
                position_value = position.amount * current_prices[pair]
                total_value += position_value
        
        # Calculate total return
        total_return = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        # Record portfolio snapshot
        portfolio_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': total_value,
            'cash_balance': self.current_balance,
            'total_return': total_return,
            'open_positions': len(self.open_positions),
            'trades_executed': len(self.trades)
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        return total_value, total_return
    
    def process_market_update(self):
        """Process a single market update cycle."""
        try:
            # Get current prices
            current_prices = self.get_current_prices()
            
            if not any(current_prices.values()):
                logger.warning("No valid prices received")
                return
            
            # Update price history
            timestamp = datetime.now()
            for pair, price in current_prices.items():
                if price > 0:
                    self.price_history[pair].append({
                        'timestamp': timestamp.isoformat(),
                        'price': price
                    })
                    
                    # Keep only recent history
                    if len(self.price_history[pair]) > 100:
                        self.price_history[pair] = self.price_history[pair][-100:]
            
            # Process trading logic for each pair
            for pair in self.pairs:
                if pair not in current_prices or current_prices[pair] <= 0:
                    continue
                
                current_price = current_prices[pair]
                
                # Check for sell signals first (manage existing positions)
                if self.should_sell(pair, current_price):
                    self.execute_sell(pair, current_price)
                
                # Check for buy signals (only if no position)
                elif self.should_buy(pair, current_price):
                    self.execute_buy(pair, current_price)
            
            # Update portfolio value
            portfolio_value, total_return = self.update_portfolio_value(current_prices)
            
            # Log status periodically
            if len(self.portfolio_history) % 10 == 0:  # Every 10 updates
                logger.info(f"Portfolio: ${portfolio_value:.2f} ({total_return:+.4f}%) | "
                          f"Positions: {len(self.open_positions)} | Trades: {len(self.trades)}")
            
        except Exception as e:
            logger.error(f"Error in market update: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.portfolio_history:
            return {'status': 'no_data'}
        
        latest = self.portfolio_history[-1]
        
        # Calculate trade statistics
        profitable_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit', 0) < 0]
        
        total_profit = sum(t.get('profit', 0) for t in self.trades)
        avg_profit_per_trade = total_profit / len(self.trades) if self.trades else 0
        
        win_rate = len(profitable_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Calculate max drawdown
        values = [p['portfolio_value'] for p in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'final_portfolio_value': latest['portfolio_value'],
            'total_return_pct': latest['total_return'],
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown_pct': max_drawdown,
            'open_positions': len(self.open_positions),
            'data_points': len(self.portfolio_history)
        }