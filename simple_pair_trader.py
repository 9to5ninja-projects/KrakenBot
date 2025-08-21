"""
Simple pair trading strategy for CAD/ETH and CAD/BTC.
Implements buy-low, sell-high strategy with technical indicators.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

import config
from exchange import ExchangeManager

@dataclass
class Trade:
    """Represents a trade execution."""
    timestamp: datetime
    pair: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    cost: float
    fee: float
    portfolio_value_before: float
    portfolio_value_after: float
    reason: str

@dataclass
class Position:
    """Represents a position in a currency pair."""
    pair: str
    base_amount: float  # Amount of base currency (ETH/BTC)
    quote_amount: float  # Amount of quote currency (CAD)
    avg_buy_price: float
    total_invested: float
    unrealized_pnl: float
    realized_pnl: float

class SimplePairTrader:
    """Simple pair trading strategy implementation."""
    
    def __init__(self, pairs: List[str] = None, initial_balance: float = None):
        """
        Initialize the simple pair trader.
        
        Args:
            pairs: List of trading pairs (default: ['ETH/CAD', 'BTC/CAD'])
            initial_balance: Initial CAD balance (default: from config)
        """
        self.exchange = ExchangeManager(use_api_keys=False)
        self.pairs = pairs or ['ETH/CAD', 'BTC/CAD']
        self.initial_balance = initial_balance or config.START_AMOUNT
        
        # Portfolio tracking
        self.cad_balance = self.initial_balance
        self.positions = {pair: Position(
            pair=pair,
            base_amount=0.0,
            quote_amount=0.0,
            avg_buy_price=0.0,
            total_invested=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        ) for pair in self.pairs}
        
        # Trading parameters (from config)
        self.min_trade_amount = config.MIN_TRADE_AMOUNT
        self.max_position_size = config.MAX_POSITION_SIZE
        self.buy_threshold = config.BUY_THRESHOLD
        self.sell_threshold = config.SELL_THRESHOLD
        self.lookback_periods = config.LOOKBACK_PERIODS
        
        # Fee structure
        self.maker_fee = config.MAKER_FEE
        self.taker_fee = config.TAKER_FEE
        
        # Data storage
        self.price_history = {pair: [] for pair in self.pairs}
        self.trades = []
        self.portfolio_history = []
        
        logger.info(f"Simple pair trader initialized:")
        logger.info(f"  Pairs: {', '.join(self.pairs)}")
        logger.info(f"  Initial balance: ${self.initial_balance:.2f} CAD")
        logger.info(f"  Min trade amount: ${self.min_trade_amount:.2f}")
        logger.info(f"  Max position size: {self.max_position_size*100:.0f}%")
        logger.info(f"  Buy threshold: {self.buy_threshold*100:.1f}%")
        logger.info(f"  Sell threshold: {self.sell_threshold*100:.1f}%")
    
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
    
    def update_price_history(self, prices: Dict[str, float]):
        """Update price history for technical analysis."""
        timestamp = datetime.now()
        
        for pair, price in prices.items():
            if price > 0:
                self.price_history[pair].append({
                    'timestamp': timestamp,
                    'price': price
                })
                
                # Keep only recent history
                if len(self.price_history[pair]) > 100:
                    self.price_history[pair] = self.price_history[pair][-100:]
    
    def calculate_signals(self, pair: str) -> Tuple[bool, bool, str]:
        """
        Calculate buy/sell signals for a pair.
        
        Returns:
            Tuple of (should_buy, should_sell, reason)
        """
        if len(self.price_history[pair]) < self.lookback_periods:
            return False, False, "Insufficient price history"
        
        recent_prices = [p['price'] for p in self.price_history[pair][-self.lookback_periods:]]
        current_price = recent_prices[-1]
        
        # Calculate recent high and low
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        
        # Calculate price change from high/low
        change_from_high = (current_price - recent_high) / recent_high
        change_from_low = (current_price - recent_low) / recent_low
        
        # Simple moving average
        sma = np.mean(recent_prices)
        price_vs_sma = (current_price - sma) / sma
        
        # Buy signal: price dropped significantly from recent high and below SMA
        should_buy = (change_from_high <= self.buy_threshold and 
                     current_price < sma and 
                     self.positions[pair].base_amount == 0)  # No existing position
        
        # Sell signal: price rose significantly from recent low or above SMA with profit
        should_sell = (self.positions[pair].base_amount > 0 and 
                      (change_from_low >= self.sell_threshold or 
                       (current_price > sma and current_price > self.positions[pair].avg_buy_price * 1.01)))
        
        reason = f"Price: ${current_price:.2f}, High: ${recent_high:.2f} ({change_from_high*100:.1f}%), Low: ${recent_low:.2f} ({change_from_low*100:.1f}%), SMA: ${sma:.2f}"
        
        return should_buy, should_sell, reason
    
    def calculate_trade_amount(self, pair: str, side: str, price: float) -> float:
        """Calculate the amount to trade."""
        if side == 'buy':
            # Use a portion of available CAD balance
            max_cad_amount = self.cad_balance * self.max_position_size
            trade_cad_amount = min(max_cad_amount, self.cad_balance * 0.1)  # 10% of balance per trade
            trade_cad_amount = max(trade_cad_amount, self.min_trade_amount)
            
            if trade_cad_amount > self.cad_balance:
                trade_cad_amount = self.cad_balance * 0.95  # Leave some buffer
            
            # Calculate base amount (ETH or BTC)
            base_amount = trade_cad_amount / price
            return base_amount
        
        else:  # sell
            # Sell all of the position
            return self.positions[pair].base_amount
    
    def execute_trade(self, pair: str, side: str, amount: float, price: float, reason: str) -> bool:
        """
        Execute a simulated trade.
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            amount: Amount of base currency
            price: Price per unit
            reason: Reason for the trade
            
        Returns:
            True if trade was executed successfully
        """
        try:
            # Calculate costs
            cost = amount * price
            fee = cost * self.taker_fee  # Assume taker fee for simplicity
            
            portfolio_value_before = self.get_portfolio_value()
            
            if side == 'buy':
                if cost + fee > self.cad_balance:
                    logger.warning(f"Insufficient CAD balance for {pair} buy: need ${cost + fee:.2f}, have ${self.cad_balance:.2f}")
                    return False
                
                # Update balances
                self.cad_balance -= (cost + fee)
                
                # Update position
                position = self.positions[pair]
                if position.base_amount == 0:
                    position.avg_buy_price = price
                else:
                    # Update average buy price
                    total_cost = position.total_invested + cost
                    total_amount = position.base_amount + amount
                    position.avg_buy_price = total_cost / total_amount
                
                position.base_amount += amount
                position.total_invested += cost
                
                logger.info(f"BUY {amount:.6f} {pair.split('/')[0]} at ${price:.2f} (Cost: ${cost:.2f}, Fee: ${fee:.2f})")
                
            else:  # sell
                if amount > self.positions[pair].base_amount:
                    logger.warning(f"Insufficient {pair.split('/')[0]} balance for sell")
                    return False
                
                # Calculate proceeds
                proceeds = cost - fee
                
                # Update balances
                self.cad_balance += proceeds
                
                # Update position
                position = self.positions[pair]
                position.base_amount -= amount
                
                # Calculate realized P&L
                cost_basis = amount * position.avg_buy_price
                realized_pnl = proceeds - cost_basis
                position.realized_pnl += realized_pnl
                
                if position.base_amount == 0:
                    position.total_invested = 0
                    position.avg_buy_price = 0
                else:
                    position.total_invested -= cost_basis
                
                logger.info(f"SELL {amount:.6f} {pair.split('/')[0]} at ${price:.2f} (Proceeds: ${proceeds:.2f}, Fee: ${fee:.2f}, P&L: ${realized_pnl:.2f})")
            
            # Record trade
            portfolio_value_after = self.get_portfolio_value()
            trade = Trade(
                timestamp=datetime.now(),
                pair=pair,
                side=side,
                amount=amount,
                price=price,
                cost=cost,
                fee=fee,
                portfolio_value_before=portfolio_value_before,
                portfolio_value_after=portfolio_value_after,
                reason=reason
            )
            self.trades.append(trade)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing {side} trade for {pair}: {e}")
            return False
    
    def get_portfolio_value(self, prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio value in CAD."""
        if prices is None:
            prices = self.get_current_prices()
        
        total_value = self.cad_balance
        
        for pair, position in self.positions.items():
            if position.base_amount > 0 and pair in prices:
                total_value += position.base_amount * prices[pair]
        
        return total_value
    
    def update_unrealized_pnl(self, prices: Dict[str, float]):
        """Update unrealized P&L for all positions."""
        for pair, position in self.positions.items():
            if position.base_amount > 0 and pair in prices:
                current_value = position.base_amount * prices[pair]
                position.unrealized_pnl = current_value - position.total_invested
    
    def run_strategy_step(self) -> Dict:
        """Run one step of the trading strategy."""
        try:
            # Get current prices
            prices = self.get_current_prices()
            
            # Update price history
            self.update_price_history(prices)
            
            # Update unrealized P&L
            self.update_unrealized_pnl(prices)
            
            # Calculate portfolio value
            portfolio_value = self.get_portfolio_value(prices)
            
            # Record portfolio snapshot
            portfolio_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'cad_balance': self.cad_balance,
                'portfolio_value': portfolio_value,
                'total_return': (portfolio_value - self.initial_balance) / self.initial_balance * 100,
                'positions': {pair: {
                    'base_amount': pos.base_amount,
                    'avg_buy_price': pos.avg_buy_price,
                    'current_price': prices.get(pair, 0),
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                } for pair, pos in self.positions.items()},
                'prices': prices
            }
            self.portfolio_history.append(portfolio_snapshot)
            
            # Check for trading signals
            trades_executed = 0
            for pair in self.pairs:
                if pair not in prices or prices[pair] == 0:
                    continue
                
                should_buy, should_sell, reason = self.calculate_signals(pair)
                
                if should_buy:
                    amount = self.calculate_trade_amount(pair, 'buy', prices[pair])
                    if amount > 0:
                        if self.execute_trade(pair, 'buy', amount, prices[pair], reason):
                            trades_executed += 1
                
                elif should_sell:
                    amount = self.calculate_trade_amount(pair, 'sell', prices[pair])
                    if amount > 0:
                        if self.execute_trade(pair, 'sell', amount, prices[pair], reason):
                            trades_executed += 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'cad_balance': self.cad_balance,
                'trades_executed': trades_executed,
                'prices': prices,
                'positions': {pair: pos.base_amount for pair, pos in self.positions.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in strategy step: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.portfolio_history:
            return {}
        
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_balance) / self.initial_balance * 100
        
        # Calculate total fees paid
        total_fees = sum(trade.fee for trade in self.trades)
        
        # Calculate total realized P&L
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'initial_balance': self.initial_balance,
            'current_value': current_value,
            'total_return_pct': total_return,
            'total_return_cad': current_value - self.initial_balance,
            'total_trades': len(self.trades),
            'total_fees_paid': total_fees,
            'total_realized_pnl': total_realized_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'cad_balance': self.cad_balance,
            'positions': {pair: {
                'amount': pos.base_amount,
                'avg_buy_price': pos.avg_buy_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            } for pair, pos in self.positions.items()}
        }
    
    def save_data(self, base_dir: str):
        """Save all trading data."""
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        if self.trades:
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    'timestamp': trade.timestamp.isoformat(),
                    'pair': trade.pair,
                    'side': trade.side,
                    'amount': trade.amount,
                    'price': trade.price,
                    'cost': trade.cost,
                    'fee': trade.fee,
                    'portfolio_value_before': trade.portfolio_value_before,
                    'portfolio_value_after': trade.portfolio_value_after,
                    'reason': trade.reason
                })
            
            with open(base_path / 'trades.json', 'w') as f:
                json.dump(trades_data, f, indent=2)
        
        # Save portfolio history
        if self.portfolio_history:
            with open(base_path / 'portfolio_history.json', 'w') as f:
                json.dump(self.portfolio_history, f, indent=2)
        
        # Save performance summary
        summary = self.get_performance_summary()
        with open(base_path / 'performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save price history
        with open(base_path / 'price_history.json', 'w') as f:
            json.dump(self.price_history, f, indent=2, default=str)
        
        logger.info(f"Trading data saved to {base_path}")