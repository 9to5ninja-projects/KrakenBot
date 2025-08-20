"""
Arbitrage calculation module for the KrakenBot triangular arbitrage system.
Handles profit calculation, fee adjustment, and opportunity detection.
"""
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

import config
from exchange import ExchangeManager

@dataclass
class ArbitrageOpportunity:
    """Data class to store arbitrage opportunity information."""
    timestamp: datetime
    start_amount: float
    end_amount: float
    profit: float
    profit_percentage: float
    prices: Dict[str, float]
    path: List[str]
    is_profitable: bool
    
    def __str__(self) -> str:
        """String representation of the arbitrage opportunity."""
        status = "✅ PROFITABLE" if self.is_profitable else "❌ Not Profitable"
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {status}\n"
            f"Path: {' → '.join(self.path)}\n"
            f"Start: ${self.start_amount:.2f} | End: ${self.end_amount:.2f}\n"
            f"Profit: {self.profit_percentage:.2f}% (${self.profit:.2f})\n"
            f"Prices: {', '.join([f'{k} = {v}' for k, v in self.prices.items()])}"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'start_amount': self.start_amount,
            'end_amount': self.end_amount,
            'profit': self.profit,
            'profit_percentage': self.profit_percentage,
            'prices': self.prices,
            'path': self.path,
            'is_profitable': self.is_profitable
        }


class ArbitrageCalculator:
    """Calculates triangular arbitrage opportunities."""
    
    def __init__(self, exchange_manager: ExchangeManager):
        """
        Initialize the arbitrage calculator.
        
        Args:
            exchange_manager: Exchange manager instance
        """
        self.exchange = exchange_manager
        self.trading_pairs = list(config.TRADING_PAIRS.values())
        self.start_amount = config.START_AMOUNT
        self.profit_threshold = config.PROFIT_THRESHOLD  # Legacy absolute profit threshold
        self.min_profit_pct = config.MIN_PROFIT_PCT  # New percentage-based threshold
        self.maker_fee = config.MAKER_FEE
        self.taker_fee = config.TAKER_FEE
        self.slippage_buffer = config.SLIPPAGE_BUFFER
        self.breakeven_pct = config.BREAKEVEN_PCT
        
        # Fee adjustment factor (includes both fee and slippage)
        self.fee_factor = 1 - self.taker_fee - self.slippage_buffer
        
        logger.info(f"Arbitrage calculator initialized with:")
        logger.info(f"  Start amount: ${self.start_amount:.2f}")
        logger.info(f"  Min profit percentage: {self.min_profit_pct:.2f}%")
        logger.info(f"  Breakeven threshold: ~{self.breakeven_pct:.2f}%")
        logger.info(f"  Fee factor per trade: {self.fee_factor:.6f}")
        logger.info(f"  Trading pairs: {', '.join(self.trading_pairs)}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all trading pairs.
        
        Returns:
            Dictionary of prices keyed by symbol
        """
        return self.exchange.get_last_prices(self.trading_pairs)
    
    def calculate_cad_btc_eth_cad(self, prices: Dict[str, float]) -> ArbitrageOpportunity:
        """
        Calculate CAD → BTC → ETH → CAD arbitrage path.
        
        Args:
            prices: Dictionary of current prices
            
        Returns:
            ArbitrageOpportunity object with calculation results
        """
        # Extract prices
        btc_cad_price = prices['BTC/CAD']
        eth_btc_price = prices['ETH/BTC']
        eth_cad_price = prices['ETH/CAD']
        
        # Calculate conversions with fees
        cad_to_btc = self.start_amount / btc_cad_price * self.fee_factor
        btc_to_eth = cad_to_btc * eth_btc_price * self.fee_factor
        eth_to_cad = btc_to_eth * eth_cad_price * self.fee_factor
        
        # Calculate profit
        profit = eth_to_cad - self.start_amount
        profit_percentage = (profit / self.start_amount) * 100
        
        # Determine if profitable based on percentage threshold
        is_profitable = profit_percentage > self.min_profit_pct
        
        # Create opportunity object
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.now(),
            start_amount=self.start_amount,
            end_amount=eth_to_cad,
            profit=profit,
            profit_percentage=profit_percentage,
            prices=prices,
            path=['CAD', 'BTC', 'ETH', 'CAD'],
            is_profitable=is_profitable
        )
        
        return opportunity
    
    def calculate_cad_eth_btc_cad(self, prices: Dict[str, float]) -> ArbitrageOpportunity:
        """
        Calculate CAD → ETH → BTC → CAD arbitrage path.
        
        Args:
            prices: Dictionary of current prices
            
        Returns:
            ArbitrageOpportunity object with calculation results
        """
        # Extract prices
        eth_cad_price = prices['ETH/CAD']
        eth_btc_price = prices['ETH/BTC']
        btc_cad_price = prices['BTC/CAD']
        
        # Calculate conversions with fees
        cad_to_eth = self.start_amount / eth_cad_price * self.fee_factor
        eth_to_btc = cad_to_eth * eth_btc_price * self.fee_factor
        btc_to_cad = eth_to_btc * btc_cad_price * self.fee_factor
        
        # Calculate profit
        profit = btc_to_cad - self.start_amount
        profit_percentage = (profit / self.start_amount) * 100
        
        # Determine if profitable based on percentage threshold
        is_profitable = profit_percentage > self.min_profit_pct
        
        # Create opportunity object
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.now(),
            start_amount=self.start_amount,
            end_amount=btc_to_cad,
            profit=profit,
            profit_percentage=profit_percentage,
            prices=prices,
            path=['CAD', 'ETH', 'BTC', 'CAD'],
            is_profitable=is_profitable
        )
        
        return opportunity
    
    def calculate_custom_path(self, path: List[str], prices: Dict[str, float]) -> ArbitrageOpportunity:
        """
        Calculate arbitrage for a custom path.
        
        Args:
            path: List of currencies in the path
            prices: Dictionary of current prices
            
        Returns:
            ArbitrageOpportunity object with calculation results
        """
        # Create the trading pairs from the path
        pairs = []
        for i in range(len(path) - 1):
            # Determine the correct symbol format (base/quote)
            base, quote = path[i], path[i+1]
            
            # Check if the pair exists in both directions
            if f"{base}/{quote}" in prices:
                pairs.append((f"{base}/{quote}", True))  # True means direct
            elif f"{quote}/{base}" in prices:
                pairs.append((f"{quote}/{base}", False))  # False means inverse
            else:
                # If the pair doesn't exist, return a non-profitable opportunity
                logger.warning(f"Trading pair for {base}-{quote} not found in prices")
                return ArbitrageOpportunity(
                    timestamp=datetime.now(),
                    start_amount=self.start_amount,
                    end_amount=self.start_amount,
                    profit=0,
                    profit_percentage=0,
                    prices=prices,
                    path=path,
                    is_profitable=False
                )
        
        # Calculate the conversion
        amount = self.start_amount
        
        for pair, is_direct in pairs:
            price = prices[pair]
            
            if is_direct:
                # Selling base for quote
                amount = amount * price * self.fee_factor
            else:
                # Buying base with quote
                amount = amount / price * self.fee_factor
        
        # Calculate profit
        profit = amount - self.start_amount
        profit_percentage = (profit / self.start_amount) * 100
        
        # Determine if profitable based on percentage threshold
        is_profitable = profit_percentage > self.min_profit_pct
        
        # Create opportunity object
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.now(),
            start_amount=self.start_amount,
            end_amount=amount,
            profit=profit,
            profit_percentage=profit_percentage,
            prices=prices,
            path=path,
            is_profitable=is_profitable
        )
        
        return opportunity
    
    def find_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Find all arbitrage opportunities.
        
        Returns:
            List of ArbitrageOpportunity objects
        """
        # Get current prices
        prices = self.get_current_prices()
        
        # Default paths
        default_paths = [
            ['CAD', 'BTC', 'ETH', 'CAD'],
            ['CAD', 'ETH', 'BTC', 'CAD']
        ]
        
        # Calculate opportunities for all paths
        opportunities = []
        
        # First, try the default paths
        for path in default_paths:
            if path[0] == 'CAD' and path[1] == 'BTC' and path[2] == 'ETH' and path[3] == 'CAD':
                opportunities.append(self.calculate_cad_btc_eth_cad(prices))
            elif path[0] == 'CAD' and path[1] == 'ETH' and path[2] == 'BTC' and path[3] == 'CAD':
                opportunities.append(self.calculate_cad_eth_btc_cad(prices))
            else:
                opportunities.append(self.calculate_custom_path(path, prices))
        
        # Look for additional paths in the trading pairs
        # This allows us to dynamically discover new triangular paths
        if len(self.trading_pairs) > 3:
            # Extract all currencies from trading pairs
            currencies = set()
            for pair in self.trading_pairs:
                base, quote = pair.split('/')
                currencies.add(base)
                currencies.add(quote)
            
            # Find base currency (assume it's the first currency in the first default path)
            base_currency = default_paths[0][0] if default_paths else 'CAD'
            
            # Find all possible triangular paths
            for c1 in currencies:
                for c2 in currencies:
                    if c1 != c2 and c1 != base_currency and c2 != base_currency:
                        path = [base_currency, c1, c2, base_currency]
                        
                        # Check if this path is already covered
                        if path not in [op.path for op in opportunities]:
                            # Check if all pairs exist
                            pair1 = f"{base_currency}/{c1}" in prices or f"{c1}/{base_currency}" in prices
                            pair2 = f"{c1}/{c2}" in prices or f"{c2}/{c1}" in prices
                            pair3 = f"{c2}/{base_currency}" in prices or f"{base_currency}/{c2}" in prices
                            
                            if pair1 and pair2 and pair3:
                                opportunities.append(self.calculate_custom_path(path, prices))
        
        return opportunities
    
    def find_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """
        Find the best arbitrage opportunity.
        
        Returns:
            Best ArbitrageOpportunity or None if no profitable opportunities
        """
        opportunities = self.find_opportunities()
        profitable_opportunities = [op for op in opportunities if op.is_profitable]
        
        if not profitable_opportunities:
            return None
        
        # Check if we're in an optimal trading hour
        current_hour = datetime.now().hour
        
        # Try to load optimal trading hours from statistical analysis
        optimal_hours = []
        try:
            from statistical_analyzer import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer()
            recommendations = analyzer.generate_trading_recommendations()
            optimal_hours = recommendations.get('optimal_trading_hours', [])
        except (ImportError, Exception):
            pass
        
        # If we're in an optimal trading hour, boost the profit score
        if optimal_hours and current_hour in optimal_hours:
            # Sort by boosted profit percentage
            return max(profitable_opportunities, key=lambda op: op.profit_percentage * 1.2)
        
        # Otherwise, return the opportunity with the highest profit percentage
        return max(profitable_opportunities, key=lambda op: op.profit_percentage)