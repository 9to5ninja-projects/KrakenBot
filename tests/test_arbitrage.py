"""
Test module for the arbitrage calculation functionality.
"""
import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbitrage import ArbitrageCalculator, ArbitrageOpportunity
from exchange import ExchangeManager

class TestArbitrageCalculator(unittest.TestCase):
    """Test cases for the ArbitrageCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the exchange manager
        self.exchange = MagicMock(spec=ExchangeManager)
        self.calculator = ArbitrageCalculator(self.exchange)
        
        # Mock prices
        self.mock_prices = {
            'BTC/CAD': 80000.0,
            'ETH/CAD': 5000.0,
            'ETH/BTC': 0.0625  # 5000/80000 = 0.0625
        }
        
        # Set up the exchange to return mock prices
        self.exchange.get_last_prices.return_value = self.mock_prices
    
    def test_get_current_prices(self):
        """Test getting current prices."""
        prices = self.calculator.get_current_prices()
        self.exchange.get_last_prices.assert_called_once_with(self.calculator.trading_pairs)
        self.assertEqual(prices, self.mock_prices)
    
    def test_calculate_cad_btc_eth_cad_no_profit(self):
        """Test CAD → BTC → ETH → CAD path with no profit."""
        # With these prices and fees, there should be no profit
        opportunity = self.calculator.calculate_cad_btc_eth_cad(self.mock_prices)
        
        # Verify the calculation
        self.assertIsInstance(opportunity, ArbitrageOpportunity)
        self.assertEqual(opportunity.path, ['CAD', 'BTC', 'ETH', 'CAD'])
        self.assertLess(opportunity.profit, 0)  # Should be a loss due to fees
        self.assertFalse(opportunity.is_profitable)
    
    def test_calculate_cad_btc_eth_cad_with_profit(self):
        """Test CAD → BTC → ETH → CAD path with profit."""
        # Create prices that would result in profit
        profitable_prices = {
            'BTC/CAD': 80000.0,
            'ETH/CAD': 5200.0,  # Higher ETH/CAD price creates arbitrage opportunity
            'ETH/BTC': 0.0625
        }
        
        # Calculate with profitable prices
        opportunity = self.calculator.calculate_cad_btc_eth_cad(profitable_prices)
        
        # Verify the calculation
        self.assertIsInstance(opportunity, ArbitrageOpportunity)
        self.assertEqual(opportunity.path, ['CAD', 'BTC', 'ETH', 'CAD'])
        self.assertGreater(opportunity.profit, 0)  # Should be a profit
        # Whether it's considered profitable depends on the threshold
        
    def test_find_opportunities(self):
        """Test finding all opportunities."""
        opportunities = self.calculator.find_opportunities()
        
        # Should return a list of two opportunities (both paths)
        self.assertEqual(len(opportunities), 2)
        self.assertIsInstance(opportunities[0], ArbitrageOpportunity)
        self.assertIsInstance(opportunities[1], ArbitrageOpportunity)
        
        # Verify paths
        paths = [op.path for op in opportunities]
        self.assertIn(['CAD', 'BTC', 'ETH', 'CAD'], paths)
        self.assertIn(['CAD', 'ETH', 'BTC', 'CAD'], paths)
    
    def test_find_best_opportunity_none_profitable(self):
        """Test finding best opportunity when none are profitable."""
        # Mock find_opportunities to return non-profitable opportunities
        with patch.object(self.calculator, 'find_opportunities') as mock_find:
            mock_find.return_value = [
                ArbitrageOpportunity(
                    timestamp=datetime.now(),
                    start_amount=10000,
                    end_amount=9900,
                    profit=-100,
                    profit_percentage=-1.0,
                    prices=self.mock_prices,
                    path=['CAD', 'BTC', 'ETH', 'CAD'],
                    is_profitable=False
                ),
                ArbitrageOpportunity(
                    timestamp=datetime.now(),
                    start_amount=10000,
                    end_amount=9950,
                    profit=-50,
                    profit_percentage=-0.5,
                    prices=self.mock_prices,
                    path=['CAD', 'ETH', 'BTC', 'CAD'],
                    is_profitable=False
                )
            ]
            
            best = self.calculator.find_best_opportunity()
            self.assertIsNone(best)
    
    def test_find_best_opportunity_with_profitable(self):
        """Test finding best opportunity when some are profitable."""
        # Mock find_opportunities to return one profitable opportunity
        with patch.object(self.calculator, 'find_opportunities') as mock_find:
            mock_find.return_value = [
                ArbitrageOpportunity(
                    timestamp=datetime.now(),
                    start_amount=10000,
                    end_amount=9900,
                    profit=-100,
                    profit_percentage=-1.0,
                    prices=self.mock_prices,
                    path=['CAD', 'BTC', 'ETH', 'CAD'],
                    is_profitable=False
                ),
                ArbitrageOpportunity(
                    timestamp=datetime.now(),
                    start_amount=10000,
                    end_amount=10050,
                    profit=50,
                    profit_percentage=0.5,
                    prices=self.mock_prices,
                    path=['CAD', 'ETH', 'BTC', 'CAD'],
                    is_profitable=True
                )
            ]
            
            best = self.calculator.find_best_opportunity()
            self.assertIsNotNone(best)
            self.assertEqual(best.profit, 50)
            self.assertEqual(best.path, ['CAD', 'ETH', 'BTC', 'CAD'])


if __name__ == '__main__':
    unittest.main()