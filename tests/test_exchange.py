"""
Test module for the exchange functionality.
"""
import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import ccxt

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from exchange import ExchangeManager

class TestExchangeManager(unittest.TestCase):
    """Test cases for the ExchangeManager class."""
    
    @patch('ccxt.kraken')
    def setUp(self, mock_kraken):
        """Set up test fixtures."""
        # Configure the mock
        self.mock_exchange = mock_kraken.return_value
        self.mock_exchange.load_markets.return_value = None
        
        # Create exchange manager with no API keys
        self.exchange_manager = ExchangeManager(use_api_keys=False)
    
    def test_initialization(self):
        """Test initialization of exchange manager."""
        self.assertEqual(self.exchange_manager.exchange_id, 'kraken')
        self.assertEqual(self.exchange_manager.exchange_params, {})
        self.mock_exchange.load_markets.assert_called_once()
    
    def test_initialization_with_api_keys(self):
        """Test initialization with API keys."""
        # Save original values
        original_api_key = config.KRAKEN_API_KEY
        original_secret = config.KRAKEN_SECRET
        
        try:
            # Set test values
            config.KRAKEN_API_KEY = 'test_api_key'
            config.KRAKEN_SECRET = 'test_secret'
            
            with patch('ccxt.kraken') as mock_kraken:
                mock_exchange = mock_kraken.return_value
                mock_exchange.load_markets.return_value = None
                
                # Create exchange manager with API keys
                exchange_manager = ExchangeManager(use_api_keys=True)
                
                # Verify API keys were used
                self.assertEqual(exchange_manager.exchange_params['apiKey'], 'test_api_key')
                self.assertEqual(exchange_manager.exchange_params['secret'], 'test_secret')
                mock_exchange.load_markets.assert_called_once()
        
        finally:
            # Restore original values
            config.KRAKEN_API_KEY = original_api_key
            config.KRAKEN_SECRET = original_secret
    
    def test_fetch_ticker(self):
        """Test fetching ticker data."""
        # Configure mock
        mock_ticker = {
            'symbol': 'BTC/CAD',
            'last': 80000.0,
            'bid': 79900.0,
            'ask': 80100.0,
            'volume': 10.5
        }
        self.mock_exchange.fetch_ticker.return_value = mock_ticker
        
        # Call method
        ticker = self.exchange_manager.fetch_ticker('BTC/CAD')
        
        # Verify
        self.mock_exchange.fetch_ticker.assert_called_once_with('BTC/CAD')
        self.assertEqual(ticker, mock_ticker)
    
    def test_fetch_ticker_rate_limit_exceeded(self):
        """Test handling of rate limit exceeded error."""
        # Configure mock to raise rate limit error once, then succeed
        self.mock_exchange.fetch_ticker.side_effect = [
            ccxt.RateLimitExceeded('Rate limit exceeded'),
            {'symbol': 'BTC/CAD', 'last': 80000.0}
        ]
        
        # Call method
        with patch('time.sleep') as mock_sleep:
            ticker = self.exchange_manager.fetch_ticker('BTC/CAD')
        
        # Verify
        self.assertEqual(self.mock_exchange.fetch_ticker.call_count, 2)
        mock_sleep.assert_called_once_with(10)
        self.assertEqual(ticker['last'], 80000.0)
    
    def test_fetch_ticker_network_error(self):
        """Test handling of network error."""
        # Configure mock to raise network error once, then succeed
        self.mock_exchange.fetch_ticker.side_effect = [
            ccxt.NetworkError('Network error'),
            {'symbol': 'BTC/CAD', 'last': 80000.0}
        ]
        
        # Call method
        with patch('time.sleep') as mock_sleep:
            ticker = self.exchange_manager.fetch_ticker('BTC/CAD')
        
        # Verify
        self.assertEqual(self.mock_exchange.fetch_ticker.call_count, 2)
        mock_sleep.assert_called_once_with(5)
        self.assertEqual(ticker['last'], 80000.0)
    
    def test_get_last_prices(self):
        """Test getting last prices for multiple symbols."""
        # Configure mock
        self.mock_exchange.fetch_ticker.side_effect = [
            {'symbol': 'BTC/CAD', 'last': 80000.0},
            {'symbol': 'ETH/CAD', 'last': 5000.0},
            {'symbol': 'ETH/BTC', 'last': 0.0625}
        ]
        
        # Call method
        prices = self.exchange_manager.get_last_prices(['BTC/CAD', 'ETH/CAD', 'ETH/BTC'])
        
        # Verify
        self.assertEqual(self.mock_exchange.fetch_ticker.call_count, 3)
        self.assertEqual(prices, {
            'BTC/CAD': 80000.0,
            'ETH/CAD': 5000.0,
            'ETH/BTC': 0.0625
        })
    
    def test_create_order_simulation(self):
        """Test creating an order in simulation mode."""
        # Save original value
        original_mode = config.TRADING_MODE
        
        try:
            # Set to simulation mode
            config.TRADING_MODE = 'simulate'
            
            # Call method
            order = self.exchange_manager.create_order(
                symbol='BTC/CAD',
                order_type='limit',
                side='buy',
                amount=0.1,
                price=80000.0
            )
            
            # Verify
            self.assertEqual(order['symbol'], 'BTC/CAD')
            self.assertEqual(order['type'], 'limit')
            self.assertEqual(order['side'], 'buy')
            self.assertEqual(order['amount'], 0.1)
            self.assertEqual(order['price'], 80000.0)
            self.assertEqual(order['status'], 'closed')
            self.assertEqual(order['filled'], 0.1)
            self.assertEqual(order['remaining'], 0)
            self.assertEqual(order['cost'], 8000.0)  # 0.1 * 80000.0
            self.assertTrue(order['info']['simulation'])
            
            # Verify exchange method was not called
            self.mock_exchange.create_order.assert_not_called()
        
        finally:
            # Restore original value
            config.TRADING_MODE = original_mode
    
    def test_create_order_live(self):
        """Test creating an order in live mode."""
        # Save original value
        original_mode = config.TRADING_MODE
        
        try:
            # Set to live mode
            config.TRADING_MODE = 'live'
            
            # Configure mock
            mock_order = {
                'id': '12345',
                'symbol': 'BTC/CAD',
                'type': 'limit',
                'side': 'buy',
                'amount': 0.1,
                'price': 80000.0,
                'status': 'open'
            }
            self.mock_exchange.create_order.return_value = mock_order
            
            # Call method
            order = self.exchange_manager.create_order(
                symbol='BTC/CAD',
                order_type='limit',
                side='buy',
                amount=0.1,
                price=80000.0
            )
            
            # Verify
            self.mock_exchange.create_order.assert_called_once_with(
                'BTC/CAD', 'limit', 'buy', 0.1, 80000.0
            )
            self.assertEqual(order, mock_order)
        
        finally:
            # Restore original value
            config.TRADING_MODE = original_mode
    
    def test_get_balance_simulation(self):
        """Test getting balance in simulation mode."""
        # Save original values
        original_mode = config.TRADING_MODE
        original_amount = config.START_AMOUNT
        
        try:
            # Set to simulation mode
            config.TRADING_MODE = 'simulate'
            config.START_AMOUNT = 10000.0
            
            # Call method
            balance = self.exchange_manager.get_balance()
            
            # Verify
            self.assertEqual(balance['CAD'], 10000.0)
            self.assertEqual(balance['BTC'], 0.0)
            self.assertEqual(balance['ETH'], 0.0)
            
            # Verify exchange method was not called
            self.mock_exchange.fetch_balance.assert_not_called()
        
        finally:
            # Restore original values
            config.TRADING_MODE = original_mode
            config.START_AMOUNT = original_amount
    
    def test_get_balance_live(self):
        """Test getting balance in live mode."""
        # Save original value
        original_mode = config.TRADING_MODE
        
        try:
            # Set to live mode
            config.TRADING_MODE = 'live'
            
            # Configure mock
            mock_balance = {
                'free': {
                    'CAD': 10000.0,
                    'BTC': 0.1,
                    'ETH': 1.0
                },
                'used': {
                    'CAD': 0.0,
                    'BTC': 0.0,
                    'ETH': 0.0
                },
                'total': {
                    'CAD': 10000.0,
                    'BTC': 0.1,
                    'ETH': 1.0
                }
            }
            self.mock_exchange.fetch_balance.return_value = mock_balance
            
            # Call method
            balance = self.exchange_manager.get_balance()
            
            # Verify
            self.mock_exchange.fetch_balance.assert_called_once()
            self.assertEqual(balance, mock_balance['free'])
        
        finally:
            # Restore original value
            config.TRADING_MODE = original_mode


if __name__ == '__main__':
    unittest.main()