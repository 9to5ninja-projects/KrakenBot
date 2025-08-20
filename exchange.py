"""
Exchange module for interacting with Kraken via ccxt.
Handles API connections, data fetching, and error handling.
"""
import ccxt
import time
from loguru import logger
from typing import Dict, Any, List, Optional

import config

class ExchangeManager:
    """Manages exchange connections and API interactions."""
    
    def __init__(self, use_api_keys: bool = False):
        """
        Initialize the exchange manager.
        
        Args:
            use_api_keys: Whether to use API keys for authenticated requests
        """
        self.exchange_id = 'kraken'
        self.exchange_params = {}
        
        if use_api_keys and config.KRAKEN_API_KEY and config.KRAKEN_SECRET:
            self.exchange_params = {
                'apiKey': config.KRAKEN_API_KEY,
                'secret': config.KRAKEN_SECRET,
            }
        
        self.exchange = self._initialize_exchange()
        self.markets = {}
        self.last_fetch_time = 0
        self.fetch_interval = 60  # Minimum time between market fetches (seconds)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_interval = 3  # Minimum time between any API requests (seconds)
        self.request_count = 0
        self.max_requests_per_minute = 15  # Kraken's public API limit is around 15-20 req/min
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize the exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class(self.exchange_params)
            exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_id}")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid hitting API limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Ensure minimum time between requests
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        
        # Track request count for per-minute limiting
        self.request_count += 1
        
        # If we're approaching the per-minute limit, slow down more
        if self.request_count >= self.max_requests_per_minute:
            logger.warning(f"Approaching rate limit ({self.request_count} requests). Slowing down...")
            time.sleep(5)  # Additional delay
            self.request_count = 0  # Reset counter
        
        # Update last request time
        self.last_request_time = time.time()
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data for a symbol with error handling and rate limiting.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/CAD')
            
        Returns:
            Ticker data dictionary
        """
        try:
            # Apply rate limiting
            self._rate_limit()
            
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit exceeded for {symbol}, waiting 30 seconds")
            time.sleep(30)
            self.request_count = 0  # Reset counter
            return self.fetch_ticker(symbol)
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            time.sleep(5)
            return self.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch ticker data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Dictionary of ticker data keyed by symbol
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch_ticker(symbol)
        return result
    
    def get_last_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get the last price for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Dictionary of last prices keyed by symbol
        """
        tickers = self.fetch_tickers(symbols)
        return {symbol: ticker['last'] for symbol, ticker in tickers.items()}
    
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create an order on the exchange.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type ('limit' or 'market')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (required for limit orders)
            
        Returns:
            Order information
        """
        if config.TRADING_MODE != 'live':
            logger.info(f"SIMULATION: Would place {order_type} {side} order for {amount} {symbol} at {price}")
            return {
                'id': 'simulation',
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'closed',
                'timestamp': int(time.time() * 1000),
                'datetime': self.exchange.iso8601(int(time.time() * 1000)),
                'fee': None,
                'cost': amount * (price or 0),
                'filled': amount,
                'remaining': 0,
                'info': {'simulation': True}
            }
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            return self.exchange.create_order(symbol, order_type, side, amount, price)
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit exceeded creating order, waiting 30 seconds")
            time.sleep(30)
            self.request_count = 0  # Reset counter
            return self.create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dictionary of balances keyed by currency
        """
        if config.TRADING_MODE != 'live':
            # Return simulated balance
            return {
                'CAD': config.START_AMOUNT,
                'BTC': 0.0,
                'ETH': 0.0
            }
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            balance = self.exchange.fetch_balance()
            return balance['free']
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit exceeded fetching balance, waiting 30 seconds")
            time.sleep(30)
            self.request_count = 0  # Reset counter
            return self.get_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_trading_fees(self) -> Dict[str, Dict[str, float]]:
        """
        Get trading fees for all markets.
        
        Returns:
            Dictionary of trading fees
        """
        try:
            return self.exchange.fetch_trading_fees()
        except Exception as e:
            logger.warning(f"Could not fetch trading fees: {e}")
            # Return default fees from config
            return {
                'maker': config.MAKER_FEE,
                'taker': config.TAKER_FEE
            }
    
    def get_historical_ohlcv(self, symbol: str, timeframe: str = '1d', 
                            since: Optional[int] = None, limit: int = 100) -> List[List]:
        """
        Get historical OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
            since: Timestamp in milliseconds for start time
            limit: Maximum number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        try:
            # Add retry logic for rate limiting
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                    return ohlcv
                except ccxt.RateLimitExceeded:
                    retry_count += 1
                    wait_time = 10 * retry_count  # Exponential backoff
                    logger.warning(f"Rate limit exceeded, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 5 * retry_count
                    logger.warning(f"Network error, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
            
            logger.error(f"Failed to fetch OHLCV data after {max_retries} retries")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return []
    
    def get_markets(self) -> List[Dict]:
        """
        Get all available markets on the exchange.
        
        Returns:
            List of market dictionaries
        """
        current_time = time.time()
        
        # Only fetch markets if we haven't fetched recently
        if not self.markets or (current_time - self.last_fetch_time) > self.fetch_interval:
            try:
                self.exchange.load_markets()
                self.markets = list(self.exchange.markets.values())
                self.last_fetch_time = current_time
            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                if not self.markets:
                    raise
        
        return self.markets