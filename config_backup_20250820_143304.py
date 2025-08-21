"""
Configuration module for the KrakenBot triangular arbitrage system.
Loads settings from environment variables or .env file.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
load_dotenv()

# API credentials
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_SECRET = os.getenv('KRAKEN_SECRET', '')

# Trading pairs
TRADING_PAIRS = {
    'PAIR_1': os.getenv('PAIR_1', 'BTC/CAD'),
    'PAIR_2': os.getenv('PAIR_2', 'ETH/CAD'),
    'PAIR_3': os.getenv('PAIR_3', 'ETH/BTC')
}

# Trading parameters
START_AMOUNT = float(os.getenv('START_AMOUNT', 100))  # Starting amount in CAD
PROFIT_THRESHOLD = float(os.getenv('PROFIT_THRESHOLD', 2))  # Minimum profit in CAD (legacy)
MIN_PROFIT_PCT = float(os.getenv('MIN_PROFIT_PCT', 1.5))  # Minimum profit percentage (1.5%)
MAKER_FEE = float(os.getenv('MAKER_FEE', 0.0025))  # 0.25% maker fee
TAKER_FEE = float(os.getenv('TAKER_FEE', 0.0040))  # 0.40% taker fee
SLIPPAGE_BUFFER = float(os.getenv('SLIPPAGE_BUFFER', 0.001))  # 0.1% slippage buffer

# Position sizing and capital management
MAX_CAPITAL = float(os.getenv('MAX_CAPITAL', 100))  # Maximum capital to use for trading
POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', 2))  # Position size as percentage of MAX_CAPITAL
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))  # Maximum number of concurrent positions

# Calculate effective breakeven threshold (3 trades with taker fee + slippage)
BREAKEVEN_PCT = (TAKER_FEE + SLIPPAGE_BUFFER) * 3 * 100  # Convert to percentage

# Monitoring settings
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 10))  # Time between checks in seconds
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'

# Simple Pair Trading Parameters
BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', -0.008))  # Buy when price drops 0.8% from recent high
SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', 0.012))  # Sell when price rises 1.2% from recent low
LOOKBACK_PERIODS = int(os.getenv('LOOKBACK_PERIODS', 20))  # Number of periods to look back for high/low
MIN_TRADE_AMOUNT = float(os.getenv('MIN_TRADE_AMOUNT', 25.0))  # Minimum CAD amount per trade
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.3))  # Maximum 30% of portfolio per position

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Trading mode
# 'monitor' - Only monitor and log opportunities
# 'simulate' - Simulate trades but don't execute
# 'live' - Execute real trades
TRADING_MODE = os.getenv('TRADING_MODE', 'monitor')

def validate_config():
    """Validate the configuration settings."""
    if TRADING_MODE not in ['monitor', 'simulate', 'live']:
        raise ValueError(f"Invalid TRADING_MODE: {TRADING_MODE}. Must be 'monitor', 'simulate', or 'live'")
    
    if TRADING_MODE == 'live' and (not KRAKEN_API_KEY or not KRAKEN_SECRET):
        raise ValueError("API credentials are required for live trading mode")
    
    # Validate trading pairs
    for pair_name, pair_value in TRADING_PAIRS.items():
        if '/' not in pair_value:
            raise ValueError(f"Invalid trading pair format for {pair_name}: {pair_value}")
    
    # Validate numeric parameters
    if START_AMOUNT <= 0:
        raise ValueError(f"START_AMOUNT must be positive, got {START_AMOUNT}")
    
    if MIN_PROFIT_PCT < 0:
        raise ValueError(f"MIN_PROFIT_PCT must be positive, got {MIN_PROFIT_PCT}")
    
    if MAKER_FEE < 0 or MAKER_FEE > 1:
        raise ValueError(f"MAKER_FEE must be between 0 and 1, got {MAKER_FEE}")
    
    if TAKER_FEE < 0 or TAKER_FEE > 1:
        raise ValueError(f"TAKER_FEE must be between 0 and 1, got {TAKER_FEE}")
    
    if SLIPPAGE_BUFFER < 0 or SLIPPAGE_BUFFER > 1:
        raise ValueError(f"SLIPPAGE_BUFFER must be between 0 and 1, got {SLIPPAGE_BUFFER}")
    
    if CHECK_INTERVAL < 1:
        raise ValueError(f"CHECK_INTERVAL must be at least 1 second, got {CHECK_INTERVAL}")
    
    # Validate position sizing and capital management parameters
    if MAX_CAPITAL <= 0:
        raise ValueError(f"MAX_CAPITAL must be positive, got {MAX_CAPITAL}")
    
    if POSITION_SIZE_PCT <= 0 or POSITION_SIZE_PCT > 100:
        raise ValueError(f"POSITION_SIZE_PCT must be between 0 and 100, got {POSITION_SIZE_PCT}")
    
    if MAX_POSITIONS < 1:
        raise ValueError(f"MAX_POSITIONS must be at least 1, got {MAX_POSITIONS}")
    
    # Warn if position size is too small
    min_position_size = (MAX_CAPITAL * POSITION_SIZE_PCT / 100)
    if min_position_size < 1:
        print(f"WARNING: Minimum position size ({min_position_size:.2f}) is very small. Consider increasing POSITION_SIZE_PCT.")
    
    return True

# Validate configuration when module is imported
validate_config()