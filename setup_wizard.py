"""
Configuration wizard for KrakenBot.
Helps users set up their environment and configure the system.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import getpass

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the wizard header."""
    clear_screen()
    print("=" * 80)
    print("KrakenBot Configuration Wizard".center(80))
    print("=" * 80)
    print("This wizard will help you set up your KrakenBot configuration.\n")

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80 + "\n")

def get_input(prompt, default=None, password=False):
    """Get user input with a default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    if password:
        value = getpass.getpass(prompt)
    else:
        value = input(prompt)
    
    return value if value else default

def get_numeric_input(prompt, default=None, min_value=None, max_value=None):
    """Get numeric input with validation."""
    while True:
        value = get_input(prompt, default)
        try:
            value = float(value)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def get_boolean_input(prompt, default=None):
    """Get a yes/no input."""
    default_str = "Y/n" if default in (True, None) else "y/N"
    prompt = f"{prompt} [{default_str}]"
    
    while True:
        value = input(f"{prompt}: ").strip().lower()
        if not value:
            return default if default is not None else True
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'")

def get_choice_input(prompt, choices, default=None):
    """Get a choice from a list of options."""
    print(f"{prompt}:")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    
    default_index = choices.index(default) + 1 if default in choices else None
    default_str = f" [{default_index}]" if default_index else ""
    
    while True:
        try:
            value = input(f"Enter your choice{default_str}: ").strip()
            if not value and default_index:
                return choices[default_index - 1]
            
            choice_index = int(value) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index]
            print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")

def configure_api_keys():
    """Configure Kraken API keys."""
    print_section("Kraken API Configuration")
    print("API keys are optional for monitoring mode but required for live trading.")
    print("You can get your API keys from https://www.kraken.com/u/security/api\n")
    
    use_api = get_boolean_input("Do you want to configure API keys now?", default=False)
    
    if use_api:
        api_key = get_input("Enter your Kraken API Key", password=False)
        api_secret = get_input("Enter your Kraken API Secret", password=True)
        return {
            'KRAKEN_API_KEY': api_key,
            'KRAKEN_SECRET': api_secret
        }
    else:
        print("\nYou can add API keys later by editing the .env file.")
        return {
            'KRAKEN_API_KEY': '',
            'KRAKEN_SECRET': ''
        }

def configure_trading_parameters():
    """Configure trading parameters."""
    print_section("Trading Parameters")
    print("Configure the basic parameters for arbitrage monitoring and trading.\n")
    
    start_amount = get_numeric_input(
        "Enter your starting amount in CAD",
        default="100",
        min_value=10
    )
    
    # Legacy absolute profit threshold
    profit_threshold = get_numeric_input(
        "Enter your minimum profit threshold in CAD (legacy parameter)",
        default="2",
        min_value=0
    )
    
    # New percentage-based profit threshold
    min_profit_pct = get_numeric_input(
        "Enter your minimum profit percentage (e.g., 1.5 for 1.5%)",
        default="1.5",
        min_value=0
    )
    
    maker_fee = get_numeric_input(
        "Enter the maker fee percentage (e.g., 0.0025 for 0.25%)",
        default="0.0025",
        min_value=0,
        max_value=0.01
    )
    
    taker_fee = get_numeric_input(
        "Enter the taker fee percentage (e.g., 0.0040 for 0.40%)",
        default="0.0040",
        min_value=0,
        max_value=0.01
    )
    
    slippage_buffer = get_numeric_input(
        "Enter the slippage buffer percentage (e.g., 0.001 for 0.1%)",
        default="0.001",
        min_value=0,
        max_value=0.01
    )
    
    # Calculate and display the breakeven threshold
    breakeven = (taker_fee + slippage_buffer) * 3 * 100
    print(f"\nBased on your settings:")
    print(f"- Approximate breakeven threshold: {breakeven:.2f}%")
    print(f"- Your profit threshold: {min_profit_pct:.2f}%")
    print(f"- Margin above breakeven: {min_profit_pct - breakeven:.2f}%")
    
    if min_profit_pct <= breakeven:
        print("\nWARNING: Your profit threshold is below or equal to the breakeven point.")
        print("This may result in unprofitable trades after fees and slippage.")
        adjust = get_boolean_input("Would you like to adjust your profit threshold?", default=True)
        if adjust:
            suggested = breakeven + 0.5
            min_profit_pct = get_numeric_input(
                f"Enter your minimum profit percentage (suggested: {suggested:.2f}%)",
                default=str(suggested),
                min_value=0
            )
    
    return {
        'START_AMOUNT': str(start_amount),
        'PROFIT_THRESHOLD': str(profit_threshold),
        'MIN_PROFIT_PCT': str(min_profit_pct),
        'MAKER_FEE': str(maker_fee),
        'TAKER_FEE': str(taker_fee),
        'SLIPPAGE_BUFFER': str(slippage_buffer)
    }

def configure_monitoring_settings():
    """Configure monitoring settings."""
    print_section("Monitoring Settings")
    print("Configure how frequently the system checks for arbitrage opportunities.\n")
    
    check_interval = get_numeric_input(
        "Enter the check interval in seconds",
        default="10",
        min_value=1
    )
    
    log_level_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = get_choice_input(
        "Select the logging level",
        log_level_choices,
        default="INFO"
    )
    
    return {
        'CHECK_INTERVAL': str(int(check_interval)),
        'LOG_LEVEL': log_level
    }

def configure_trading_pairs():
    """Configure trading pairs."""
    print_section("Trading Pairs")
    print("Configure the trading pairs for triangular arbitrage.\n")
    print("The default pairs are BTC/CAD, ETH/CAD, and ETH/BTC.")
    
    customize_pairs = get_boolean_input("Do you want to customize the trading pairs?", default=False)
    
    if customize_pairs:
        pair1 = get_input("Enter the first trading pair", default="BTC/CAD")
        pair2 = get_input("Enter the second trading pair", default="ETH/CAD")
        pair3 = get_input("Enter the third trading pair", default="ETH/BTC")
        return {
            'PAIR_1': pair1,
            'PAIR_2': pair2,
            'PAIR_3': pair3
        }
    else:
        return {
            'PAIR_1': "BTC/CAD",
            'PAIR_2': "ETH/CAD",
            'PAIR_3': "ETH/BTC"
        }

def configure_notifications():
    """Configure notification settings."""
    print_section("Notifications")
    print("Configure how you want to be notified of profitable opportunities.\n")
    
    enable_notifications = get_boolean_input("Do you want to enable notifications?", default=True)
    
    if enable_notifications:
        notification_methods = ["Console Only", "Email", "Telegram", "Discord"]
        notification_method = get_choice_input(
            "Select your preferred notification method",
            notification_methods,
            default="Console Only"
        )
        
        notification_config = {
            'ENABLE_NOTIFICATIONS': 'true',
            'NOTIFICATION_METHOD': notification_method
        }
        
        if notification_method == "Email":
            smtp_server = get_input("Enter your SMTP server", default="smtp.gmail.com")
            smtp_port = get_input("Enter your SMTP port", default="587")
            smtp_username = get_input("Enter your email address")
            smtp_password = get_input("Enter your email password or app password", password=True)
            recipient_email = get_input("Enter the recipient email address", default=smtp_username)
            
            notification_config.update({
                'SMTP_SERVER': smtp_server,
                'SMTP_PORT': smtp_port,
                'SMTP_USERNAME': smtp_username,
                'SMTP_PASSWORD': smtp_password,
                'RECIPIENT_EMAIL': recipient_email
            })
        
        elif notification_method == "Telegram":
            bot_token = get_input("Enter your Telegram bot token")
            chat_id = get_input("Enter your Telegram chat ID")
            
            notification_config.update({
                'TELEGRAM_BOT_TOKEN': bot_token,
                'TELEGRAM_CHAT_ID': chat_id
            })
        
        elif notification_method == "Discord":
            webhook_url = get_input("Enter your Discord webhook URL")
            
            notification_config.update({
                'DISCORD_WEBHOOK_URL': webhook_url
            })
        
        return notification_config
    else:
        return {
            'ENABLE_NOTIFICATIONS': 'false',
            'NOTIFICATION_METHOD': 'Console Only'
        }

def save_configuration(config):
    """Save the configuration to .env file."""
    env_path = Path(__file__).resolve().parent / '.env'
    
    with open(env_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    print(f"\nConfiguration saved to {env_path}")

def run_wizard():
    """Run the configuration wizard."""
    print_header()
    
    # Get user's name for a personalized experience
    user_name = get_input("What's your name?", default="Trader")
    print(f"\nHello, {user_name}! Let's set up your KrakenBot configuration.\n")
    time.sleep(1)
    
    # Configure each section
    config = {}
    config.update(configure_api_keys())
    config.update(configure_trading_parameters())
    config.update(configure_monitoring_settings())
    config.update(configure_trading_pairs())
    config.update(configure_notifications())
    
    # Add trading mode
    print_section("Trading Mode")
    print("Select the trading mode for your KrakenBot.\n")
    
    mode_choices = ["monitor", "simulate", "live"]
    mode_descriptions = {
        "monitor": "Monitor only - No trades will be executed",
        "simulate": "Simulation - Trades will be simulated but not executed",
        "live": "Live Trading - Real trades will be executed (requires API keys)"
    }
    
    for mode in mode_choices:
        print(f"  - {mode}: {mode_descriptions[mode]}")
    
    trading_mode = get_choice_input(
        "Select your trading mode",
        mode_choices,
        default="monitor"
    )
    
    if trading_mode == "live" and not config.get('KRAKEN_API_KEY'):
        print("\nWarning: Live trading requires API keys. Please configure them first.")
        configure_api = get_boolean_input("Do you want to configure API keys now?", default=True)
        if configure_api:
            config.update(configure_api_keys())
        else:
            print("\nSwitching to monitor mode since API keys are not configured.")
            trading_mode = "monitor"
    
    config['TRADING_MODE'] = trading_mode
    
    # Review and save
    print_section("Configuration Review")
    print("Please review your configuration:\n")
    
    # Hide sensitive information
    review_config = config.copy()
    if review_config.get('KRAKEN_API_KEY'):
        review_config['KRAKEN_API_KEY'] = '*' * 8 + review_config['KRAKEN_API_KEY'][-4:]
    if review_config.get('KRAKEN_SECRET'):
        review_config['KRAKEN_SECRET'] = '*' * 12
    if review_config.get('SMTP_PASSWORD'):
        review_config['SMTP_PASSWORD'] = '*' * 8
    
    for key, value in review_config.items():
        print(f"  {key} = {value}")
    
    save_config = get_boolean_input("\nDo you want to save this configuration?", default=True)
    
    if save_config:
        save_configuration(config)
        print("\nConfiguration saved successfully!")
        
        start_bot = get_boolean_input("Do you want to start KrakenBot now?", default=True)
        if start_bot:
            print("\nStarting KrakenBot...")
            time.sleep(1)
            # Import here to avoid circular imports
            from main import main
            sys.argv = ['main.py', '--mode', trading_mode]
            main()
        else:
            print("\nYou can start KrakenBot later by running:")
            print(f"  python main.py --mode {trading_mode}")
    else:
        print("\nConfiguration not saved. You can run the wizard again to configure KrakenBot.")

if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        print("\n\nConfiguration wizard interrupted. You can run it again anytime.")
        sys.exit(0)