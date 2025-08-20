"""
Dashboard module for the KrakenBot triangular arbitrage system.
Provides a Streamlit-based web dashboard for monitoring arbitrage opportunities.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import threading

import config
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator

# Set page configuration
st.set_page_config(
    page_title="KrakenBot Arbitrage Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
refresh_interval = 10  # seconds
last_refresh_time = datetime.now()
exchange = None
calculator = None
stop_thread = False

def load_data():
    """Load data from CSV files."""
    opportunities_file = config.DATA_DIR / "opportunities.csv"
    trades_file = config.DATA_DIR / "trades.csv"
    stats_file = config.DATA_DIR / "stats.json"
    
    # Load opportunities
    if opportunities_file.exists():
        opportunities_df = pd.read_csv(opportunities_file)
        opportunities_df['timestamp'] = pd.to_datetime(opportunities_df['timestamp'])
        opportunities_df = opportunities_df.sort_values('timestamp', ascending=False)
    else:
        opportunities_df = pd.DataFrame()
    
    # Load trades
    if trades_file.exists():
        trades_df = pd.read_csv(trades_file)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)
    else:
        trades_df = pd.DataFrame()
    
    # Load stats
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {
            'total_opportunities': 0,
            'profitable_opportunities': 0,
            'total_profit': 0.0,
            'average_profit': 0.0,
            'max_profit': 0.0,
            'last_updated': datetime.now().isoformat(),
            'hourly_stats': {},
            'daily_stats': {}
        }
    
    return opportunities_df, trades_df, stats

def fetch_current_prices():
    """Fetch current prices from the exchange."""
    global exchange, calculator
    
    if exchange is None:
        exchange = ExchangeManager(use_api_keys=False)
        calculator = ArbitrageCalculator(exchange)
    
    try:
        prices = calculator.get_current_prices()
        opportunities = calculator.find_opportunities()
        return prices, opportunities
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return {}, []

def background_refresh():
    """Background thread to refresh data periodically."""
    global stop_thread
    
    while not stop_thread:
        # This will trigger a rerun of the app
        if 'refresh_state' in st.session_state:
            st.session_state.refresh_state += 1
        time.sleep(refresh_interval)

def main():
    """Main dashboard function."""
    global last_refresh_time, refresh_interval
    
    # Initialize session state for background refresh
    if 'refresh_state' not in st.session_state:
        st.session_state.refresh_state = 0
        # Start background thread
        threading.Thread(target=background_refresh, daemon=True).start()
    
    # Title and description
    st.title("KrakenBot Triangular Arbitrage Monitor")
    st.markdown("""
    This dashboard displays real-time arbitrage opportunities between BTC, ETH, and CAD on Kraken.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    
    # Load data
    opportunities_df, trades_df, stats = load_data()
    
    # Current time and last refresh
    current_time = datetime.now()
    st.sidebar.write(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write(f"Last data refresh: {last_refresh_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Manual refresh button
    if st.sidebar.button("Refresh Data"):
        st.session_state.refresh_state += 1
        last_refresh_time = current_time
    
    # Display statistics
    st.header("Arbitrage Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Opportunities", stats.get('total_opportunities', 0))
    
    with col2:
        st.metric("Profitable Opportunities", stats.get('profitable_opportunities', 0))
    
    with col3:
        st.metric("Total Profit", f"${stats.get('total_profit', 0):.2f}")
    
    with col4:
        st.metric("Average Profit", f"${stats.get('average_profit', 0):.2f}")
    
    # Current prices and opportunities
    st.header("Current Market Status")
    
    # Fetch current prices and opportunities
    prices, current_opportunities = fetch_current_prices()
    
    # Display current prices
    if prices:
        price_col1, price_col2, price_col3 = st.columns(3)
        
        with price_col1:
            st.metric("BTC/CAD", f"${prices.get('BTC/CAD', 0):.2f}")
        
        with price_col2:
            st.metric("ETH/CAD", f"${prices.get('ETH/CAD', 0):.2f}")
        
        with price_col3:
            st.metric("ETH/BTC", f"{prices.get('ETH/BTC', 0):.6f}")
    
    # Display current opportunities
    if current_opportunities:
        st.subheader("Current Arbitrage Opportunities")
        
        for i, opportunity in enumerate(current_opportunities):
            with st.expander(f"Path: {' → '.join(opportunity.path)} | Profit: ${opportunity.profit:.2f} ({opportunity.profit_percentage:.4f}%)"):
                st.write(f"Start Amount: ${opportunity.start_amount:.2f}")
                st.write(f"End Amount: ${opportunity.end_amount:.2f}")
                st.write(f"Profit: ${opportunity.profit:.2f} ({opportunity.profit_percentage:.4f}%)")
                st.write(f"Profitable: {'Yes' if opportunity.is_profitable else 'No'}")
                st.write(f"Prices: {opportunity.prices}")
    
    # Historical opportunities
    st.header("Historical Opportunities")
    
    if not opportunities_df.empty:
        # Filter to show only last 24 hours by default
        time_filter = st.selectbox(
            "Time filter",
            ["Last hour", "Last 24 hours", "Last 7 days", "All time"]
        )
        
        if time_filter == "Last hour":
            filtered_df = opportunities_df[opportunities_df['timestamp'] > (current_time - timedelta(hours=1))]
        elif time_filter == "Last 24 hours":
            filtered_df = opportunities_df[opportunities_df['timestamp'] > (current_time - timedelta(days=1))]
        elif time_filter == "Last 7 days":
            filtered_df = opportunities_df[opportunities_df['timestamp'] > (current_time - timedelta(days=7))]
        else:
            filtered_df = opportunities_df
        
        # Show profitable only checkbox
        show_profitable_only = st.checkbox("Show profitable opportunities only")
        if show_profitable_only:
            filtered_df = filtered_df[filtered_df['is_profitable'] == True]
        
        # Display opportunities table
        st.dataframe(filtered_df)
        
        # Plot profit over time
        if not filtered_df.empty:
            st.subheader("Profit Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            filtered_df.set_index('timestamp')['profit'].plot(ax=ax)
            ax.set_ylabel("Profit (CAD)")
            ax.set_xlabel("Time")
            ax.grid(True)
            st.pyplot(fig)
    else:
        st.info("No historical data available yet. Start the monitor to collect data.")
    
    # Footer
    st.markdown("---")
    st.markdown("KrakenBot Triangular Arbitrage Monitor | Developed for educational purposes")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Set stop flag for background thread when app is closed
        stop_thread = True