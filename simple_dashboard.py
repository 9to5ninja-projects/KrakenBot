"""
Simplified dashboard for KrakenBot with reliable tab loading.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

import config
from exchange import ExchangeManager

# Set page configuration
st.set_page_config(
    page_title="KrakenBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_arbitrage_data():
    """Load triangular arbitrage data."""
    try:
        opportunities_file = config.DATA_DIR / "opportunities.csv"
        
        if opportunities_file.exists():
            df = pd.read_csv(opportunities_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading arbitrage data: {e}")
        return pd.DataFrame()

def load_simple_pair_data():
    """Load simple pair trading data from the most recent session."""
    try:
        data_dir = Path("data")
        
        # Find the most recent simple pair monitor directory
        monitor_dirs = list(data_dir.glob("simple_pair_monitor_*")) + \
                      list(data_dir.glob("sample_simple_pair_*")) + \
                      list(data_dir.glob("simulated_volatile_*")) + \
                      list(data_dir.glob("test_simple_pair_*"))
        
        if not monitor_dirs:
            return {}, {}, {}, {}
        
        latest_dir = max(monitor_dirs, key=lambda x: x.stat().st_mtime)
        
        # Load data files
        trades_file = latest_dir / "trades.json"
        portfolio_file = latest_dir / "portfolio_history.json"
        performance_file = latest_dir / "performance_summary.json"
        price_history_file = latest_dir / "price_history.json"
        
        trades_data = []
        portfolio_data = []
        performance_data = {}
        price_history_data = {}
        
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                trades_data = json.load(f)
        
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
        
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
        
        if price_history_file.exists():
            with open(price_history_file, 'r') as f:
                price_history_data = json.load(f)
        
        return trades_data, portfolio_data, performance_data, price_history_data
    
    except Exception as e:
        st.error(f"Error loading simple pair data: {e}")
        return {}, {}, {}, {}

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_current_prices():
    """Get current market prices with caching."""
    try:
        exchange = ExchangeManager(use_api_keys=False)
        pairs = ['BTC/CAD', 'ETH/CAD', 'ETH/BTC']
        prices = {}
        
        for pair in pairs:
            ticker = exchange.fetch_ticker(pair)
            prices[pair] = ticker['last']
        
        return prices
    except Exception as e:
        st.error(f"Error fetching current prices: {e}")
        return {}

def show_overview():
    """Show overview of both strategies."""
    st.header("📊 Strategy Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔺 Triangular Arbitrage")
        arb_data = load_arbitrage_data()
        
        if not arb_data.empty:
            recent_opportunities = len(arb_data[arb_data['timestamp'] > datetime.now() - timedelta(hours=24)])
            profitable_opportunities = len(arb_data[arb_data['is_profitable'] == True])
            best_profit = arb_data['profit_percentage'].max()
            
            st.metric("24h Opportunities", recent_opportunities)
            st.metric("Profitable Found", profitable_opportunities)
            st.metric("Best Profit %", f"{best_profit:.4f}%")
        else:
            st.info("No triangular arbitrage data available")
    
    with col2:
        st.subheader("💱 Simple Pair Trading")
        trades_data, portfolio_data, performance_data, _ = load_simple_pair_data()
        
        if performance_data:
            st.metric("Total Return", f"{performance_data.get('total_return_pct', 0):.2f}%")
            st.metric("Total Trades", performance_data.get('total_trades', 0))
            st.metric("Current Value", f"${performance_data.get('current_value', 0):.2f}")
        else:
            st.info("No simple pair trading data available")
    
    # Portfolio performance chart
    if portfolio_data:
        st.subheader("📈 Portfolio Performance Over Time")
        
        df = pd.DataFrame(portfolio_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timestamp'], df['portfolio_value'], linewidth=2, color='#1f77b4')
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (CAD)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

def show_triangular_arbitrage():
    """Show triangular arbitrage analysis."""
    st.header("🔺 Triangular Arbitrage Analysis")
    
    arb_data = load_arbitrage_data()
    
    if arb_data.empty:
        st.info("No triangular arbitrage data available. Run the monitor to collect data.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_opportunities = len(arb_data)
        st.metric("Total Opportunities", total_opportunities)
    
    with col2:
        profitable_count = len(arb_data[arb_data['is_profitable'] == True])
        st.metric("Profitable Opportunities", profitable_count)
    
    with col3:
        if total_opportunities > 0:
            success_rate = (profitable_count / total_opportunities) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "0%")
    
    with col4:
        best_profit = arb_data['profit_percentage'].max()
        st.metric("Best Profit", f"{best_profit:.4f}%")
    
    # Profit distribution
    st.subheader("📊 Profit Distribution")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(arb_data['profit_percentage'], bins=50, alpha=0.7, color='#1f77b4')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax.set_xlabel('Profit Percentage (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Profit Percentages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Recent opportunities table
    st.subheader("🕐 Recent Opportunities")
    
    recent_data = arb_data.head(20)[['timestamp', 'path', 'profit_percentage', 'is_profitable', 'start_amount', 'end_amount']]
    recent_data['profit_percentage'] = recent_data['profit_percentage'].round(4)
    
    st.dataframe(recent_data, use_container_width=True)

def show_simple_pair_trading():
    """Show simple pair trading analysis."""
    st.header("💱 Simple Pair Trading Analysis")
    
    trades_data, portfolio_data, performance_data, price_history_data = load_simple_pair_data()
    
    if not performance_data:
        st.info("No simple pair trading data available. Run the simple pair monitor to collect data.")
        return
    
    # Performance summary
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        initial_balance = performance_data.get('initial_balance', 0)
        current_value = performance_data.get('current_value', 0)
        st.metric("Initial Balance", f"${initial_balance:.2f}")
        st.metric("Current Value", f"${current_value:.2f}")
    
    with col2:
        total_return_pct = performance_data.get('total_return_pct', 0)
        total_return_cad = performance_data.get('total_return_cad', 0)
        st.metric("Total Return", f"{total_return_pct:.2f}%", f"${total_return_cad:.2f}")
    
    with col3:
        total_trades = performance_data.get('total_trades', 0)
        total_fees = performance_data.get('total_fees_paid', 0)
        st.metric("Total Trades", total_trades)
        st.metric("Total Fees", f"${total_fees:.2f}")
    
    with col4:
        realized_pnl = performance_data.get('total_realized_pnl', 0)
        unrealized_pnl = performance_data.get('total_unrealized_pnl', 0)
        st.metric("Realized P&L", f"${realized_pnl:.2f}")
        st.metric("Unrealized P&L", f"${unrealized_pnl:.2f}")
    
    # Portfolio value over time
    if portfolio_data:
        st.subheader("📊 Portfolio Value Over Time")
        
        df = pd.DataFrame(portfolio_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timestamp'], df['portfolio_value'], linewidth=2, color='#1f77b4', label='Portfolio Value')
        ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        ax.set_title('Portfolio Performance')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (CAD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Current positions
    st.subheader("💼 Current Positions")
    
    positions = performance_data.get('positions', {})
    if positions:
        pos_data = []
        for pair, pos_info in positions.items():
            pos_data.append({
                'Pair': pair,
                'Amount': f"{pos_info.get('amount', 0):.6f}",
                'Avg Buy Price': f"${pos_info.get('avg_buy_price', 0):.2f}",
                'Unrealized P&L': f"${pos_info.get('unrealized_pnl', 0):.2f}",
                'Realized P&L': f"${pos_info.get('realized_pnl', 0):.2f}"
            })
        
        if pos_data:
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
        else:
            st.info("No current positions")
    
    # Trading history
    if trades_data:
        st.subheader("📋 Trading History")
        
        trades_df = pd.DataFrame(trades_data)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)
        
        # Format for display
        display_df = trades_df[['timestamp', 'pair', 'side', 'amount', 'price', 'cost', 'fee']].copy()
        display_df['amount'] = display_df['amount'].round(6)
        display_df['price'] = display_df['price'].round(2)
        display_df['cost'] = display_df['cost'].round(2)
        display_df['fee'] = display_df['fee'].round(4)
        
        st.dataframe(display_df, use_container_width=True)
    
    # Price charts
    if price_history_data:
        st.subheader("💹 Price History")
        
        for pair, price_data in price_history_data.items():
            if price_data:
                df = pd.DataFrame(price_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df['timestamp'], df['price'], linewidth=2, label=pair)
                
                # Add trade markers if available
                if trades_data:
                    pair_trades = [t for t in trades_data if t['pair'] == pair]
                    if pair_trades:
                        buy_trades = [t for t in pair_trades if t['side'] == 'buy']
                        sell_trades = [t for t in pair_trades if t['side'] == 'sell']
                        
                        if buy_trades:
                            buy_times = [pd.to_datetime(t['timestamp']) for t in buy_trades]
                            buy_prices = [t['price'] for t in buy_trades]
                            ax.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
                        
                        if sell_trades:
                            sell_times = [pd.to_datetime(t['timestamp']) for t in sell_trades]
                            sell_prices = [t['price'] for t in sell_trades]
                            ax.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell', zorder=5)
                
                ax.set_title(f'{pair} Price History')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price (CAD)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)

def show_live_monitoring():
    """Show live monitoring interface."""
    st.header("🔴 Live Monitoring")
    
    # Current market status
    st.subheader("📊 Current Market Status")
    
    # Add a loading message
    with st.spinner("Fetching current prices..."):
        current_prices = get_current_prices()
    
    if current_prices:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BTC/CAD", f"${current_prices.get('BTC/CAD', 0):,.2f}")
        
        with col2:
            st.metric("ETH/CAD", f"${current_prices.get('ETH/CAD', 0):,.2f}")
        
        with col3:
            st.metric("ETH/BTC", f"{current_prices.get('ETH/BTC', 0):.6f}")
        
        # Show last update time
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} (cached for 30 seconds)")
    else:
        st.warning("Unable to fetch current prices. Please check your connection.")
    
    # Active monitoring status
    st.subheader("🔄 Active Monitoring")
    
    # Check for active monitoring sessions
    data_dir = Path("data")
    monitor_dirs = list(data_dir.glob("simple_pair_monitor_*"))
    
    if monitor_dirs:
        latest_dir = max(monitor_dirs, key=lambda x: x.stat().st_mtime)
        last_modified = datetime.fromtimestamp(latest_dir.stat().st_mtime)
        time_diff = datetime.now() - last_modified
        
        if time_diff.total_seconds() < 300:  # Less than 5 minutes ago
            st.success(f"✅ Active monitoring session detected: {latest_dir.name}")
            st.info(f"Last activity: {time_diff.seconds} seconds ago")
        else:
            st.info(f"📁 Latest session: {latest_dir.name} (inactive)")
    else:
        st.info("No monitoring sessions found. Run a monitor to start collecting data.")
    
    # System status
    st.subheader("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configuration:**")
        st.write(f"- Start Amount: ${config.START_AMOUNT:.2f}")
        st.write(f"- Maker Fee: {config.MAKER_FEE*100:.2f}%")
        st.write(f"- Taker Fee: {config.TAKER_FEE*100:.2f}%")
    
    with col2:
        st.write("**Trading Pairs:**")
        st.write(f"- Pair 1: {config.TRADING_PAIRS.get('PAIR_1', 'BTC/CAD')}")
        st.write(f"- Pair 2: {config.TRADING_PAIRS.get('PAIR_2', 'ETH/CAD')}")
        st.write(f"- Pair 3: {config.TRADING_PAIRS.get('PAIR_3', 'ETH/BTC')}")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Run 5-min Test"):
            st.code("python run_simple_pair_monitor.py --test")
            st.info("Copy and run this command in your terminal")
    
    with col2:
        if st.button("📊 Run 1-hour Test"):
            st.code("python run_full_day_monitor.py --test")
            st.info("Copy and run this command in your terminal")
    
    with col3:
        if st.button("🔄 Full Day Monitor"):
            st.code("python run_full_day_monitor.py")
            st.info("Copy and run this command in your terminal")

def main():
    """Main dashboard function."""
    st.title("🚀 KrakenBot Dashboard")
    st.markdown("Real-time monitoring of triangular arbitrage and simple pair trading strategies")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Strategy selection
    strategy_mode = st.sidebar.selectbox(
        "Select Strategy View",
        ["Overview", "Triangular Arbitrage", "Simple Pair Trading", "Live Monitoring"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Current prices section
    st.sidebar.markdown("### Current Prices")
    current_prices = get_current_prices()
    
    for pair, price in current_prices.items():
        st.sidebar.metric(pair, f"${price:,.2f}")
    
    # Main content based on selected strategy
    try:
        if strategy_mode == "Overview":
            show_overview()
        elif strategy_mode == "Triangular Arbitrage":
            show_triangular_arbitrage()
        elif strategy_mode == "Simple Pair Trading":
            show_simple_pair_trading()
        elif strategy_mode == "Live Monitoring":
            show_live_monitoring()
    except Exception as e:
        st.error(f"Error loading {strategy_mode}: {e}")
        st.write("Please check the data files and try refreshing.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()