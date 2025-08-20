"""
Enhanced dashboard for KrakenBot with both triangular arbitrage and simple pair trading.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

import config
from exchange import ExchangeManager
from arbitrage import ArbitrageCalculator
from simple_pair_trader import SimplePairTrader

# Set page configuration
st.set_page_config(
    page_title="KrakenBot Enhanced Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit-positive {
        color: #28a745;
    }
    .profit-negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_arbitrage_data():
    """Load triangular arbitrage data."""
    opportunities_file = config.DATA_DIR / "opportunities.csv"
    
    if opportunities_file.exists():
        df = pd.read_csv(opportunities_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp', ascending=False)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_simple_pair_data():
    """Load simple pair trading data from the most recent session."""
    data_dir = Path("data")
    
    # Find the most recent simple pair monitor directory
    monitor_dirs = list(data_dir.glob("simple_pair_monitor_*"))
    if not monitor_dirs:
        return {}, {}, {}, {}
    
    latest_dir = max(monitor_dirs, key=lambda x: x.stat().st_mtime)
    
    # Load data files
    trades_file = latest_dir / "trades.json"
    portfolio_file = latest_dir / "portfolio_history.json"
    performance_file = latest_dir / "performance_summary.json"
    price_history_file = latest_dir / "price_history.json"
    
    trades_data = {}
    portfolio_data = {}
    performance_data = {}
    price_history_data = {}
    
    try:
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
    
    except Exception as e:
        st.error(f"Error loading simple pair data: {e}")
    
    return trades_data, portfolio_data, performance_data, price_history_data

def get_current_prices():
    """Get current market prices."""
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

def main():
    """Main dashboard function."""
    st.title("🚀 KrakenBot Enhanced Dashboard")
    st.markdown("Real-time monitoring of triangular arbitrage and simple pair trading strategies")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Strategy selection
    strategy_mode = st.sidebar.selectbox(
        "Select Strategy View",
        ["Overview", "Triangular Arbitrage", "Simple Pair Trading", "Live Monitoring"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Current prices section
    st.sidebar.markdown("### Current Prices")
    current_prices = get_current_prices()
    
    for pair, price in current_prices.items():
        st.sidebar.metric(pair, f"${price:,.2f}")
    
    # Main content based on selected strategy
    if strategy_mode == "Overview":
        show_overview()
    elif strategy_mode == "Triangular Arbitrage":
        show_triangular_arbitrage()
    elif strategy_mode == "Simple Pair Trading":
        show_simple_pair_trading()
    elif strategy_mode == "Live Monitoring":
        show_live_monitoring()

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
    
    # Combined performance chart
    if portfolio_data:
        st.subheader("📈 Portfolio Performance Over Time")
        
        df = pd.DataFrame(portfolio_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="Value (CAD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

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
    
    fig = px.histogram(
        arb_data, 
        x='profit_percentage', 
        nbins=50,
        title="Distribution of Profit Percentages",
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.update_layout(xaxis_title="Profit Percentage (%)", yaxis_title="Frequency")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("⏰ Opportunities Over Time")
    
    # Resample data by hour
    arb_data_hourly = arb_data.set_index('timestamp').resample('1H').agg({
        'profit_percentage': ['count', 'mean', 'max'],
        'is_profitable': 'sum'
    }).reset_index()
    
    arb_data_hourly.columns = ['timestamp', 'count', 'avg_profit', 'max_profit', 'profitable_count']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Opportunities per Hour', 'Average Profit per Hour'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=arb_data_hourly['timestamp'], y=arb_data_hourly['count'], 
                  mode='lines+markers', name='Opportunities'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=arb_data_hourly['timestamp'], y=arb_data_hourly['avg_profit'], 
                  mode='lines+markers', name='Avg Profit %'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
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
        
        fig = go.Figure()
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Initial balance line
        fig.add_hline(
            y=initial_balance,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Balance"
        )
        
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Time",
            yaxis_title="Value (CAD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Current positions
    st.subheader("💼 Current Positions")
    
    positions = performance_data.get('positions', {})
    if positions:
        pos_data = []
        for pair, pos_info in positions.items():
            if pos_info['amount'] > 0:
                pos_data.append({
                    'Pair': pair,
                    'Amount': f"{pos_info['amount']:.6f}",
                    'Avg Buy Price': f"${pos_info['avg_buy_price']:.2f}",
                    'Unrealized P&L': f"${pos_info['unrealized_pnl']:.2f}",
                    'Realized P&L': f"${pos_info['realized_pnl']:.2f}"
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
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    mode='lines',
                    name=pair,
                    line=dict(width=2)
                ))
                
                # Add trade markers if available
                if trades_data:
                    pair_trades = [t for t in trades_data if t['pair'] == pair]
                    if pair_trades:
                        trade_times = [pd.to_datetime(t['timestamp']) for t in pair_trades]
                        trade_prices = [t['price'] for t in pair_trades]
                        trade_sides = [t['side'] for t in pair_trades]
                        
                        buy_times = [t for t, s in zip(trade_times, trade_sides) if s == 'buy']
                        buy_prices = [p for p, s in zip(trade_prices, trade_sides) if s == 'buy']
                        sell_times = [t for t, s in zip(trade_times, trade_sides) if s == 'sell']
                        sell_prices = [p for p, s in zip(trade_prices, trade_sides) if s == 'sell']
                        
                        if buy_times:
                            fig.add_trace(go.Scatter(
                                x=buy_times, y=buy_prices,
                                mode='markers',
                                name='Buy',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        
                        if sell_times:
                            fig.add_trace(go.Scatter(
                                x=sell_times, y=sell_prices,
                                mode='markers',
                                name='Sell',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                
                fig.update_layout(
                    title=f"{pair} Price History",
                    xaxis_title="Time",
                    yaxis_title="Price (CAD)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_live_monitoring():
    """Show live monitoring interface."""
    st.header("🔴 Live Monitoring")
    
    # Current market status
    st.subheader("📊 Current Market Status")
    
    current_prices = get_current_prices()
    
    if current_prices:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BTC/CAD", f"${current_prices.get('BTC/CAD', 0):,.2f}")
        
        with col2:
            st.metric("ETH/CAD", f"${current_prices.get('ETH/CAD', 0):,.2f}")
        
        with col3:
            st.metric("ETH/BTC", f"{current_prices.get('ETH/BTC', 0):.6f}")
    
    # Live arbitrage check
    st.subheader("🔍 Live Arbitrage Check")
    
    if st.button("Check Current Opportunities"):
        with st.spinner("Checking arbitrage opportunities..."):
            try:
                exchange = ExchangeManager(use_api_keys=False)
                calculator = ArbitrageCalculator(exchange)
                
                opportunities = calculator.find_opportunities()
                
                if opportunities:
                    for i, opp in enumerate(opportunities, 1):
                        status = "✅ PROFITABLE" if opp.is_profitable else "❌ Not Profitable"
                        
                        with st.expander(f"Opportunity {i}: {status}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Path:** {' → '.join(opp.path)}")
                                st.write(f"**Profit:** {opp.profit_percentage:.4f}%")
                                st.write(f"**Start Amount:** ${opp.start_amount:.2f}")
                                st.write(f"**End Amount:** ${opp.end_amount:.2f}")
                            
                            with col2:
                                st.write("**Prices Used:**")
                                for pair, price in opp.prices.items():
                                    st.write(f"  {pair}: ${price:.2f}")
                else:
                    st.info("No opportunities found")
                    
            except Exception as e:
                st.error(f"Error checking opportunities: {e}")
    
    # System status
    st.subheader("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configuration:**")
        st.write(f"- Start Amount: ${config.START_AMOUNT:.2f}")
        st.write(f"- Min Profit: {config.MIN_PROFIT_PCT:.2f}%")
        st.write(f"- Maker Fee: {config.MAKER_FEE*100:.2f}%")
        st.write(f"- Taker Fee: {config.TAKER_FEE*100:.2f}%")
    
    with col2:
        st.write("**Trading Pairs:**")
        for pair_name, pair_value in config.TRADING_PAIRS.items():
            st.write(f"- {pair_name}: {pair_value}")

if __name__ == "__main__":
    main()