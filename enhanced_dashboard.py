"""
Enhanced Dashboard for KrakenBot with proper session data handling
Fixes time span issues and synchronizes data across all tabs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="KrakenBot Enhanced Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_session_data(session_name):
    """Load comprehensive data from a specific session."""
    session_dir = Path("data") / session_name
    
    data = {
        'portfolio_history': [],
        'trades': [],
        'price_history': [],
        'performance_summary': {},
        'session_exists': False
    }
    
    if not session_dir.exists():
        return data
    
    data['session_exists'] = True
    
    # Load portfolio history
    portfolio_file = session_dir / "portfolio_history.json"
    if portfolio_file.exists():
        try:
            with open(portfolio_file, 'r') as f:
                data['portfolio_history'] = json.load(f)
        except Exception as e:
            st.error(f"Error loading portfolio history: {e}")
    
    # Load trades
    trades_file = session_dir / "trades.json"
    if trades_file.exists():
        try:
            with open(trades_file, 'r') as f:
                data['trades'] = json.load(f)
        except Exception as e:
            st.error(f"Error loading trades: {e}")
    
    # Load price history
    price_file = session_dir / "price_history.json"
    if price_file.exists():
        try:
            with open(price_file, 'r') as f:
                data['price_history'] = json.load(f)
        except Exception as e:
            st.error(f"Error loading price history: {e}")
    
    # Load performance summary
    perf_file = session_dir / "performance_summary.json"
    if perf_file.exists():
        try:
            with open(perf_file, 'r') as f:
                data['performance_summary'] = json.load(f)
        except Exception as e:
            st.error(f"Error loading performance summary: {e}")
    
    return data

def get_available_sessions():
    """Get list of available trading sessions."""
    data_dir = Path("data")
    sessions = []
    
    try:
        # Look for different session types
        patterns = [
            "live_session_*",
            "optimized_session_*", 
            "aggressive_test_*",
            "simple_pair_monitor_*",
            "extended_validation_*"
        ]
        
        for pattern in patterns:
            try:
                for session_dir in data_dir.glob(pattern):
                    if session_dir.is_dir():
                        # Check if it has data files
                        has_data = any([
                            (session_dir / "portfolio_history.json").exists(),
                            (session_dir / "trades.json").exists(),
                            (session_dir / "price_history.json").exists()
                        ])
                        
                        if has_data:
                            sessions.append(session_dir.name)
            except Exception as e:
                continue  # Skip problematic patterns
        
        # Sort by modification time (newest first)
        if sessions:
            sessions.sort(key=lambda x: (data_dir / x).stat().st_mtime, reverse=True)
        
        # Always return at least one session if any exist
        if not sessions:
            # Fallback - just get any directory with data
            for item in data_dir.iterdir():
                if item.is_dir() and any([
                    (item / "portfolio_history.json").exists(),
                    (item / "trades.json").exists()
                ]):
                    sessions.append(item.name)
                    break
    
    except Exception as e:
        # Return empty list if there's an error
        sessions = []
    
    return sessions

def create_portfolio_chart(portfolio_data, trades_data):
    """Create enhanced portfolio performance chart."""
    if not portfolio_data:
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame(portfolio_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value & Returns', 'Price Movement'),
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Portfolio Value</b><br>Time: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add trade markers
    if trades_data:
        trade_times = []
        trade_values = []
        trade_colors = []
        trade_text = []
        
        for trade in trades_data:
            trade_time = pd.to_datetime(trade['timestamp'])
            # Find closest portfolio value
            closest_idx = df.iloc[(df['timestamp'] - trade_time).abs().argsort()[:1]].index[0]
            portfolio_val = df.iloc[closest_idx]['portfolio_value']
            
            trade_times.append(trade_time)
            trade_values.append(portfolio_val)
            
            if trade.get('side') == 'buy' or trade.get('action') == 'buy':
                trade_colors.append('green')
                trade_text.append(f"BUY {trade['pair']}<br>${trade['price']:.2f}")
            else:
                trade_colors.append('red')
                profit = trade.get('profit', 0)
                trade_text.append(f"SELL {trade['pair']}<br>${trade['price']:.2f}<br>P&L: ${profit:.2f}")
        
        if trade_times:
            fig.add_trace(
                go.Scatter(
                    x=trade_times,
                    y=trade_values,
                    mode='markers',
                    name='Trades',
                    marker=dict(
                        color=trade_colors,
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    text=trade_text,
                    hovertemplate='<b>Trade</b><br>%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add price data if available
    if 'prices' in df.columns and len(df) > 0:
        # Extract ETH and BTC prices
        eth_prices = []
        btc_prices = []
        
        for _, row in df.iterrows():
            prices = row.get('prices', {})
            eth_prices.append(prices.get('ETH/CAD', None))
            btc_prices.append(prices.get('BTC/CAD', None))
        
        # Add ETH price
        if any(p is not None for p in eth_prices):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=eth_prices,
                    mode='lines',
                    name='ETH/CAD',
                    line=dict(color='purple', width=1),
                    yaxis='y3'
                ),
                row=2, col=1
            )
        
        # Add BTC price (scaled down)
        if any(p is not None for p in btc_prices):
            btc_scaled = [p/25 if p else None for p in btc_prices]  # Scale down for visibility
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=btc_scaled,
                    mode='lines',
                    name='BTC/CAD (÷25)',
                    line=dict(color='orange', width=1),
                    yaxis='y3'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title="Enhanced Portfolio Performance",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig

def create_trades_analysis(trades_data):
    """Create comprehensive trades analysis."""
    if not trades_data:
        return None, None
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades_data)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Separate buy and sell trades
    buy_trades = trades_df[trades_df.get('side', trades_df.get('action', '')) == 'buy']
    sell_trades = trades_df[trades_df.get('side', trades_df.get('action', '')) == 'sell']
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    
    with col2:
        st.metric("Buy Orders", len(buy_trades))
    
    with col3:
        st.metric("Sell Orders", len(sell_trades))
    
    with col4:
        if len(sell_trades) > 0:
            total_profit = sell_trades.get('profit', pd.Series([0])).sum()
            st.metric("Total P&L", f"${total_profit:.2f}")
        else:
            st.metric("Total P&L", "$0.00")
    
    # Create trades timeline
    if len(trades_df) > 0:
        fig = go.Figure()
        
        for _, trade in trades_df.iterrows():
            color = 'green' if trade.get('side', trade.get('action')) == 'buy' else 'red'
            symbol = 'triangle-up' if trade.get('side', trade.get('action')) == 'buy' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']],
                    y=[trade['price']],
                    mode='markers',
                    name=f"{trade.get('side', trade.get('action')).upper()} {trade['pair']}",
                    marker=dict(
                        color=color,
                        size=15,
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>{trade.get('side', trade.get('action')).upper()}</b><br>" +
                                f"Pair: {trade['pair']}<br>" +
                                f"Price: ${trade['price']:.2f}<br>" +
                                f"Amount: {trade['amount']:.6f}<br>" +
                                f"Time: %{{x}}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title="Trades Timeline",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400
        )
        
        return fig, trades_df
    
    return None, trades_df

def main():
    """Main dashboard function."""
    st.title("🤖 KrakenBot Enhanced Dashboard")
    st.markdown("Real-time monitoring with proper session data synchronization")
    
    # Sidebar for session selection
    st.sidebar.header("📊 Session Selection")
    
    available_sessions = get_available_sessions()
    
    if not available_sessions:
        st.error("No trading sessions found. Run a trading session first.")
        return
    
    selected_session = st.sidebar.selectbox(
        "Select Trading Session",
        available_sessions,
        help="Choose a session to analyze"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        # Auto refresh every 30 seconds
        st.rerun()
    
    # Load session data
    session_data = load_session_data(selected_session)
    
    if not session_data['session_exists']:
        st.error(f"Session '{selected_session}' not found or has no data.")
        return
    
    # Display session info
    st.sidebar.markdown(f"**Current Session:** {selected_session}")
    
    if session_data['portfolio_history']:
        latest_data = session_data['portfolio_history'][-1]
        st.sidebar.metric("Portfolio Value", f"${latest_data['portfolio_value']:.2f}")
        st.sidebar.metric("Total Return", f"{latest_data['total_return']:.4f}%")
        st.sidebar.metric("Data Points", len(session_data['portfolio_history']))
        st.sidebar.metric("Trades Executed", len(session_data['trades']))
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio", "🔄 Trades", "📊 Analysis", "⚙️ Settings"])
    
    with tab1:
        st.header("Portfolio Performance")
        
        if session_data['portfolio_history']:
            # Create enhanced portfolio chart
            portfolio_fig = create_portfolio_chart(
                session_data['portfolio_history'], 
                session_data['trades']
            )
            st.plotly_chart(portfolio_fig, use_container_width=True)
            
            # Performance metrics
            latest = session_data['portfolio_history'][-1]
            first = session_data['portfolio_history'][0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Value",
                    f"${latest['portfolio_value']:.2f}",
                    f"{latest['total_return']:+.4f}%"
                )
            
            with col2:
                duration_hours = len(session_data['portfolio_history']) / 60  # Assuming 1-minute intervals
                st.metric("Session Duration", f"{duration_hours:.1f}h")
            
            with col3:
                if session_data['trades']:
                    trades_per_hour = len(session_data['trades']) / duration_hours if duration_hours > 0 else 0
                    st.metric("Trade Frequency", f"{trades_per_hour:.2f}/h")
                else:
                    st.metric("Trade Frequency", "0.00/h")
            
            with col4:
                if 'performance_summary' in session_data and session_data['performance_summary']:
                    unrealized_pnl = session_data['performance_summary'].get('total_unrealized_pnl', 0)
                    st.metric("Unrealized P&L", f"${unrealized_pnl:.2f}")
                else:
                    st.metric("Unrealized P&L", "$0.00")
        else:
            st.info("No portfolio data available for this session.")
    
    with tab2:
        st.header("Trading Activity")
        
        if session_data['trades']:
            trades_fig, trades_df = create_trades_analysis(session_data['trades'])
            
            if trades_fig:
                st.plotly_chart(trades_fig, use_container_width=True)
            
            # Detailed trades table
            st.subheader("Trade Details")
            
            # Format trades for display
            display_trades = []
            for trade in session_data['trades']:
                display_trade = {
                    'Time': pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S'),
                    'Pair': trade['pair'],
                    'Action': trade.get('side', trade.get('action', 'unknown')).upper(),
                    'Price': f"${trade['price']:.2f}",
                    'Amount': f"{trade['amount']:.6f}",
                    'Value': f"${trade.get('cost', trade.get('value', 0)):.2f}",
                    'Fee': f"${trade.get('fee', 0):.2f}"
                }
                
                if 'profit' in trade:
                    display_trade['Profit'] = f"${trade['profit']:.2f}"
                    display_trade['Profit %'] = f"{trade.get('profit_pct', 0):+.3f}%"
                
                display_trades.append(display_trade)
            
            st.dataframe(pd.DataFrame(display_trades), use_container_width=True)
        else:
            st.info("No trades executed in this session yet.")
            
            # Show why no trades might be happening
            if session_data['portfolio_history']:
                st.subheader("Market Conditions")
                latest = session_data['portfolio_history'][-1]
                
                if 'prices' in latest:
                    prices = latest['prices']
                    st.write("**Current Prices:**")
                    for pair, price in prices.items():
                        st.write(f"• {pair}: ${price:.2f}")
                
                st.info("""
                **No trades executed yet. This could be because:**
                - Market conditions haven't met the trading thresholds
                - Thresholds may be too conservative
                - Insufficient price movement in the lookback period
                
                **Current Aggressive Settings:**
                - Buy Threshold: -0.3% (triggers on 0.3% price drop)
                - Sell Threshold: +0.5% (triggers on 0.5% price gain)
                - Lookback Periods: 5 (very responsive)
                """)
    
    with tab3:
        st.header("Performance Analysis")
        
        if session_data['portfolio_history'] and len(session_data['portfolio_history']) > 1:
            # Calculate performance metrics
            df = pd.DataFrame(session_data['portfolio_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Returns analysis
            returns = df['total_return'].diff().dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Return Distribution")
                fig_hist = px.histogram(
                    returns, 
                    title="Return Distribution",
                    labels={'value': 'Return (%)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Performance Metrics")
                
                # Calculate metrics
                total_return = df['total_return'].iloc[-1]
                volatility = returns.std() if len(returns) > 1 else 0
                max_drawdown = 0
                
                # Calculate max drawdown
                portfolio_values = df['portfolio_value']
                peak = portfolio_values.iloc[0]
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
                
                st.metric("Total Return", f"{total_return:.4f}%")
                st.metric("Volatility", f"{volatility:.4f}%")
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                if session_data['trades']:
                    sell_trades = [t for t in session_data['trades'] if t.get('side') == 'sell' or t.get('action') == 'sell']
                    if sell_trades:
                        profits = [t.get('profit', 0) for t in sell_trades]
                        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                        avg_profit = sum(profits) / len(profits)
                        
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                        st.metric("Avg Profit/Trade", f"${avg_profit:.2f}")
        else:
            st.info("Insufficient data for performance analysis.")
    
    with tab4:
        st.header("Session Settings & Info")
        
        # Display current configuration
        st.subheader("Current Trading Parameters")
        
        try:
            import config
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Thresholds:**")
                st.write(f"• Buy Threshold: {config.BUY_THRESHOLD*100:.2f}%")
                st.write(f"• Sell Threshold: {config.SELL_THRESHOLD*100:.2f}%")
                st.write(f"• Lookback Periods: {config.LOOKBACK_PERIODS}")
                
                st.write("**Position Sizing:**")
                st.write(f"• Min Trade Amount: ${config.MIN_TRADE_AMOUNT:.2f}")
                st.write(f"• Max Position Size: {config.MAX_POSITION_SIZE*100:.0f}%")
            
            with col2:
                st.write("**Risk Management:**")
                if hasattr(config, 'STOP_LOSS_PCT'):
                    st.write(f"• Stop Loss: {config.STOP_LOSS_PCT*100:.2f}%")
                if hasattr(config, 'TAKE_PROFIT_PCT'):
                    st.write(f"• Take Profit: {config.TAKE_PROFIT_PCT*100:.2f}%")
                if hasattr(config, 'MAX_HOLD_TIME'):
                    st.write(f"• Max Hold Time: {config.MAX_HOLD_TIME}s")
                
                st.write("**Fees:**")
                st.write(f"• Maker Fee: {config.MAKER_FEE*100:.2f}%")
                st.write(f"• Taker Fee: {config.TAKER_FEE*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
        
        # Session information
        st.subheader("Session Information")
        st.write(f"**Selected Session:** {selected_session}")
        
        if session_data['performance_summary']:
            st.json(session_data['performance_summary'])

if __name__ == "__main__":
    main()