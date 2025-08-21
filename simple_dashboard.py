"""
Simple, reliable dashboard for monitoring trading sessions
No complex features that might cause startup issues
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="KrakenBot Simple Monitor",
    page_icon="📊",
    layout="wide"
)

def load_latest_session():
    """Load the most recent session data."""
    data_dir = Path("data")
    
    # Find the most recent session directory
    session_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "portfolio_history.json").exists():
            session_dirs.append(item)
    
    if not session_dirs:
        return None, "No sessions found"
    
    # Sort by modification time
    latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
    
    # Load data
    data = {}
    
    # Portfolio history
    portfolio_file = latest_session / "portfolio_history.json"
    if portfolio_file.exists():
        with open(portfolio_file, 'r') as f:
            data['portfolio'] = json.load(f)
    
    # Trades
    trades_file = latest_session / "trades.json"
    if trades_file.exists():
        with open(trades_file, 'r') as f:
            data['trades'] = json.load(f)
    else:
        data['trades'] = []
    
    return data, latest_session.name

def main():
    st.title("📊 KrakenBot Simple Monitor")
    
    # Load latest session
    data, session_name = load_latest_session()
    
    if data is None:
        st.error(session_name)  # Error message
        return
    
    st.success(f"Monitoring: {session_name}")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Basic metrics
    if 'portfolio' in data and data['portfolio']:
        latest = data['portfolio'][-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${latest['portfolio_value']:.2f}")
        
        with col2:
            st.metric("Total Return", f"{latest['total_return']:.4f}%")
        
        with col3:
            st.metric("Data Points", len(data['portfolio']))
        
        with col4:
            st.metric("Trades", len(data['trades']))
    
    # Recent trades
    if data['trades']:
        st.subheader("Recent Trades")
        
        # Show last 5 trades
        recent_trades = data['trades'][-5:]
        
        for i, trade in enumerate(reversed(recent_trades), 1):
            with st.expander(f"Trade {len(data['trades']) - len(recent_trades) + len(recent_trades) - i + 1}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Pair:** {trade['pair']}")
                    st.write(f"**Action:** {trade.get('side', trade.get('action', 'unknown')).upper()}")
                
                with col2:
                    st.write(f"**Price:** ${trade['price']:.2f}")
                    st.write(f"**Amount:** {trade['amount']:.6f}")
                
                with col3:
                    timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                    st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
                    if 'profit' in trade:
                        profit_color = "green" if trade['profit'] > 0 else "red"
                        st.markdown(f"**Profit:** <span style='color:{profit_color}'>${trade['profit']:.2f}</span>", unsafe_allow_html=True)
    else:
        st.info("No trades executed yet")
    
    # Portfolio chart (simple)
    if 'portfolio' in data and len(data['portfolio']) > 1:
        st.subheader("Portfolio Value Over Time")
        
        # Create simple dataframe
        df = pd.DataFrame(data['portfolio'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Simple line chart
        st.line_chart(df.set_index('timestamp')['portfolio_value'])
    
    # Current positions
    if 'portfolio' in data and data['portfolio']:
        latest = data['portfolio'][-1]
        positions = latest.get('positions', {})
        
        active_positions = {k: v for k, v in positions.items() if v.get('base_amount', 0) > 0}
        
        if active_positions:
            st.subheader("Current Positions")
            
            for pair, pos in active_positions.items():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**{pair}**")
                
                with col2:
                    st.write(f"Amount: {pos['base_amount']:.6f}")
                
                with col3:
                    unrealized = pos.get('unrealized_pnl', 0)
                    color = "green" if unrealized > 0 else "red"
                    st.markdown(f"P&L: <span style='color:{color}'>${unrealized:.2f}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()