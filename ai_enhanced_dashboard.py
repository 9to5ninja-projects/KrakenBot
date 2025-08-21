"""
AI-Enhanced Dashboard for KrakenBot
Integrates technical analysis, AI predictions, multi-coin analysis, and NLP insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional

# Import our enhanced modules
from advanced_technical_indicators import AdvancedTechnicalAnalyzer
from ai_strategy_optimizer import AIStrategyOptimizer
from multi_coin_analyzer import MultiCoinAnalyzer
from enhanced_nlp_analyzer import EnhancedNLPAnalyzer

st.set_page_config(
    page_title="KrakenBot AI Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #2ca02c;
    }
    .warning-metric {
        border-left-color: #ff7f0e;
    }
    .danger-metric {
        border-left-color: #d62728;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_session_data(session_name: str) -> Dict:
    """Load session data with caching."""
    data_dir = Path("data") / session_name
    
    if not data_dir.exists():
        return {}
    
    data = {}
    
    # Load portfolio history
    portfolio_file = data_dir / "portfolio_history.json"
    if portfolio_file.exists():
        with open(portfolio_file, 'r') as f:
            data['portfolio_history'] = json.load(f)
    
    # Load trades
    trades_file = data_dir / "trades.json"
    if trades_file.exists():
        with open(trades_file, 'r') as f:
            data['trades'] = json.load(f)
    
    # Load opportunities
    opportunities_file = data_dir / "opportunities.json"
    if opportunities_file.exists():
        with open(opportunities_file, 'r') as f:
            data['opportunities'] = json.load(f)
    
    # Load final analysis if available
    analysis_file = data_dir / "final_analysis.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            data['final_analysis'] = json.load(f)
    
    return data

def get_available_sessions() -> List[str]:
    """Get list of available trading sessions."""
    data_dir = Path("data")
    sessions = []
    
    if not data_dir.exists():
        return sessions
    
    # Look for all sessions with data files
    for session_dir in data_dir.iterdir():
        if session_dir.is_dir():
            # Check if it has enhanced simulation data files
            has_data = any([
                (session_dir / "portfolio_history.json").exists(),
                (session_dir / "trades.json").exists(),
                (session_dir / "opportunities.json").exists()
            ])
            
            if has_data:
                sessions.append(session_dir.name)

    
    # Sort by modification time (newest first)
    if sessions:
        sessions.sort(key=lambda x: (data_dir / x).stat().st_mtime, reverse=True)
    
    return sessions

def create_enhanced_portfolio_chart(portfolio_data: List[Dict]) -> go.Figure:
    """Create enhanced portfolio chart with technical indicators."""
    if not portfolio_data:
        return go.Figure()
    
    df = pd.DataFrame(portfolio_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure required columns exist
    required_columns = ['portfolio_value', 'total_return', 'cash_balance']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Portfolio Value', 'Return %', 'Cash Balance'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Portfolio value with moving average
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add moving average if enough data
    if len(df) > 5:
        df['ma_5'] = df['portfolio_value'].rolling(window=5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['ma_5'],
                mode='lines',
                name='5-Period MA',
                line=dict(color='orange', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Return percentage
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['total_return'],
            mode='lines+markers',
            name='Total Return %',
            line=dict(color='green' if df['total_return'].iloc[-1] > 0 else 'red', width=2),
            marker=dict(size=4),
            fill='tonexty' if df['total_return'].iloc[-1] > 0 else None
        ),
        row=2, col=1
    )
    
    # Add zero line for returns
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Cash balance
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['cash_balance'],
            mode='lines+markers',
            name='Cash Balance',
            line=dict(color='purple', width=2),
            marker=dict(size=4)
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Enhanced Portfolio Performance",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Cash ($)", row=3, col=1)
    
    return fig

def create_technical_indicators_chart(session_data: Dict) -> go.Figure:
    """Create technical indicators visualization."""
    # This would integrate with real-time technical analysis
    # For now, create a placeholder chart
    
    fig = go.Figure()
    
    # Add placeholder data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='1H')
    
    # Simulated RSI
    rsi_values = np.random.uniform(30, 70, len(dates))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=rsi_values,
        mode='lines',
        name='RSI',
        line=dict(color='blue')
    ))
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
    
    fig.update_layout(
        title="Technical Indicators",
        yaxis_title="RSI",
        height=300
    )
    
    return fig

def create_ai_predictions_chart(session_data: Dict) -> go.Figure:
    """Create AI predictions visualization."""
    portfolio_data = session_data.get('portfolio_history', [])
    
    if not portfolio_data:
        return go.Figure()
    
    # Extract AI insights from portfolio data
    timestamps = []
    ai_scores = []
    confidence_scores = []
    
    for entry in portfolio_data:
        if 'insights' in entry:
            timestamps.append(pd.to_datetime(entry['timestamp']))
            # Extract AI-related insights (simplified)
            ai_scores.append(np.random.uniform(40, 80))  # Placeholder
            confidence_scores.append(np.random.uniform(60, 95))  # Placeholder
    
    if not timestamps:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('AI Signal Strength', 'Prediction Confidence'),
        vertical_spacing=0.15
    )
    
    # AI Signal Strength
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=ai_scores,
            mode='lines+markers',
            name='AI Signal',
            line=dict(color='purple', width=2)
        ),
        row=1, col=1
    )
    
    # Confidence
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=confidence_scores,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="AI Predictions & Confidence",
        height=400
    )
    
    return fig

def display_trading_opportunities(session_data: Dict):
    """Display current trading opportunities."""
    opportunities = session_data.get('opportunities', [])
    
    if not opportunities:
        st.info("No trading opportunities identified in current session")
        return
    
    st.subheader("TARGET: Trading Opportunities")
    
    for i, opp in enumerate(opportunities[-5:], 1):  # Show last 5
        with st.expander(f"Opportunity {i}: {opp.get('pair', 'Unknown')} - {opp.get('signal', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entry Price", f"${opp.get('entry_price', 0):.2f}")
                st.metric("Target Price", f"${opp.get('target_price', 0):.2f}")
            
            with col2:
                st.metric("Expected Profit", f"{opp.get('expected_profit_pct', 0)*100:.2f}%")
                st.metric("Risk/Reward", f"{opp.get('risk_reward_ratio', 0):.2f}")
            
            with col3:
                confidence = opp.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1f}%")
                urgency = opp.get('urgency', 'UNKNOWN')
                st.metric("Urgency", urgency)
            
            if 'reasoning' in opp:
                st.write(f"**Reasoning:** {opp['reasoning']}")

def display_nlp_insights(session_data: Dict):
    """Display NLP-generated insights."""
    final_analysis = session_data.get('final_analysis', {})
    
    if not final_analysis:
        st.info("NLP analysis not available for this session")
        return
    
    st.subheader("AI: AI-Generated Insights")
    
    # Session summary
    if 'session_summary' in final_analysis:
        st.markdown(f"""
        <div class="insight-box">
        <h4>SUMMARY: Session Summary</h4>
        <p>{final_analysis['session_summary']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    if 'key_insights' in final_analysis:
        st.markdown("**INSIGHTS: Key Insights:**")
        for insight in final_analysis['key_insights']:
            st.write(f"• {insight}")
    
    # Strategic recommendations
    if 'strategic_recommendations' in final_analysis:
        st.markdown("**TARGET: Strategic Recommendations:**")
        for rec in final_analysis['strategic_recommendations']:
            st.write(f"• {rec}")
    
    # Risk assessment
    if 'risk_assessment' in final_analysis:
        risk_data = final_analysis['risk_assessment']
        risk_level = risk_data.get('overall_risk_level', 'UNKNOWN')
        
        risk_color = {
            'LOW': 'success-metric',
            'MEDIUM': 'warning-metric',
            'HIGH': 'danger-metric',
            'CRITICAL': 'danger-metric'
        }.get(risk_level, 'metric-card')
        
        st.markdown(f"""
        <div class="metric-card {risk_color}">
        <h4>RISK: Risk Assessment: {risk_level}</h4>
        <p>{risk_data.get('detailed_analysis', 'No detailed analysis available')}</p>
        </div>
        """, unsafe_allow_html=True)

def display_current_positions(session_data: Dict):
    """Display current trading positions."""
    portfolio_data = session_data.get('portfolio_history', [])
    
    if not portfolio_data:
        st.info("No position data available")
        return
    
    # Get latest portfolio state
    latest_data = portfolio_data[-1]
    positions = latest_data.get('positions', {})
    
    if not positions:
        st.info("No open positions")
        return
    
    # Display positions
    for pair, position in positions.items():
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pair", pair)
            
            with col2:
                amount = position.get('amount', 0)
                st.metric("Amount", f"{amount:.6f}")
            
            with col3:
                value = position.get('value', 0)
                st.metric("Value", f"${value:.2f}")

def display_trade_history(session_data: Dict):
    """Display recent trade history."""
    trades_data = session_data.get('trades', [])
    
    if not trades_data:
        st.info("No trades executed yet")
        return
    
    # Show last 10 trades
    recent_trades = trades_data[-10:]
    
    for trade in reversed(recent_trades):  # Most recent first
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pair = trade.get('pair', 'Unknown')
                action = trade.get('action', trade.get('side', 'Unknown'))
                st.write(f"**{pair}**")
                st.write(f"{action.upper()}")
            
            with col2:
                amount = trade.get('amount', 0)
                price = trade.get('price', 0)
                st.write(f"Amount: {amount:.6f}")
                st.write(f"Price: ${price:.2f}")
            
            with col3:
                profit = trade.get('profit', 0)
                timestamp = trade.get('timestamp', '')
                if timestamp:
                    time_str = pd.to_datetime(timestamp).strftime('%H:%M:%S')
                    st.write(f"Time: {time_str}")
                
                if profit != 0:
                    color = "green" if profit > 0 else "red"
                    st.markdown(f"<span style='color: {color}'>P&L: ${profit:.2f}</span>", 
                              unsafe_allow_html=True)
            
            st.divider()

def display_ai_insights_compact(session_data: Dict):
    """Display compact AI insights."""
    portfolio_data = session_data.get('portfolio_history', [])
    
    if not portfolio_data:
        st.info("No AI insights available")
        return
    
    # Get latest insights
    latest_data = portfolio_data[-1]
    insights = latest_data.get('insights', [])
    
    if insights:
        for insight in insights[-3:]:  # Show last 3 insights
            st.write(f"• {insight}")
    else:
        st.info("No AI insights generated yet")
    
    # Show market volatility if available
    if len(portfolio_data) > 1:
        volatility_data = []
        for data in portfolio_data[-5:]:  # Last 5 data points
            if 'portfolio_value' in data:
                volatility_data.append(data['portfolio_value'])
        
        if len(volatility_data) > 1:
            volatility = np.std(volatility_data) / np.mean(volatility_data)
            st.metric("Market Volatility", f"{volatility:.3f}")

def create_technical_indicators_chart_for_pair(session_data: Dict, selected_pair: str) -> go.Figure:
    """Create technical indicators chart for selected pair."""
    fig = go.Figure()
    
    # Generate realistic price data based on the pair
    base_prices = {
        "ETH/CAD": 4200,
        "BTC/CAD": 85000,
        "SOL/CAD": 180,
        "XRP/CAD": 0.85
    }
    
    base_price = base_prices.get(selected_pair, 1000)
    
    # Create timestamps for last 24 hours
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                              end=datetime.now(), freq='1H')
    
    # Generate realistic price movement
    np.random.seed(42)  # For consistent data
    price_changes = np.random.normal(0, 0.02, len(timestamps))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Price line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines',
        name=f'{selected_pair} Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Moving averages
    if len(prices) >= 5:
        ma_5 = pd.Series(prices).rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ma_5,
            mode='lines',
            name='MA(5)',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if len(prices) >= 12:
        ma_12 = pd.Series(prices).rolling(window=12, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ma_12,
            mode='lines',
            name='MA(12)',
            line=dict(color='red', width=1, dash='dot')
        ))
    
    # Add some volume bars at the bottom
    volumes = np.random.uniform(1000, 5000, len(timestamps))
    
    fig.add_trace(go.Bar(
        x=timestamps,
        y=volumes,
        name='Volume',
        yaxis='y2',
        opacity=0.3,
        marker_color='gray'
    ))
    
    # Update layout with dual y-axis
    fig.update_layout(
        title=f"Technical Analysis: {selected_pair}",
        xaxis_title="Time",
        yaxis=dict(title="Price ($)", side='left'),
        yaxis2=dict(title="Volume", side='right', overlaying='y', showgrid=False),
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def display_performance_metrics(session_data: Dict):
    """Display enhanced performance metrics."""
    portfolio_data = session_data.get('portfolio_history', [])
    trades_data = session_data.get('trades', [])
    
    if not portfolio_data:
        st.warning("No portfolio data available")
        return
    
    # Calculate metrics
    initial_value = portfolio_data[0]['portfolio_value']
    current_value = portfolio_data[-1]['portfolio_value']
    total_return = ((current_value - initial_value) / initial_value) * 100
    
    # Trade metrics
    total_trades = len(trades_data)
    profitable_trades = sum(1 for trade in trades_data if trade.get('profit', 0) > 0)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${current_value:,.2f}",
            f"{total_return:+.3f}%"
        )
    
    with col2:
        st.metric(
            "Total Trades",
            total_trades,
            f"{win_rate:.1f}% win rate"
        )
    
    with col3:
        if portfolio_data:
            duration = pd.to_datetime(portfolio_data[-1]['timestamp']) - pd.to_datetime(portfolio_data[0]['timestamp'])
            hours = duration.total_seconds() / 3600
            st.metric(
                "Session Duration",
                f"{hours:.1f}h",
                f"{len(portfolio_data)} data points"
            )
    
    with col4:
        # Calculate Sharpe ratio (simplified)
        if len(portfolio_data) > 1:
            returns = []
            for i in range(1, len(portfolio_data)):
                ret = (portfolio_data[i]['portfolio_value'] - portfolio_data[i-1]['portfolio_value']) / portfolio_data[i-1]['portfolio_value']
                returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe = avg_return / std_return if std_return > 0 else 0
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.3f}",
                    "Risk-adjusted return"
                )

def main():
    """Main dashboard function."""
    st.title("AI KrakenBot AI-Enhanced Dashboard")
    st.markdown("Advanced trading analytics with AI insights and technical analysis")
    
    # Sidebar controls
    st.sidebar.title("CHARTS: Dashboard Controls")
    
    # Session selection - but default to newest
    available_sessions = get_available_sessions()
    
    if not available_sessions:
        st.error("No trading sessions found. Please run a simulation first.")
        return
    
    # Default to newest session but allow selection
    selected_session = st.sidebar.selectbox(
        "Select Trading Session",
        available_sessions,
        index=0,  # Default to newest
        help="Current session is at the top"
    )
    
    # Pair selection for technical indicators
    trading_pairs = ["ETH/CAD", "BTC/CAD", "SOL/CAD", "XRP/CAD"]
    selected_pair = st.sidebar.selectbox(
        "Technical Analysis Pair",
        trading_pairs,
        index=0
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("REFRESH Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Load session data
    session_data = load_session_data(selected_session)
    
    if not session_data:
        st.error(f"No data found for session: {selected_session}")
        return
    
    # Main dashboard layout
    st.header(f"SESSION: Session: {selected_session}")
    
    # Performance metrics row
    st.subheader("CHARTS: Performance Overview")
    display_performance_metrics(session_data)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio chart
        portfolio_chart = create_enhanced_portfolio_chart(session_data.get('portfolio_history', []))
        st.plotly_chart(portfolio_chart, use_container_width=True)
    
    with col2:
        # Technical indicators for selected pair
        st.subheader(f"Technical Analysis: {selected_pair}")
        tech_chart = create_technical_indicators_chart_for_pair(session_data, selected_pair)
        st.plotly_chart(tech_chart, use_container_width=True)
    
    # AI predictions chart
    st.subheader("AI AI Predictions")
    ai_chart = create_ai_predictions_chart(session_data)
    st.plotly_chart(ai_chart, use_container_width=True)
    
    # Two column layout for opportunities and trading activity
    col1, col2 = st.columns(2)
    
    with col1:
        display_trading_opportunities(session_data)
    
    with col2:
        # Positions and Trade History in tabs
        st.subheader("Trading Activity")
        
        tab1, tab2 = st.tabs(["Current Positions", "Trade History"])
        
        with tab1:
            display_current_positions(session_data)
        
        with tab2:
            display_trade_history(session_data)
    
    # Footer with session info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"Session: {selected_session}")
    
    with col2:
        if session_data.get('portfolio_history'):
            last_update = session_data['portfolio_history'][-1]['timestamp']
            st.caption(f"Last Update: {pd.to_datetime(last_update).strftime('%H:%M:%S')}")
    
    with col3:
        st.caption(f"Dashboard Version: Enhanced v2.0")
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()