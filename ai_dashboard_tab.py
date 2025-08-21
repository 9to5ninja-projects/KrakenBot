"""
AI Dashboard Tab
Streamlit component for displaying AI analysis and insights
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from ai.nlp.log_analyzer import TradingLogAnalyzer
from ai.nlp.report_generator import AIReportGenerator

def render_ai_dashboard_tab():
    """Render the AI analysis dashboard tab."""
    st.header("🤖 AI Analysis & Insights")
    
    # Find available sessions
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        st.warning("No live sessions found. Please start a trading session first.")
        return
    
    # Session selector
    session_options = {session.name: str(session) for session in live_sessions}
    selected_session_name = st.selectbox(
        "Select Session for AI Analysis",
        options=list(session_options.keys()),
        index=0
    )
    
    selected_session = session_options[selected_session_name]
    
    # Initialize AI components
    if 'ai_analyzer' not in st.session_state:
        st.session_state.ai_analyzer = TradingLogAnalyzer(llm_provider="mock")
        st.session_state.ai_report_generator = AIReportGenerator(llm_provider="mock")
    
    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔍 Run AI Analysis", type="primary"):
            with st.spinner("Running AI analysis..."):
                analysis = st.session_state.ai_analyzer.analyze_session_data(selected_session)
                st.session_state.current_analysis = analysis
    
    with col2:
        if st.button("📊 Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                report_dir = Path(selected_session) / "ai_reports"
                report = st.session_state.ai_report_generator.generate_session_report(
                    selected_session, str(report_dir)
                )
                st.session_state.current_report = report
                st.success(f"Report saved to {report_dir}")
    
    with col3:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            st.rerun()
    
    # Display analysis results
    if 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        
        if "error" in analysis:
            st.error(f"Analysis failed: {analysis['error']}")
            return
        
        # Executive Summary
        st.subheader("📋 Executive Summary")
        
        session_info = analysis.get('session_info', {})
        performance = analysis.get('trading_performance', {})
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Duration",
                f"{session_info.get('duration_hours', 0):.2f}h",
                delta=None
            )
        
        with col2:
            return_pct = performance.get('total_return_percent', 0)
            st.metric(
                "Total Return",
                f"{return_pct:.4f}%",
                delta=f"{return_pct:.4f}%"
            )
        
        with col3:
            total_trades = performance.get('total_trades', 0)
            st.metric(
                "Total Trades",
                total_trades,
                delta=None
            )
        
        with col4:
            success_rate = performance.get('success_rate_percent', 0)
            st.metric(
                "Success Rate",
                f"{success_rate:.2f}%",
                delta=None
            )
        
        # AI Summary
        st.subheader("🤖 AI Summary")
        ai_summary = analysis.get('ai_summary', 'No summary available')
        st.info(ai_summary)
        
        # Market Conditions Analysis
        st.subheader("📈 Market Conditions Analysis")
        
        market_conditions = analysis.get('market_conditions', {})
        market_data = market_conditions.get('market_data', {})
        
        if market_data:
            # Create market conditions chart
            pairs = list(market_data.keys())
            price_changes = [data.get('percent_change', 0) for data in market_data.values()]
            volatilities = [data.get('volatility', 0) for data in market_data.values()]
            trends = [data.get('trend', 'unknown') for data in market_data.values()]
            
            # Price change chart
            fig_price = go.Figure()
            colors = ['green' if change > 0 else 'red' for change in price_changes]
            
            fig_price.add_trace(go.Bar(
                x=pairs,
                y=price_changes,
                marker_color=colors,
                name="Price Change %"
            ))
            
            fig_price.update_layout(
                title="Price Changes by Pair",
                xaxis_title="Trading Pairs",
                yaxis_title="Price Change (%)",
                height=400
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Volatility chart
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=pairs,
                y=volatilities,
                marker_color='orange',
                name="Volatility"
            ))
            
            fig_vol.update_layout(
                title="Volatility by Pair",
                xaxis_title="Trading Pairs",
                yaxis_title="Volatility",
                height=400
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Market conditions table
            market_df = pd.DataFrame([
                {
                    'Pair': pair,
                    'Price Change (%)': f"{data.get('percent_change', 0):.4f}",
                    'Trend': data.get('trend', 'unknown').title(),
                    'Volatility': f"{data.get('volatility', 0):.6f}",
                    'Start Price': f"{data.get('start_price', 0):.2f}",
                    'End Price': f"{data.get('end_price', 0):.2f}"
                }
                for pair, data in market_data.items()
            ])
            
            st.dataframe(market_df, use_container_width=True)
        
        # Strategy Insights
        st.subheader("💡 Strategy Insights")
        
        strategy_insights = analysis.get('strategy_insights', [])
        if strategy_insights:
            for i, insight in enumerate(strategy_insights, 1):
                st.write(f"{i}. {insight}")
        else:
            st.info("No specific strategy insights available for this session.")
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec}")
        else:
            st.info("No specific recommendations available.")
        
        # Missed Opportunities
        st.subheader("⚠️ Missed Opportunities")
        
        missed_opportunities = analysis.get('missed_opportunities', [])
        if missed_opportunities:
            st.write(f"**{len(missed_opportunities)} missed opportunities identified:**")
            
            # Create missed opportunities DataFrame
            missed_df = pd.DataFrame([
                {
                    'Timestamp': opp.get('timestamp', 'Unknown'),
                    'Pair': opp.get('pair', 'Unknown'),
                    'Price Change (%)': f"{opp.get('price_change_percent', 0):.4f}",
                    'Opportunity Type': opp.get('opportunity_type', 'Unknown').title(),
                    'Reason': opp.get('reason', 'Unknown')
                }
                for opp in missed_opportunities
            ])
            
            st.dataframe(missed_df, use_container_width=True)
            
            # Missed opportunities chart
            if len(missed_opportunities) > 1:
                fig_missed = px.scatter(
                    missed_df,
                    x='Timestamp',
                    y='Price Change (%)',
                    color='Pair',
                    size=[abs(float(x.replace('%', ''))) for x in missed_df['Price Change (%)']],
                    hover_data=['Opportunity Type', 'Reason'],
                    title="Missed Opportunities Over Time"
                )
                
                fig_missed.update_layout(height=400)
                st.plotly_chart(fig_missed, use_container_width=True)
        else:
            st.success("No missed opportunities identified - good job!")
        
        # Performance Trends (if multiple analyses available)
        st.subheader("📊 Performance Trends")
        
        # Check for historical AI analyses
        ai_analysis_dir = Path(selected_session) / "ai_analysis"
        if ai_analysis_dir.exists():
            analysis_files = list(ai_analysis_dir.glob("analysis_*.json"))
            
            if len(analysis_files) > 1:
                # Load historical analyses
                historical_data = []
                
                for file in sorted(analysis_files):
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            analysis_data = data.get('analysis', {})
                            perf = analysis_data.get('trading_performance', {})
                            
                            historical_data.append({
                                'timestamp': data.get('timestamp', ''),
                                'return_percent': perf.get('total_return_percent', 0),
                                'total_trades': perf.get('total_trades', 0),
                                'success_rate': perf.get('success_rate_percent', 0)
                            })
                    except:
                        continue
                
                if historical_data:
                    hist_df = pd.DataFrame(historical_data)
                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                    
                    # Performance over time chart
                    fig_trend = go.Figure()
                    
                    fig_trend.add_trace(go.Scatter(
                        x=hist_df['timestamp'],
                        y=hist_df['return_percent'],
                        mode='lines+markers',
                        name='Return %',
                        line=dict(color='blue')
                    ))
                    
                    fig_trend.update_layout(
                        title="Portfolio Return Over Time",
                        xaxis_title="Time",
                        yaxis_title="Return (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Trading activity over time
                    fig_activity = go.Figure()
                    
                    fig_activity.add_trace(go.Scatter(
                        x=hist_df['timestamp'],
                        y=hist_df['total_trades'],
                        mode='lines+markers',
                        name='Total Trades',
                        line=dict(color='green')
                    ))
                    
                    fig_activity.update_layout(
                        title="Trading Activity Over Time",
                        xaxis_title="Time",
                        yaxis_title="Total Trades",
                        height=400
                    )
                    
                    st.plotly_chart(fig_activity, use_container_width=True)
            else:
                st.info("Run multiple AI analyses over time to see performance trends.")
        else:
            st.info("No historical AI analysis data available yet.")
    
    else:
        st.info("Click 'Run AI Analysis' to generate insights for the selected session.")
    
    # Display comprehensive report if available
    if 'current_report' in st.session_state:
        report = st.session_state.current_report
        
        if "error" not in report:
            st.subheader("📋 Comprehensive AI Report")
            
            with st.expander("View Executive Summary", expanded=True):
                exec_summary = report.get('executive_summary', {})
                
                st.write(f"**Session Rating:** {exec_summary.get('session_rating', 'Unknown')}")
                
                highlights = exec_summary.get('highlights', [])
                if highlights:
                    st.write("**Highlights:**")
                    for highlight in highlights:
                        st.write(f"• {highlight}")
                
                concerns = exec_summary.get('concerns', [])
                if concerns:
                    st.write("**Concerns:**")
                    for concern in concerns:
                        st.write(f"• {concern}")
            
            with st.expander("Risk Assessment"):
                risk_assessment = report.get('risk_assessment', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", risk_assessment.get('risk_level', 'Unknown'))
                with col2:
                    st.metric("Risk Score", f"{risk_assessment.get('risk_score', 0)}/100")
                
                risk_factors = risk_assessment.get('risk_factors', [])
                if risk_factors:
                    st.write("**Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
            
            with st.expander("Action Items"):
                action_items = report.get('action_items', [])
                if action_items:
                    for item in action_items:
                        priority = item.get('priority', 'Unknown')
                        action = item.get('action', 'Unknown')
                        
                        if priority == 'High':
                            st.error(f"🔴 **{priority}:** {action}")
                        elif priority == 'Medium':
                            st.warning(f"🟡 **{priority}:** {action}")
                        else:
                            st.info(f"🔵 **{priority}:** {action}")
                else:
                    st.info("No specific action items identified.")

def render_ai_settings():
    """Render AI settings and configuration."""
    st.subheader("⚙️ AI Configuration")
    
    # LLM Provider selection
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["mock", "openai", "local"],
        index=0,
        help="Select the language model provider for AI analysis"
    )
    
    # Analysis frequency
    analysis_frequency = st.slider(
        "Auto-Analysis Frequency (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="How often to run automatic AI analysis"
    )
    
    # Report settings
    st.subheader("📊 Report Settings")
    
    auto_reports = st.checkbox(
        "Generate Automatic Reports",
        value=True,
        help="Automatically generate comprehensive reports"
    )
    
    report_frequency = st.selectbox(
        "Report Frequency",
        options=["Hourly", "Every 2 hours", "Every 4 hours", "Daily"],
        index=1
    )
    
    # Save settings
    if st.button("Save AI Settings"):
        settings = {
            "llm_provider": llm_provider,
            "analysis_frequency": analysis_frequency,
            "auto_reports": auto_reports,
            "report_frequency": report_frequency,
            "updated_at": datetime.now().isoformat()
        }
        
        # Save to file
        settings_file = Path("data") / "ai_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        st.success("AI settings saved successfully!")

# Example usage in main dashboard
if __name__ == "__main__":
    st.set_page_config(
        page_title="KrakenBot AI Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 KrakenBot AI Analysis Dashboard")
    
    tab1, tab2 = st.tabs(["AI Analysis", "Settings"])
    
    with tab1:
        render_ai_dashboard_tab()
    
    with tab2:
        render_ai_settings()