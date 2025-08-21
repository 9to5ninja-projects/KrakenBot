"""
AI Price Prediction Dashboard Components
Streamlit components for price prediction and volatility analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import json

from ai.prediction.price_predictor import PricePredictor
from ai.prediction.volatility_analyzer import VolatilityAnalyzer
from ai.strategy.strategy_optimizer import StrategyOptimizer

def render_price_prediction_tab():
    """Render the price prediction dashboard tab."""
    st.header("📈 AI Price Prediction")
    
    # Find available sessions
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        st.warning("No live sessions found. Please start a trading session first.")
        return
    
    # Session selector
    session_options = {session.name: str(session) for session in live_sessions}
    selected_session_name = st.selectbox(
        "Select Session for Price Prediction",
        options=list(session_options.keys()),
        index=0
    )
    
    selected_session = session_options[selected_session_name]
    
    # Initialize predictor
    if 'price_predictor' not in st.session_state:
        st.session_state.price_predictor = PricePredictor(model_type="ensemble")
    
    # Prediction controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        prediction_minutes = st.slider(
            "Prediction Horizon (minutes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5
        )
    
    with col2:
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Training models and generating predictions..."):
                # Load session data
                price_history = load_session_price_data(selected_session)
                
                if price_history:
                    # Train models
                    statistical_models = st.session_state.price_predictor.train_statistical_model(price_history)
                    ml_models = st.session_state.price_predictor.train_ml_model(price_history)
                    
                    if statistical_models or ml_models:
                        # Generate predictions
                        predictions = st.session_state.price_predictor.predict_prices(
                            price_history, prediction_minutes
                        )
                        
                        if predictions:
                            st.session_state.current_predictions = predictions
                            st.success(f"Predictions generated successfully for {len(predictions)} pairs!")
                            
                            # Show a quick preview
                            st.write("**Quick Preview:**")
                            for pair, pred in list(predictions.items())[:2]:  # Show first 2
                                st.write(f"• {pair}: {pred['price_change_percent']:+.2f}% (Confidence: {pred['confidence']:.2f})")
                        else:
                            st.error("Failed to generate predictions - no valid predictions returned")
                    else:
                        st.error("Insufficient data for model training (need 20+ data points)")
                else:
                    st.error("No price data available for predictions")
    
    with col3:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.1
        )
    
    # Display predictions
    if 'current_predictions' in st.session_state:
        predictions = st.session_state.current_predictions
        
        if predictions:
            # Predictions summary
            st.subheader("🎯 Price Predictions Summary")
            
            # Create predictions DataFrame
            pred_data = []
            for pair, pred in predictions.items():
                pred_data.append({
                    'Pair': pair,
                    'Current Price': f"${pred['current_price']:.2f}",
                    'Predicted Price': f"${pred['predicted_price']:.2f}",
                    'Price Change': f"{pred['price_change']:+.2f}",
                    'Change %': f"{pred['price_change_percent']:+.4f}%",
                    'Confidence': f"{pred['confidence']:.2f}",
                    'Method': pred['method'].title()
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True)
            
            # Predictions visualization
            st.subheader("📊 Price Predictions Chart")
            
            fig = go.Figure()
            
            pairs = list(predictions.keys())
            current_prices = [predictions[pair]['current_price'] for pair in pairs]
            predicted_prices = [predictions[pair]['predicted_price'] for pair in pairs]
            confidences = [predictions[pair]['confidence'] for pair in pairs]
            
            # Current prices
            fig.add_trace(go.Bar(
                x=pairs,
                y=current_prices,
                name='Current Price',
                marker_color='blue',
                opacity=0.7
            ))
            
            # Predicted prices
            fig.add_trace(go.Bar(
                x=pairs,
                y=predicted_prices,
                name='Predicted Price',
                marker_color='green',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"Price Predictions ({prediction_minutes} minutes ahead)",
                xaxis_title="Trading Pairs",
                yaxis_title="Price ($)",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price change visualization
            st.subheader("📈 Expected Price Changes")
            
            price_changes = [predictions[pair]['price_change_percent'] for pair in pairs]
            colors = ['green' if change > 0 else 'red' for change in price_changes]
            
            fig_change = go.Figure()
            fig_change.add_trace(go.Bar(
                x=pairs,
                y=price_changes,
                marker_color=colors,
                name="Price Change %"
            ))
            
            fig_change.update_layout(
                title="Expected Price Changes (%)",
                xaxis_title="Trading Pairs",
                yaxis_title="Price Change (%)",
                height=400
            )
            
            st.plotly_chart(fig_change, use_container_width=True)
            
            # Trading signals
            st.subheader("🎯 AI Trading Signals")
            
            signals = st.session_state.price_predictor.get_trading_signals(
                predictions, confidence_threshold, min_price_change=0.5
            )
            
            if signals:
                signal_data = []
                for pair, signal in signals.items():
                    signal_data.append({
                        'Pair': pair,
                        'Signal': signal['signal'],
                        'Strength': f"{signal['strength']:.2f}",
                        'Confidence': f"{signal['confidence']:.2f}",
                        'Expected Change': f"{signal['predicted_change_percent']:+.4f}%",
                        'Current Price': f"${signal['current_price']:.2f}",
                        'Target Price': f"${signal['predicted_price']:.2f}",
                        'Method': signal['method'].title()
                    })
                
                signal_df = pd.DataFrame(signal_data)
                
                # Color code signals
                def color_signal(val):
                    if val == 'BUY':
                        return 'background-color: lightgreen'
                    elif val == 'SELL':
                        return 'background-color: lightcoral'
                    else:
                        return 'background-color: lightyellow'
                
                styled_df = signal_df.style.applymap(color_signal, subset=['Signal'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Signal distribution
                signal_counts = signal_df['Signal'].value_counts()
                
                fig_signals = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="Trading Signal Distribution",
                    color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'yellow'}
                )
                
                st.plotly_chart(fig_signals, use_container_width=True)
            else:
                st.info("No trading signals generated with current confidence threshold.")
            
            # Model performance (if available)
            if hasattr(st.session_state.price_predictor, 'models') and 'ml' in st.session_state.price_predictor.models:
                st.subheader("🔬 Model Performance")
                
                ml_models = st.session_state.price_predictor.models['ml']
                
                perf_data = []
                for pair, model_data in ml_models.items():
                    perf = model_data['performance']
                    perf_data.append({
                        'Pair': pair,
                        'Train MSE': f"{perf['train_mse']:.6f}",
                        'Test MSE': f"{perf['test_mse']:.6f}",
                        'Test MAE': f"{perf['test_mae']:.6f}",
                        'Train Samples': perf['train_samples'],
                        'Test Samples': perf['test_samples']
                    })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No predictions available. Click 'Generate Predictions' to start.")
    else:
        st.info("Click 'Generate Predictions' to see AI-powered price forecasts.")

def render_volatility_trading_tab():
    """Render the volatility trading dashboard tab."""
    st.header("⚡ Volatility Trading System")
    
    # Find available sessions
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        st.warning("No live sessions found. Please start a trading session first.")
        return
    
    # Session selector
    session_options = {session.name: str(session) for session in live_sessions}
    selected_session_name = st.selectbox(
        "Select Session for Volatility Analysis",
        options=list(session_options.keys()),
        index=0
    )
    
    selected_session = session_options[selected_session_name]
    
    # Initialize volatility analyzer
    if 'volatility_analyzer' not in st.session_state:
        st.session_state.volatility_analyzer = VolatilityAnalyzer(lookback_periods=20)
    
    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("⚡ Analyze Volatility", type="primary"):
            with st.spinner("Analyzing volatility patterns..."):
                # Load session data
                price_history = load_session_price_data(selected_session)
                
                if price_history:
                    # Calculate volatility metrics
                    volatility_metrics = st.session_state.volatility_analyzer.calculate_volatility_metrics(price_history)
                    
                    if volatility_metrics:
                        st.session_state.volatility_metrics = volatility_metrics
                        
                        # Detect opportunities
                        opportunities = st.session_state.volatility_analyzer.detect_volatility_opportunities(volatility_metrics)
                        st.session_state.volatility_opportunities = opportunities
                        
                        # Generate summary
                        summary = st.session_state.volatility_analyzer.get_volatility_summary(volatility_metrics)
                        st.session_state.volatility_summary = summary
                        
                        st.success("Volatility analysis completed!")
                    else:
                        st.error("Insufficient data for volatility analysis (need 20+ data points)")
                else:
                    st.error("No price data available for volatility analysis")
    
    with col2:
        opportunity_threshold = st.slider(
            "Opportunity Threshold",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        st.session_state.volatility_analyzer.opportunity_threshold = opportunity_threshold
    
    with col3:
        lookback_periods = st.slider(
            "Lookback Periods",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )
        st.session_state.volatility_analyzer.lookback_periods = lookback_periods
    
    # Display volatility analysis
    if 'volatility_metrics' in st.session_state:
        metrics = st.session_state.volatility_metrics
        
        # Volatility summary
        if 'volatility_summary' in st.session_state:
            summary = st.session_state.volatility_summary
            
            st.subheader("📊 Market Volatility Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Pairs Analyzed",
                    summary['total_pairs_analyzed']
                )
            
            with col2:
                st.metric(
                    "Avg Volatility Ratio",
                    f"{summary['average_volatility_ratio']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Market State",
                    summary['market_volatility_state'].replace('_', ' ').title()
                )
            
            with col4:
                st.metric(
                    "Trading Recommendation",
                    summary['trading_recommendation'].replace('_', ' ').title()
                )
        
        # Volatility metrics table
        st.subheader("📈 Volatility Metrics by Pair")
        
        vol_data = []
        for pair, metric in metrics.items():
            vol_data.append({
                'Pair': pair,
                'Current Vol': f"{metric['current_volatility']:.6f}",
                'Vol Ratio': f"{metric['volatility_ratio']:.2f}",
                'Regime': metric['volatility_regime'].replace('_', ' ').title(),
                'Vol Percentile': f"{metric['volatility_percentile']:.1f}%",
                'Price Trend': metric['price_trend'].title(),
                'Vol Trend': metric['volatility_trend'].title(),
                'Current Price': f"${metric['current_price']:.2f}"
            })
        
        vol_df = pd.DataFrame(vol_data)
        st.dataframe(vol_df, use_container_width=True)
        
        # Volatility visualization
        st.subheader("⚡ Volatility Analysis Charts")
        
        pairs = list(metrics.keys())
        vol_ratios = [metrics[pair]['volatility_ratio'] for pair in pairs]
        vol_percentiles = [metrics[pair]['volatility_percentile'] for pair in pairs]
        
        # Volatility ratio chart
        fig_vol = go.Figure()
        colors = ['red' if ratio > 1.5 else 'orange' if ratio > 1.2 else 'green' for ratio in vol_ratios]
        
        fig_vol.add_trace(go.Bar(
            x=pairs,
            y=vol_ratios,
            marker_color=colors,
            name="Volatility Ratio"
        ))
        
        fig_vol.add_hline(y=1.0, line_dash="dash", line_color="blue", 
                         annotation_text="Normal Volatility")
        fig_vol.add_hline(y=opportunity_threshold, line_dash="dash", line_color="red", 
                         annotation_text="Opportunity Threshold")
        
        fig_vol.update_layout(
            title="Volatility Ratios (Current vs Historical)",
            xaxis_title="Trading Pairs",
            yaxis_title="Volatility Ratio",
            height=500
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Volatility percentile chart
        fig_perc = go.Figure()
        fig_perc.add_trace(go.Scatter(
            x=pairs,
            y=vol_percentiles,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=vol_percentiles,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Percentile")
            ),
            name="Volatility Percentile"
        ))
        
        fig_perc.add_hline(y=80, line_dash="dash", line_color="red", 
                          annotation_text="High Volatility (80th percentile)")
        fig_perc.add_hline(y=20, line_dash="dash", line_color="green", 
                          annotation_text="Low Volatility (20th percentile)")
        
        fig_perc.update_layout(
            title="Volatility Percentiles",
            xaxis_title="Trading Pairs",
            yaxis_title="Volatility Percentile (%)",
            height=400
        )
        
        st.plotly_chart(fig_perc, use_container_width=True)
        
        # Volatility opportunities
        if 'volatility_opportunities' in st.session_state:
            opportunities = st.session_state.volatility_opportunities
            
            st.subheader("🎯 Volatility Trading Opportunities")
            
            if opportunities:
                opp_data = []
                for opp in opportunities:
                    opp_data.append({
                        'Pair': opp['pair'],
                        'Opportunity': opp['opportunity_type'].replace('_', ' ').title(),
                        'Signal': opp['signal'].replace('_', ' ').title(),
                        'Strategy': opp['strategy'].replace('_', ' ').title(),
                        'Confidence': f"{opp['confidence']:.2f}",
                        'Duration': opp['expected_duration'],
                        'Risk Level': opp['risk_level'].title(),
                        'Current Price': f"${opp['details']['current_price']:.2f}"
                    })
                
                opp_df = pd.DataFrame(opp_data)
                
                # Color code by risk level
                def color_risk(val):
                    if val == 'High':
                        return 'background-color: lightcoral'
                    elif val == 'Medium':
                        return 'background-color: lightyellow'
                    else:
                        return 'background-color: lightgreen'
                
                styled_opp_df = opp_df.style.applymap(color_risk, subset=['Risk Level'])
                st.dataframe(styled_opp_df, use_container_width=True)
                
                # Generate trading strategies for top opportunities
                st.subheader("📋 Detailed Trading Strategies")
                
                for i, opp in enumerate(opportunities[:3]):  # Show top 3
                    with st.expander(f"Strategy {i+1}: {opp['pair']} - {opp['opportunity_type'].replace('_', ' ').title()}"):
                        strategy = st.session_state.volatility_analyzer.generate_volatility_trading_strategy(opp)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Entry Strategy:**")
                            entry = strategy['entry_strategy']
                            st.write(f"- Method: {entry['method'].replace('_', ' ').title()}")
                            st.write(f"- Condition: {entry['entry_condition']}")
                            st.write(f"- Price Range: {entry['entry_price_range']}")
                            st.write(f"- Max Wait: {entry['max_wait_time']}")
                            
                            st.write("**Position Sizing:**")
                            sizing = strategy['position_sizing']
                            st.write(f"- Recommended Size: {sizing['recommended_size']}")
                            st.write(f"- Base Size: {sizing['base_size']}")
                            st.write(f"- Confidence Multiplier: {sizing['confidence_multiplier']:.2f}")
                        
                        with col2:
                            st.write("**Exit Strategy:**")
                            exit_strat = strategy['exit_strategy']
                            st.write(f"- Profit Target: {exit_strat['profit_target']}")
                            st.write(f"- Stop Loss: {exit_strat['stop_loss']}")
                            st.write(f"- Time Exit: {exit_strat['time_exit']}")
                            
                            st.write("**Risk Management:**")
                            risk = strategy['risk_management']
                            st.write(f"- Max Position: {risk['max_position_size']}")
                            for key, value in risk.items():
                                if key != 'max_position_size':
                                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                st.info("No volatility opportunities detected with current threshold settings.")
    else:
        st.info("Click 'Analyze Volatility' to detect trading opportunities based on market volatility.")

def load_session_price_data(session_dir: str) -> dict:
    """Load price history data from session directory."""
    try:
        price_file = Path(session_dir) / "price_history.json"
        if price_file.exists():
            with open(price_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return {}

# Add these functions to the simple_dashboard.py imports
def render_strategy_optimization_tab():
    """Render the strategy optimization dashboard tab."""
    st.header("🎯 AI Strategy Optimization")
    
    # Find available sessions
    data_dir = Path("data")
    live_sessions = list(data_dir.glob("live_session_*"))
    
    if not live_sessions:
        st.warning("No live sessions found. Please start a trading session first.")
        return
    
    # Session selector
    session_options = {session.name: str(session) for session in live_sessions}
    selected_session_name = st.selectbox(
        "Select Session for Strategy Optimization",
        options=list(session_options.keys()),
        index=0
    )
    
    selected_session = session_options[selected_session_name]
    
    # Initialize optimizer
    if 'strategy_optimizer' not in st.session_state:
        st.session_state.strategy_optimizer = StrategyOptimizer()
    
    # Optimization controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🎯 Optimize Strategy", type="primary"):
            with st.spinner("Analyzing performance and optimizing strategy..."):
                # Load session data
                session_data = load_full_session_data(selected_session)
                
                if session_data:
                    # Analyze current performance
                    analysis = st.session_state.strategy_optimizer.analyze_current_performance(session_data)
                    st.session_state.optimization_analysis = analysis
                    
                    # Generate optimized parameters
                    current_params = session_data.get('metadata', {}).get('parameters', {})
                    recommendations = analysis.get('optimization_recommendations', [])
                    
                    optimized = st.session_state.strategy_optimizer.generate_optimized_parameters(
                        current_params, recommendations
                    )
                    st.session_state.optimized_parameters = optimized
                    
                    st.success("Strategy optimization completed!")
                else:
                    st.error("No session data available for optimization")
    
    with col2:
        if st.button("💾 Save Optimized Parameters"):
            if 'optimized_parameters' in st.session_state:
                # Save optimized parameters to file
                opt_file = Path(selected_session) / "optimized_parameters.json"
                with open(opt_file, 'w') as f:
                    json.dump(st.session_state.optimized_parameters, f, indent=2, default=str)
                st.success(f"Parameters saved to {opt_file}")
            else:
                st.warning("No optimized parameters to save. Run optimization first.")
    
    # Display optimization results
    if 'optimization_analysis' in st.session_state:
        analysis = st.session_state.optimization_analysis
        
        # Performance overview
        st.subheader("📊 Current Performance Analysis")
        
        perf = analysis['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{perf['total_return_percent']:.4f}%",
                delta=f"{perf['total_return_percent']:.4f}%"
            )
        
        with col2:
            st.metric(
                "Success Rate",
                f"{perf['success_rate_percent']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                perf['total_trades']
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{perf['max_drawdown']:.2f}%"
            )
        
        # Parameter effectiveness
        st.subheader("⚙️ Parameter Effectiveness Analysis")
        
        param_eff = analysis['parameter_effectiveness']
        
        # Thresholds effectiveness
        if 'thresholds' in param_eff:
            st.write("**Trading Thresholds:**")
            
            thresh = param_eff['thresholds']
            
            col1, col2 = st.columns(2)
            
            with col1:
                buy_thresh = thresh['buy_threshold']
                st.write(f"Buy Threshold: {buy_thresh['current_value']}")
                st.progress(buy_thresh['effectiveness_score'])
                st.write(f"Effectiveness: {buy_thresh['effectiveness_score']:.2f}")
            
            with col2:
                sell_thresh = thresh['sell_threshold']
                st.write(f"Sell Threshold: {sell_thresh['current_value']}")
                st.progress(sell_thresh['effectiveness_score'])
                st.write(f"Effectiveness: {sell_thresh['effectiveness_score']:.2f}")
        
        # Improvement opportunities
        st.subheader("🎯 Improvement Opportunities")
        
        opportunities = analysis['improvement_opportunities']
        
        if opportunities:
            opp_data = []
            for opp in opportunities:
                opp_data.append({
                    'Type': opp['type'].replace('_', ' ').title(),
                    'Priority': opp['priority'].title(),
                    'Current Value': opp['current_value'],
                    'Target Value': opp['target_value'],
                    'Description': opp['description'],
                    'Impact': opp['impact_estimate'].title()
                })
            
            opp_df = pd.DataFrame(opp_data)
            
            # Color code by priority
            def color_priority(val):
                if val == 'High':
                    return 'background-color: lightcoral'
                elif val == 'Medium':
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightgreen'
            
            styled_opp_df = opp_df.style.applymap(color_priority, subset=['Priority'])
            st.dataframe(styled_opp_df, use_container_width=True)
        else:
            st.info("No specific improvement opportunities identified.")
        
        # Optimization recommendations
        st.subheader("💡 Optimization Recommendations")
        
        recommendations = analysis['optimization_recommendations']
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}: {rec['parameter'].replace('_', ' ').title()} ({rec['priority'].title()} Priority)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Current Value:** {rec['current_value']}")
                        st.write(f"**Recommended Value:** {rec['recommended_value']}")
                        st.write(f"**Change:** {rec['change_percent']:+.1f}%")
                    
                    with col2:
                        st.write(f"**Rationale:** {rec['rationale']}")
                        st.write(f"**Expected Impact:** {rec['expected_impact']}")
        
        # Optimized parameters
        if 'optimized_parameters' in st.session_state:
            st.subheader("🎯 Optimized Parameter Set")
            
            optimized = st.session_state.optimized_parameters
            
            # Show applied changes
            st.write("**Applied Changes:**")
            
            changes = optimized['applied_changes']
            
            if changes:
                change_data = []
                for change in changes:
                    change_data.append({
                        'Parameter': change['parameter'].replace('_', ' ').title(),
                        'Old Value': change['old_value'],
                        'New Value': change['new_value'],
                        'Change %': f"{change['change_percent']:+.1f}%",
                        'Rationale': change['rationale']
                    })
                
                change_df = pd.DataFrame(change_data)
                st.dataframe(change_df, use_container_width=True)
                
                # Show complete optimized parameter set
                st.write("**Complete Optimized Parameters:**")
                
                opt_params = optimized['optimized_parameters']
                param_display = []
                
                for param, value in opt_params.items():
                    param_display.append({
                        'Parameter': param.replace('_', ' ').title(),
                        'Value': value,
                        'Type': type(value).__name__
                    })
                
                param_df = pd.DataFrame(param_display)
                st.dataframe(param_df, use_container_width=True)
                
                # Optimization summary
                summary = optimized['optimization_summary']
                
                st.write("**Optimization Summary:**")
                st.write(f"- Total Recommendations: {summary['total_recommendations']}")
                st.write(f"- Applied Changes: {summary['applied_changes']}")
                st.write("- Expected Improvements:")
                for improvement in summary['expected_improvements']:
                    st.write(f"  • {improvement}")
            else:
                st.info("No parameter changes were applied.")
    else:
        st.info("Click 'Optimize Strategy' to analyze performance and generate parameter recommendations.")

def load_full_session_data(session_dir: str) -> dict:
    """Load complete session data from directory."""
    try:
        session_path = Path(session_dir)
        data = {}
        
        # Load all session files
        files_to_load = {
            'metadata': 'session_metadata.json',
            'portfolio_history': 'portfolio_history.json',
            'price_history': 'price_history.json',
            'trades': 'trades.json',
            'performance': 'performance_summary.json'
        }
        
        for key, filename in files_to_load.items():
            file_path = session_path / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[key] = json.load(f)
            else:
                data[key] = {} if key in ['metadata', 'performance'] else []
        
        return data
    except Exception as e:
        st.error(f"Error loading session data: {e}")
        return {}