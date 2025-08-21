"""
Data preparation utilities for AI models
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

def prepare_training_data(session_dirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Prepare training data from multiple trading sessions.
    
    Args:
        session_dirs: List of session directory paths
        
    Returns:
        Dictionary containing prepared datasets
    """
    all_data = {
        'trades': [],
        'portfolio_history': [],
        'price_history': [],
        'performance_metrics': []
    }
    
    for session_dir in session_dirs:
        session_path = Path(session_dir)
        if not session_path.exists():
            continue
            
        # Load session data
        session_data = load_session_files(session_path)
        
        # Add session identifier
        session_id = session_path.name
        
        # Process trades
        trades = session_data.get('trades', [])
        for trade in trades:
            trade['session_id'] = session_id
            all_data['trades'].append(trade)
        
        # Process portfolio history
        portfolio_history = session_data.get('portfolio_history', [])
        for entry in portfolio_history:
            entry['session_id'] = session_id
            all_data['portfolio_history'].append(entry)
        
        # Process price history
        price_history = session_data.get('price_history', [])
        for entry in price_history:
            entry['session_id'] = session_id
            all_data['price_history'].append(entry)
        
        # Add performance metrics
        performance = session_data.get('performance', {})
        if performance:
            performance['session_id'] = session_id
            all_data['performance_metrics'].append(performance)
    
    # Convert to DataFrames
    datasets = {}
    for key, data in all_data.items():
        if data:
            datasets[key] = pd.DataFrame(data)
        else:
            datasets[key] = pd.DataFrame()
    
    return datasets

def load_session_files(session_path: Path) -> Dict[str, Any]:
    """Load all data files from a session directory."""
    data = {}
    
    # Define file mappings
    file_mappings = {
        'metadata': 'session_metadata.json',
        'trades': 'trades.json',
        'portfolio_history': 'portfolio_history.json',
        'price_history': 'price_history.json',
        'performance': 'performance_summary.json'
    }
    
    for key, filename in file_mappings.items():
        file_path = session_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data[key] = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path}")
                data[key] = {}
        else:
            data[key] = {} if key in ['metadata', 'performance'] else []
    
    return data

def validate_session_data(session_dir: str) -> Dict[str, Any]:
    """
    Validate session data for completeness and quality.
    
    Args:
        session_dir: Path to session directory
        
    Returns:
        Validation results
    """
    session_path = Path(session_dir)
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'data_quality': {}
    }
    
    if not session_path.exists():
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Session directory not found: {session_dir}")
        return validation_results
    
    # Check required files
    required_files = [
        'session_metadata.json',
        'portfolio_history.json',
        'price_history.json'
    ]
    
    for filename in required_files:
        file_path = session_path / filename
        if not file_path.exists():
            validation_results['errors'].append(f"Missing required file: {filename}")
            validation_results['is_valid'] = False
    
    # Load and validate data
    try:
        session_data = load_session_files(session_path)
        
        # Validate metadata
        metadata = session_data.get('metadata', {})
        if not metadata.get('session_id'):
            validation_results['warnings'].append("Missing session_id in metadata")
        
        # Validate portfolio history
        portfolio_history = session_data.get('portfolio_history', [])
        if len(portfolio_history) < 2:
            validation_results['warnings'].append("Insufficient portfolio history data")
        
        validation_results['data_quality']['portfolio_entries'] = len(portfolio_history)
        
        # Validate price history
        price_history = session_data.get('price_history', [])
        if len(price_history) < 2:
            validation_results['warnings'].append("Insufficient price history data")
        
        validation_results['data_quality']['price_entries'] = len(price_history)
        
        # Validate trades
        trades = session_data.get('trades', [])
        validation_results['data_quality']['total_trades'] = len(trades)
        
        # Check data consistency
        if portfolio_history and price_history:
            portfolio_timestamps = [entry.get('timestamp') for entry in portfolio_history]
            price_timestamps = [entry.get('timestamp') for entry in price_history]
            
            if len(set(portfolio_timestamps) & set(price_timestamps)) < min(len(portfolio_timestamps), len(price_timestamps)) * 0.8:
                validation_results['warnings'].append("Timestamp mismatch between portfolio and price data")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Error validating data: {str(e)}")
    
    return validation_results

def create_feature_matrix(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create feature matrix for machine learning models.
    
    Args:
        datasets: Dictionary of DataFrames from prepare_training_data
        
    Returns:
        Feature matrix DataFrame
    """
    features = []
    
    # Process price history for features
    price_df = datasets.get('price_history', pd.DataFrame())
    if not price_df.empty:
        # Sort by timestamp
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.sort_values('timestamp')
        
        # Create price-based features
        for session_id in price_df['session_id'].unique():
            session_prices = price_df[price_df['session_id'] == session_id].copy()
            
            # Extract price data for each pair
            for idx, row in session_prices.iterrows():
                prices = row.get('prices', {})
                
                feature_row = {
                    'session_id': session_id,
                    'timestamp': row['timestamp'],
                }
                
                # Add price features
                for pair, price in prices.items():
                    feature_row[f'{pair}_price'] = price
                
                features.append(feature_row)
    
    if features:
        feature_df = pd.DataFrame(features)
        
        # Add technical indicators
        feature_df = add_technical_indicators(feature_df)
        
        return feature_df
    
    return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the feature matrix."""
    if df.empty:
        return df
    
    # Get price columns
    price_columns = [col for col in df.columns if col.endswith('_price')]
    
    for col in price_columns:
        pair_name = col.replace('_price', '')
        
        # Moving averages
        df[f'{pair_name}_ma_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{pair_name}_ma_10'] = df[col].rolling(window=10, min_periods=1).mean()
        
        # Price changes
        df[f'{pair_name}_price_change'] = df[col].pct_change()
        df[f'{pair_name}_price_change_5'] = df[col].pct_change(periods=5)
        
        # Volatility (rolling standard deviation)
        df[f'{pair_name}_volatility'] = df[col].rolling(window=10, min_periods=1).std()
        
        # Price position relative to recent high/low
        df[f'{pair_name}_high_10'] = df[col].rolling(window=10, min_periods=1).max()
        df[f'{pair_name}_low_10'] = df[col].rolling(window=10, min_periods=1).min()
        df[f'{pair_name}_position'] = (df[col] - df[f'{pair_name}_low_10']) / (df[f'{pair_name}_high_10'] - df[f'{pair_name}_low_10'])
    
    # Fill NaN values
    df = df.fillna(method='forward').fillna(0)
    
    return df

def prepare_labels(datasets: Dict[str, pd.DataFrame], prediction_horizon: int = 1) -> pd.DataFrame:
    """
    Prepare labels for supervised learning.
    
    Args:
        datasets: Dictionary of DataFrames
        prediction_horizon: Number of periods ahead to predict
        
    Returns:
        DataFrame with labels
    """
    labels = []
    
    # Use portfolio history to create labels
    portfolio_df = datasets.get('portfolio_history', pd.DataFrame())
    if not portfolio_df.empty:
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df = portfolio_df.sort_values(['session_id', 'timestamp'])
        
        for session_id in portfolio_df['session_id'].unique():
            session_data = portfolio_df[portfolio_df['session_id'] == session_id].copy()
            
            # Calculate future returns
            session_data['future_return'] = session_data['total_value'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
            
            # Create binary labels (profitable vs not profitable)
            session_data['profitable'] = (session_data['future_return'] > 0).astype(int)
            
            # Create categorical labels
            session_data['return_category'] = pd.cut(
                session_data['future_return'],
                bins=[-float('inf'), -0.01, 0.01, float('inf')],
                labels=['negative', 'neutral', 'positive']
            )
            
            labels.extend(session_data.to_dict('records'))
    
    return pd.DataFrame(labels) if labels else pd.DataFrame()

def split_data(features: pd.DataFrame, labels: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        features: Feature matrix
        labels: Labels DataFrame
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if features.empty or labels.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Merge features and labels on timestamp and session_id
    merged = features.merge(labels, on=['timestamp', 'session_id'], how='inner')
    
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Split by session to avoid data leakage
    sessions = merged['session_id'].unique()
    n_test_sessions = max(1, int(len(sessions) * test_size))
    
    test_sessions = sessions[-n_test_sessions:]
    train_sessions = sessions[:-n_test_sessions]
    
    train_data = merged[merged['session_id'].isin(train_sessions)]
    test_data = merged[merged['session_id'].isin(test_sessions)]
    
    # Separate features and labels
    feature_columns = [col for col in merged.columns if col not in ['session_id', 'timestamp', 'future_return', 'profitable', 'return_category']]
    
    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]
    y_train = train_data[['profitable', 'future_return']]
    y_test = test_data[['profitable', 'future_return']]
    
    return X_train, X_test, y_train, y_test