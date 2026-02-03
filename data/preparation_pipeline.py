"""
Data Preparation Pipeline for FC-MT-LSTM Model
Handles data loading, preprocessing, feature engineering, and dataset creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose


class CrimeDataset(Dataset):
    """
    PyTorch Dataset for crime prediction data
    """
    def __init__(self, X, y, groups):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.groups = torch.LongTensor(groups)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.groups[idx]


def create_temporal_features(df):
    """
    Generate time-based features from datetime
    """
    df['year'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.year
    df['month'] = df['month']
    df['quarter'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.quarter
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    return df


def create_lag_features(df, target='crime_rate', lags=[1, 2, 3, 6, 12]):
    """
    Create lagged versions of target variable
    """
    if 'date' not in df.columns:
        df = create_temporal_features(df)
    
    df = df.sort_values(['state', 'district', 'protected_group', 'date']).copy()
    
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df.groupby(['state', 'district', 'protected_group'])[target].shift(lag)
    
    return df


def create_rolling_features(df, target='crime_rate', windows=[3, 6, 12]):
    """
    Rolling mean, std, min, max
    """
    if 'date' not in df.columns:
        df = create_temporal_features(df)
    
    df = df.sort_values(['state', 'district', 'protected_group', 'date']).copy()
    
    for window in windows:
        # Rolling mean
        rolling_means = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f'{target}_rolling_mean_{window}'] = rolling_means.values
        
        # Rolling std
        rolling_stds = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .std()
        )
        # Fill NaN values with 0 for std (when window has only 1 value)
        df[f'{target}_rolling_std_{window}'] = rolling_stds.values
        
        # Rolling min
        rolling_mins = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .min()
        )
        df[f'{target}_rolling_min_{window}'] = rolling_mins.values
        
        # Rolling max
        rolling_maxs = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .max()
        )
        df[f'{target}_rolling_max_{window}'] = rolling_maxs.values
    
    return df


def extract_trend_seasonality(df, target='crime_rate'):
    """
    Decompose time series into trend and seasonal components
    """
    if 'date' not in df.columns:
        df = create_temporal_features(df)
    
    df = df.sort_values(['state', 'district', 'protected_group', 'date'])
    
    # Initialize columns with NaN of correct dtype
    df = df.copy()  # Make a copy to avoid SettingWithCopyWarning
    df['trend'] = np.nan
    df['seasonality'] = np.nan
    df['residual'] = np.nan
    
    for (state, district, group), group_df in df.groupby(['state', 'district', 'protected_group']):
        if len(group_df) >= 24:  # Need at least 2 years
            try:
                # Convert to time series and resample if needed
                ts = group_df.set_index('date')[target]
                
                # Ensure we have monthly data
                ts = ts.resample('M').mean()  # Monthly resampling
                
                if len(ts) >= 24:  # Still enough data after resampling
                    decomposition = seasonal_decompose(
                        ts.values,
                        model='additive',
                        period=12,
                        extrapolate_trend='freq'
                    )
                    
                    # Map back to original dataframe - corrected indexing
                    original_indices = group_df.index
                    values_to_assign = min(len(decomposition.trend), len(original_indices))
                    df.loc[original_indices[:values_to_assign], 'trend'] = decomposition.trend[:values_to_assign]
                    df.loc[original_indices[:values_to_assign], 'seasonality'] = decomposition.seasonal[:values_to_assign]
                    df.loc[original_indices[:values_to_assign], 'residual'] = decomposition.resid[:values_to_assign]
            except Exception as e:
                print(f"Warning: Could not decompose series for ({state}, {district}, {group}): {e}")
                continue
    
    return df


def create_crime_type_features(df):
    """
    Aggregate and normalize crime types
    """
    crime_types = ['murder', 'rape', 'kidnapping', 'dacoity', 'robbery',
                   'burglary', 'theft', 'riots', 'arson', 'hurt', 
                   'cheating', 'counterfeiting']
    
    # Filter out columns that don't exist in df
    existing_crime_types = [col for col in crime_types if col in df.columns]
    
    # Calculate total crimes
    df['total_crimes'] = df[existing_crime_types].sum(axis=1, min_count=1) if existing_crime_types else 0
    
    # Calculate crime type ratios where possible
    for crime in existing_crime_types:
        df[f'{crime}_ratio'] = df[crime] / (df['total_crimes'] + 1)  # +1 to avoid division by zero
    
    # Violent vs non-violent
    violent_crimes = [crime for crime in ['murder', 'rape', 'kidnapping', 'robbery', 'dacoity'] if crime in df.columns]
    if violent_crimes:
        df['violent_crime_count'] = df[violent_crimes].sum(axis=1, min_count=1)
        df['violent_crime_ratio'] = df['violent_crime_count'] / (df['total_crimes'] + 1)
    
    # Property crimes
    property_crimes = [crime for crime in ['burglary', 'theft', 'cheating', 'counterfeiting'] if crime in df.columns]
    if property_crimes:
        df['property_crime_count'] = df[property_crimes].sum(axis=1, min_count=1)
        df['property_crime_ratio'] = df['property_crime_count'] / (df['total_crimes'] + 1)
    
    return df


def create_regional_features(df, demographic_data=None):
    """
    Merge demographic and regional metadata
    """
    if demographic_data is not None and not demographic_data.empty:
        # Merge demographic data
        df = df.merge(
            demographic_data[['state', 'district', 'year', 'population', 
                              'literacy_rate', 'poverty_index', 'urban_rural']],
            on=['state', 'district', 'year'],
            how='left'
        )
        
        # Calculate crime rate per capita
        df['crime_rate_per_capita'] = (df['total_crimes'] / df['population']) * 100000
    
    # Encode categorical variables
    if 'urban_rural' in df.columns:
        df['urban_rural_encoded'] = (df['urban_rural'] == 'urban').astype(int)
    
    # State encoding (one-hot or label encoding)
    if 'state' in df.columns:
        df['state_encoded'] = pd.Categorical(df['state']).codes
    
    return df


def encode_protected_groups(df):
    """
    Encode protected groups as numerical labels
    """
    group_mapping = {
        'SC': 0,
        'ST': 1,
        'Women': 2,
        'Children': 3
    }
    
    if 'protected_group' in df.columns:
        df['group_label'] = df['protected_group'].map(group_mapping)
    else:
        df['group_label'] = 0  # Default to first group if column doesn't exist
    
    return df


def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data by time to prevent data leakage
    """
    if 'date' not in df.columns:
        df = create_temporal_features(df)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Val: {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, val_df, test_df


def create_sequences(df, sequence_length=12, target='crime_rate'):
    """
    Create overlapping sequences for LSTM input
    """
    if 'date' not in df.columns:
        df = create_temporal_features(df)
    
    X_sequences = []
    y_targets = []
    group_labels = []
    
    for (state, district, group), group_df in df.groupby(['state', 'district', 'protected_group']):
        # Sort by date
        group_df = group_df.sort_values('date')
        
        # Get feature columns (excluding target, date, protected_group related)
        feature_cols = [col for col in group_df.columns 
                       if col not in ['crime_rate', 'date', 'protected_group', 'group_label']
                       and not col.startswith('crime_rate')]
        
        if not feature_cols:
            print(f"Warning: No feature columns found for ({state}, {district}, {group})")
            continue
            
        # Extract features and target
        features = group_df[feature_cols].values
        targets = group_df[target].values
        group_label_vals = group_df['group_label'].values
        
        # Ensure we have enough data for sequences
        if len(group_df) < sequence_length + 1:
            print(f"Warning: Insufficient data for ({state}, {district}, {group}). Need at least {sequence_length + 1}, got {len(group_df)}")
            continue
        
        # Create sequences
        for i in range(len(group_df) - sequence_length):
            X_sequences.append(features[i:i+sequence_length])
            y_targets.append(targets[i+sequence_length])
            group_labels.append(group_label_vals[i])  # Use the current group label
    
    if X_sequences:
        X = np.array(X_sequences)  # Shape: (num_samples, sequence_length, num_features)
        y = np.array(y_targets)    # Shape: (num_samples,)
        groups = np.array(group_labels)  # Shape: (num_samples,)
        
        print(f"Created {X.shape[0]} sequences of length {sequence_length} with {X.shape[2]} features each")
        return X, y, groups
    else:
        print("Warning: No sequences could be created from the data")
        return np.array([]), np.array([]), np.array([])


def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training set statistics
    """
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print("Warning: One or more datasets are empty. Skipping normalization.")
        return X_train, X_val, X_test, None
    
    # Reshape to 2D for scaling
    n_train, seq_len, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_features)
    
    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_crime_data_pipeline(data_path, sequence_length=12, target_col='crime_rate'):
    """
    Complete data preparation pipeline
    """
    print("Loading and preparing data...")
    
    # Load raw data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    # Ensure required columns exist, create dummy ones if missing
    required_cols = ['state', 'district', 'year', 'month', 'protected_group', target_col]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in data. Creating dummy values.")
            if col == 'state':
                df['state'] = 'State_' + df.index.astype(str)
            elif col == 'district':
                df['district'] = 'District_' + df.index.astype(str)
            elif col == 'year':
                df['year'] = 2018
            elif col == 'month':
                df['month'] = 1
            elif col == 'protected_group':
                df['protected_group'] = 'SC'  # Default to first group
            elif col == target_col:
                df[target_col] = 10.0  # Default target value
    
    # Add group labels
    df = encode_protected_groups(df)
    
    # Feature Engineering
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating lag features...")
    df = create_lag_features(df, target=target_col)
    
    print("Creating rolling features...")
    df = create_rolling_features(df, target=target_col)
    
    print("Creating crime type features...")
    df = create_crime_type_features(df)
    
    # Note: We'll skip demographic features unless a specific file is provided
    print("Creating regional features...")
    df = create_regional_features(df)
    
    print("Creating trend and seasonality features...")
    df = extract_trend_seasonality(df, target=target_col)
    
    print(f"Data shape after feature engineering: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Time-based split
    print("Splitting data (time-based)...")
    train_df, val_df, test_df = time_based_split(df, train_ratio=0.7, val_ratio=0.15)
    
    # Create sequences
    print("Creating sequences for LSTM...")
    X_train, y_train, groups_train = create_sequences(train_df, sequence_length, target_col)
    X_val, y_val, groups_val = create_sequences(val_df, sequence_length, target_col)
    X_test, y_test, groups_test = create_sequences(test_df, sequence_length, target_col)
    
    print(f"Train samples: {len(X_train) if X_train.size > 0 else 0}")
    print(f"Val samples: {len(X_val) if X_val.size > 0 else 0}")
    print(f"Test samples: {len(X_test) if X_test.size > 0 else 0}")
    
    # Normalize features
    if X_train.size > 0 and X_val.size > 0 and X_test.size > 0:
        print("Normalizing features...")
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    else:
        print("Skipping normalization due to insufficient data")
        scaler = None
    
    # Create datasets and dataloaders
    if X_train.size > 0:
        train_dataset = CrimeDataset(X_train, y_train, groups_train)
        val_dataset = CrimeDataset(X_val, y_val, groups_val)
        test_dataset = CrimeDataset(X_test, y_test, groups_test)
        
        # Create default collate function to handle potential issues with empty batches
        def collate_fn(batch):
            # Filter out any None values from batch
            batch = [item for item in batch if item[0] is not None]
            if not batch:
                return None, None, None
            return torch.utils.data.dataloader.default_collate(batch)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    else:
        print("Not enough data to create datasets")
        train_loader = val_loader = test_loader = None
        scaler = None
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'groups_train': groups_train,
        'groups_val': groups_val,
        'groups_test': groups_test
    }


def create_synthetic_data(n_samples=1000, n_features=20):
    """
    Create synthetic crime data for testing purposes
    """
    np.random.seed(42)
    
    # Create a DataFrame with required columns
    df = pd.DataFrame({
        'state': np.random.choice(['Maharashtra', 'Uttar Pradesh', 'Delhi', 'Karnataka'], n_samples),
        'district': ['District_' + str(i % 100) for i in range(n_samples)],
        'year': np.random.choice([2018, 2019, 2020, 2021], n_samples),
        'month': np.random.choice(range(1, 13), n_samples),
        'protected_group': np.random.choice(['SC', 'ST', 'Women', 'Children'], n_samples),
        'crime_rate': np.random.uniform(0, 50, n_samples),  # Target variable
    })
    
    # Add some additional features
    for i in range(n_features):
        df[f'feature_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Add crime type features
    crime_types = ['murder', 'rape', 'kidnapping', 'dacoity', 'robbery', 'burglary', 'theft']
    for crime in crime_types:
        df[crime] = np.random.uniform(0, 20, n_samples)
    
    # Create a temporary CSV file
    temp_file = 'synthetic_crime_data.csv'
    df.to_csv(temp_file, index=False)
    
    print(f"Created synthetic data with {n_samples} samples and saved to {temp_file}")
    return temp_file, df


if __name__ == "__main__":
    print("Testing Data Preparation Pipeline...")
    
    # Create synthetic data for testing
    temp_file, df = create_synthetic_data(n_samples=2000, n_features=15)
    
    # Test the pipeline
    data_prepared = prepare_crime_data_pipeline(temp_file, sequence_length=6, target_col='crime_rate')
    
    if data_prepared['train_loader'] is not None:
        # Get a sample batch
        for batch in data_prepared['train_loader']:
            x_batch, y_batch, groups_batch = batch
            print(f"Sample batch - X: {x_batch.shape}, y: {y_batch.shape}, groups: {groups_batch.shape}")
            break
    
    print("✅ Data preparation pipeline test completed!")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)