#!/usr/bin/env python3
"""
Fix missing values in lag and rolling features
Input: data/splits/train_data.csv, test_data.csv
Output: data/splits/train_data_fixed.csv, test_data_fixed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
SPLITS_DIR = Path(".")
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

def load_data():
    """Load train and test datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test: {len(test_df):,} records")
    return train_df, test_df

def identify_feature_types(df):
    """Identify different types of features"""
    
    # Lag features (shift operations)
    lag_features = [col for col in df.columns if '_lag_' in col]
    
    # Year-over-year change features
    yoy_features = [col for col in df.columns if '_yoy_change' in col or '_prev_year' in col]
    
    # Rolling statistics
    rolling_features = [col for col in df.columns if '_rolling_' in col]
    
    return lag_features, yoy_features, rolling_features

def fill_missing_lag_features(df, lag_features):
    """
    Fill missing lag features intelligently

    Strategy:
    - For first year of each district/group: fill with 0 (no previous data)
    - For subsequent missing values: forward fill from same district/group
    """

    print("\nFilling missing lag features...")
    print(f"  Features to fix: {len(lag_features)}")

    # Sort to ensure proper filling
    df = df.sort_values(['state_name', 'district_name', 'protected_group', 'year'])

    for col in lag_features:
        missing_before = df[col].isnull().sum()

        if missing_before > 0:
            # Fill with 0 first (for initial years)
            df[col] = df[col].fillna(0)

            # Then forward fill within groups (in case of gaps)
            df[col] = df.groupby(['state_name', 'district_name', 'protected_group'])[col].ffill()

            # If still missing, fill with 0
            df[col] = df[col].fillna(0)

            missing_after = df[col].isnull().sum()
            print(f"    {col}: {missing_before} → {missing_after}")

    print(f"  ✓ Lag features fixed")

    return df

def fill_missing_yoy_features(df, yoy_features):
    """
    Fill missing year-over-year features
    
    Strategy:
    - _prev_year: fill with 0 (no previous year data)
    - _yoy_change: fill with 0 (assume no change if no previous data)
    """
    
    print("\nFilling missing YoY features...")
    print(f"  Features to fix: {len(yoy_features)}")
    
    for col in yoy_features:
        missing_before = df[col].isnull().sum()
        
        if missing_before > 0:
            df[col] = df[col].fillna(0)
            
            missing_after = df[col].isnull().sum()
            print(f"    {col}: {missing_before} → {missing_after}")
    
    print(f"  ✓ YoY features fixed")
    
    return df

def fill_missing_rolling_features(df, rolling_features):
    """
    Fill missing rolling statistics
    
    Strategy:
    - Rolling mean: use actual value for first year(s)
    - Rolling std: fill with 0 for first year(s) (no variation yet)
    """
    
    print("\nFilling missing rolling features...")
    print(f"  Features to fix: {len(rolling_features)}")
    
    # Sort to ensure proper filling
    df = df.sort_values(['state_name', 'district_name', 'protected_group', 'year'])
    
    for col in rolling_features:
        missing_before = df[col].isnull().sum()
        
        if missing_before > 0:
            if '_mean_' in col:
                # For rolling mean: use current value if no history
                base_col = col.split('_rolling_mean_')
                if base_col in df.columns:
                    df[col] = df[col].fillna(df[base_col])
                else:
                    df[col] = df[col].fillna(0)
            
            elif '_std_' in col:
                # For rolling std: fill with 0 (no variation initially)
                df[col] = df[col].fillna(0)
            
            else:
                # Default: fill with 0
                df[col] = df[col].fillna(0)
            
            missing_after = df[col].isnull().sum()
            print(f"    {col}: {missing_before} → {missing_after}")
    
    print(f"  ✓ Rolling features fixed")
    
    return df

def verify_no_missing_values(df, name):
    """Verify that all missing values are fixed"""
    
    print(f"\nVerifying {name} dataset...")
    
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        print(f"  ✗ Still have missing values in {len(missing_cols)} columns:")
        for col in missing_cols[:10]:
            count = df[col].isnull().sum()
            print(f"    {col}: {count} missing")
        if len(missing_cols) > 10:
            print(f"    ... and {len(missing_cols)-10} more")
        return False
    else:
        print(f"  ✓ No missing values!")
        return True

def save_fixed_data(train_df, test_df):
    """Save fixed datasets"""
    
    # Backup original files
    backup_dir = SPLITS_DIR / "backup_original"
    backup_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(TRAIN_FILE, backup_dir / "train_data_original.csv")
    shutil.copy(TEST_FILE, backup_dir / "test_data_original.csv")
    print(f"\n✓ Original files backed up to: {backup_dir}")
    
    # Save fixed files (overwrite originals)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print(f"\n✓ Fixed train data saved to: {TRAIN_FILE}")
    print(f"  Shape: {train_df.shape}")
    print(f"✓ Fixed test data saved to: {TEST_FILE}")
    print(f"  Shape: {test_df.shape}")
    
    # Save fix metadata
    fix_metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_features': len(train_df.columns),
        'test_features': len(test_df.columns),
        'missing_values_before': {
            'train': 'See validation_report.json',
            'test': 'See validation_report.json'
        },
        'missing_values_after': {
            'train': int(train_df.isnull().sum().sum()),
            'test': int(test_df.isnull().sum().sum())
        },
        'fill_strategy': {
            'lag_features': 'Fill with 0 for initial years',
            'yoy_features': 'Fill with 0 (assume no change)',
            'rolling_mean': 'Fill with current value',
            'rolling_std': 'Fill with 0 (no variation)'
        }
    }
    
    metadata_path = SPLITS_DIR / "fix_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(fix_metadata, f, indent=2)
    
    print(f"✓ Fix metadata saved to: {metadata_path}")

def main():
    print("="*70)
    print("FIXING MISSING LAG AND ROLLING FEATURES")
    print("="*70)
    
    # Load data
    train_df, test_df = load_data()
    
    # Identify feature types
    lag_features, yoy_features, rolling_features = identify_feature_types(train_df)
    
    print(f"\nFeature counts:")
    print(f"  Lag features: {len(lag_features)}")
    print(f"  YoY features: {len(yoy_features)}")
    print(f"  Rolling features: {len(rolling_features)}")
    
    # Fix train data
    print("\n" + "="*70)
    print("FIXING TRAIN DATA")
    print("="*70)
    
    train_df = fill_missing_lag_features(train_df, lag_features)
    train_df = fill_missing_yoy_features(train_df, yoy_features)
    train_df = fill_missing_rolling_features(train_df, rolling_features)
    
    train_valid = verify_no_missing_values(train_df, "train")
    
    # Fix test data
    print("\n" + "="*70)
    print("FIXING TEST DATA")
    print("="*70)
    
    test_df = fill_missing_lag_features(test_df, lag_features)
    test_df = fill_missing_yoy_features(test_df, yoy_features)
    test_df = fill_missing_rolling_features(test_df, rolling_features)
    
    test_valid = verify_no_missing_values(test_df, "test")
    
    # Save if valid
    if train_valid and test_valid:
        save_fixed_data(train_df, test_df)
        
        print("\n" + "="*70)
        print("✓ ALL MISSING VALUES FIXED!")
        print("="*70)
        print("\nYou can now train models without missing value errors.")
    else:
        print("\n" + "="*70)
        print("✗ SOME ISSUES REMAIN")
        print("="*70)
        print("Please review the output above.")

if __name__ == "__main__":
    main()

