#!/usr/bin/env python3
"""
Temporal train/test split
Input: data/features/crime_data_features.csv
Output: data/splits/train_data.csv, test_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
INPUT_FILE = Path("crime_data_features.csv")
OUTPUT_DIR = Path("../splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Split configuration
TEST_YEAR = 2022  # Use 2022+ as test set (adjust based on your data)

def load_data():
    """Load feature dataset"""
    print("Loading feature dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} records")
    print(f"  Years: {df['year'].min()} - {df['year'].max()}")
    return df

def temporal_split(df, test_year=TEST_YEAR):
    """Split data temporally"""
    
    print(f"\nCreating temporal split (test year >= {test_year})...")
    
    # Train: all years before test_year
    train_df = df[df['year'] < test_year].copy()
    
    # Test: test_year and after
    test_df = df[df['year'] >= test_year].copy()
    
    print(f"\nTrain set:")
    print(f"  Records: {len(train_df):,}")
    print(f"  Years: {train_df['year'].min()} - {train_df['year'].max()}")
    print(f"  Groups: {train_df['protected_group'].value_counts().to_dict()}")
    
    print(f"\nTest set:")
    print(f"  Records: {len(test_df):,}")
    print(f"  Years: {test_df['year'].min()} - {test_df['year'].max()}")
    print(f"  Groups: {test_df['protected_group'].value_counts().to_dict()}")
    
    return train_df, test_df

def validate_split(train_df, test_df):
    """Validate that split has no temporal leakage"""
    
    print("\nValidating split...")
    
    # Check 1: No overlap in years
    train_years = set(train_df['year'].unique())
    test_years = set(test_df['year'].unique())
    overlap = train_years.intersection(test_years)
    
    if overlap:
        print(f"  ✗ WARNING: Year overlap detected: {overlap}")
        return False
    else:
        print(f"  ✓ No year overlap")
    
    # Check 2: All groups present in both sets
    train_groups = set(train_df['protected_group'].unique())
    test_groups = set(test_df['protected_group'].unique())
    
    if train_groups != test_groups:
        print(f"  ✗ WARNING: Different groups in train vs test")
        return False
    else:
        print(f"  ✓ Same groups in both sets: {train_groups}")
    
    # Check 3: Reasonable split ratio
    split_ratio = len(test_df) / len(train_df)
    if split_ratio < 0.1 or split_ratio > 0.5:
        print(f"  ⚠ WARNING: Unusual split ratio: {split_ratio:.2f}")
    else:
        print(f"  ✓ Split ratio: {split_ratio:.2f}")
    
    return True

def save_splits(train_df, test_df):
    """Save train and test datasets"""
    
    # Save train
    train_path = OUTPUT_DIR / "train_data.csv"
    train_df.to_csv(train_path, index=False)
    print(f"\n✓ Train data saved to: {train_path}")
    
    # Save test
    test_path = OUTPUT_DIR / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Test data saved to: {test_path}")
    
    # Save metadata
    metadata = {
        'test_year_threshold': TEST_YEAR,
        'train': {
            'records': len(train_df),
            'years': f"{train_df['year'].min()}-{train_df['year'].max()}",
            'groups': train_df['protected_group'].value_counts().to_dict()
        },
        'test': {
            'records': len(test_df),
            'years': f"{test_df['year'].min()}-{test_df['year'].max()}",
            'groups': test_df['protected_group'].value_counts().to_dict()
        },
        'split_ratio': len(test_df) / len(train_df)
    }
    
    metadata_path = OUTPUT_DIR / "split_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")

def main():
    print("="*70)
    print("TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Split
    train_df, test_df = temporal_split(df, test_year=TEST_YEAR)
    
    # Validate
    is_valid = validate_split(train_df, test_df)
    
    if is_valid:
        # Save
        save_splits(train_df, test_df)
        print("\n✓ Train/test split complete!")
    else:
        print("\n✗ Split validation failed. Please review.")

if __name__ == "__main__":
    main()

