#!/usr/bin/env python3
"""
Clean master crime dataset
Input: data/raw/master_crime_data.csv
Output: data/cleaned/crime_data_cleaned.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
INPUT_FILE = Path("master_crime_data.csv")
OUTPUT_DIR = Path("../cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load master dataset"""
    print("Loading master dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} records")
    return df

def remove_invalid_records(df):
    """Remove records with invalid metadata"""
    
    initial_count = len(df)
    print("\nRemoving invalid records...")
    
    # Remove records with invalid years
    df = df[df['year'].between(2017, 2025)]
    print(f"  After year filter: {len(df):,} records")
    
    # Remove records with missing state/district
    df = df.dropna(subset=['state_name', 'district_name'])
    print(f"  After location filter: {len(df):,} records")
    
    # Remove special districts (Crime Branch, CID, Railway, etc.)
    exclude_keywords = ['cid', 'grp', 'railway', 'rpo', 'stf', 'crime branch', 
                       'economic offences', 'cyber', 'terrorist']
    
    for keyword in exclude_keywords:
        df = df[~df['district_name'].str.lower().str.contains(keyword, na=False)]
    
    print(f"  After special district filter: {len(df):,} records")
    
    removed = initial_count - len(df)
    print(f"  Total removed: {removed:,} records ({removed/initial_count*100:.1f}%)")
    
    return df

def handle_missing_values(df):
    """Handle missing values in crime columns"""
    
    print("\nHandling missing values...")
    
    # Get crime columns
    crime_cols = ['violent_crimes', 'sexual_crimes', 'property_crimes', 
                  'kidnapping_crimes', 'total_crimes']
    
    # Check missing values
    missing_before = df[crime_cols].isnull().sum()
    print(f"  Missing values before:")
    for col in crime_cols:
        if missing_before[col] > 0:
            print(f"    {col}: {missing_before[col]}")
    
    # Fill missing values with 0 (assumption: not reported = 0 cases)
    df[crime_cols] = df[crime_cols].fillna(0)
    
    print(f"  ✓ Missing values filled with 0")
    
    return df

def detect_and_handle_outliers(df):
    """Detect and cap extreme outliers"""
    
    print("\nDetecting outliers...")
    
    crime_cols = ['violent_crimes', 'sexual_crimes', 'property_crimes', 
                  'kidnapping_crimes', 'total_crimes']
    
    outliers_info = {}
    
    for col in crime_cols:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (3*IQR for extreme outliers)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"  {col}:")
            print(f"    Outliers found: {len(outliers)}")
            print(f"    Upper bound: {upper_bound:.1f}")
            print(f"    Max value: {df[col].max():.1f}")
            
            # Cap outliers at upper bound (keep lower outliers as they might be valid zeros)
            df.loc[df[col] > upper_bound, col] = upper_bound
            
            outliers_info[col] = {
                'count': len(outliers),
                'upper_bound': float(upper_bound),
                'max_before': float(df[col].max())
            }
        else:
            print(f"  {col}: No extreme outliers")
    
    return df, outliers_info

def add_derived_features(df):
    """Add useful derived features"""
    
    print("\nAdding derived features...")
    
    # Crime rate per capita (using district as proxy)
    # Note: You'll need population data for accurate rates
    # For now, we create placeholder
    df['crimes_per_district'] = df.groupby(['state_name', 'district_name', 'year'])['total_crimes'].transform('sum')
    
    # Year-over-year change
    df = df.sort_values(['state_name', 'district_name', 'protected_group', 'year'])
    
    for crime_type in ['violent_crimes', 'sexual_crimes', 'property_crimes', 'total_crimes']:
        df[f'{crime_type}_prev_year'] = df.groupby(['state_name', 'district_name', 'protected_group'])[crime_type].shift(1)
        df[f'{crime_type}_yoy_change'] = df[crime_type] - df[f'{crime_type}_prev_year']
    
    print(f"  ✓ Added year-over-year change features")
    
    return df

def save_cleaned_data(df, outliers_info):
    """Save cleaned dataset and metadata"""
    
    # Save cleaned data
    output_path = OUTPUT_DIR / "crime_data_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    
    # Save cleaning metadata
    metadata = {
        'total_records': len(df),
        'years': {
            'min': int(df['year'].min()),
            'max': int(df['year'].max())
        },
        'groups': df['protected_group'].value_counts().to_dict(),
        'outliers_handled': outliers_info,
        'missing_values': 'Filled with 0',
        'excluded_districts': 'CID, Railway, Crime Branch, etc.'
    }
    
    metadata_path = OUTPUT_DIR / "cleaning_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")

def main():
    print("="*70)
    print("DATA CLEANING PIPELINE")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Clean data
    df = remove_invalid_records(df)
    df = handle_missing_values(df)
    df, outliers_info = detect_and_handle_outliers(df)
    df = add_derived_features(df)
    
    # Save
    save_cleaned_data(df, outliers_info)
    
    # Summary
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"Final dataset: {len(df):,} records")
    print(f"Groups: {df['protected_group'].nunique()}")
    print(f"Years: {df['year'].nunique()} ({df['year'].min()}-{df['year'].max()})")
    print(f"States: {df['state_name'].nunique()}")
    print(f"Districts: {df['district_name'].nunique()}")
    
    print("\n✓ Data cleaning complete!")

if __name__ == "__main__":
    main()

