#!/usr/bin/env python3
"""
Load and consolidate crime data from 4 protected groups
Input: 4 CSV files (ST, SC, Women, Children)
Output: data/raw/master_crime_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configuration
RAW_DATA_DIR = Path(".")  # Place your uploaded files here
OUTPUT_DIR = Path("transform/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
FILES = {
    'ST': 'districtwise-crime-against-sts-2017-onwards.csv',
    'SC': 'districtwise-crime-against-scs-2017-onwards.csv',
    'Women': 'districtwise-crime-against-women-2017-onwards.csv',
    'Children': 'districtwise-crime-against-children-2017-onwards.csv'
}

def load_and_label_group(file_path, group_name):
    """Load a CSV and add group label"""
    print(f"Loading {group_name} data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    
    # Add protected group column
    df['protected_group'] = group_name
    
    print(f"  Loaded {len(df):,} records for {group_name}")
    print(f"  Years: {df['year'].min()} to {df['year'].max()}")
    print(f"  Columns: {len(df.columns)}")
    
    return df

def standardize_columns(df):
    """Standardize column names across all files"""
    
    # Convert all column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Standardize common variations
    rename_map = {
        'statename': 'state_name',
        'statecode': 'state_code',
        'districtname': 'district_name',
        'districtcode': 'district_code',
        'registrationcircles': 'registration_circles'
    }
    
    df = df.rename(columns=rename_map)
    
    return df

def aggregate_crime_categories(df):
    """
    Aggregate detailed crime columns into broader categories using keyword matching

    Categories:
    - violent_crimes: murder, hurt, assault, acid attack
    - sexual_crimes: rape, sexual harassment, assault, POCSO
    - property_crimes: robbery, dacoity, arson, criminal intimidation
    - kidnapping_crimes: all kidnapping and abduction types
    - total_crimes: sum of all crimes
    """

    # Identify crime columns (numeric columns, excluding metadata)
    metadata_cols = ['id', 'year', 'state_name', 'state_code',
                     'district_name', 'district_code',
                     'registration_circles', 'protected_group']

    crime_cols = [col for col in df.columns
                  if col not in metadata_cols
                  and df[col].dtype in ['int64', 'float64']]

    print(f"\nFound {len(crime_cols)} crime columns")

    # Convert all crime columns to numeric (handle any strings)
    for col in crime_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Define category mappings using KEYWORD matching
    # Keywords are matched case-insensitively against column names
    category_map = {
        'violent_crimes': {
            'keywords': ['murder', 'hurt', 'assault', 'acid_attack', 'grievous_hurt'],
            'exclude': ['rape', 'sexual']  # Exclude sexual crimes from violent
        },
        'sexual_crimes': {
            'keywords': ['rape', 'sexual_harassment', 'sexual_assault', 'pocso', 
                        'modesty', 'prostitution', 'trafficking'],
            'exclude': []
        },
        'property_crimes': {
            'keywords': ['robbery', 'dacoity', 'arson', 'criminal_intimidation', 
                        'theft', 'burglary'],
            'exclude': []
        },
        'kidnapping_crimes': {
            'keywords': ['kidnap', 'abduct', 'traffick', 'importation'],
            'exclude': []
        }
    }

    # Create aggregated columns
    for category, config in category_map.items():
        keywords = config['keywords']
        exclude = config.get('exclude', [])
        
        matching_cols = []
        for col in crime_cols:
            col_lower = col.lower()
            # Check if any keyword matches
            if any(kw.lower() in col_lower for kw in keywords):
                # Check if any exclude keyword matches
                if not any(ex.lower() in col_lower for ex in exclude):
                    matching_cols.append(col)

        if matching_cols:
            df[category] = df[matching_cols].sum(axis=1)
            print(f"  {category}: {len(matching_cols)} columns aggregated")
            if len(matching_cols) <= 10:
                print(f"    Columns: {', '.join(matching_cols)}")
        else:
            df[category] = 0
            print(f"  {category}: No matching columns found")

    # Total crimes (sum of all numeric crime columns)
    df['total_crimes'] = df[crime_cols].sum(axis=1)

    return df

def create_master_dataset():
    """Main function to create master consolidated dataset"""
    
    print("="*70)
    print("STEP 1: LOADING AND CONSOLIDATING CRIME DATA")
    print("="*70)
    
    all_dataframes = []
    
    # Load each file
    for group_name, filename in FILES.items():
        file_path = RAW_DATA_DIR / filename
        
        if not file_path.exists():
            print(f"WARNING: File not found: {file_path}")
            print(f"  Skipping {group_name} data")
            continue
        
        df = load_and_label_group(file_path, group_name)
        df = standardize_columns(df)
        all_dataframes.append(df)
    
    if not all_dataframes:
        print("\nERROR: No data files found!")
        print(f"Please place your CSV files in: {RAW_DATA_DIR}")
        sys.exit(1)
    
    # Concatenate all dataframes
    print(f"\nConcatenating {len(all_dataframes)} datasets...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nMaster dataset created:")
    print(f"  Total records: {len(master_df):,}")
    print(f"  Years: {master_df['year'].min()} to {master_df['year'].max()}")
    print(f"  Groups: {master_df['protected_group'].unique()}")
    print(f"  States: {master_df['state_name'].nunique()}")
    print(f"  Districts: {master_df['district_name'].nunique()}")
    
    # Aggregate crime categories
    print("\n" + "="*70)
    print("STEP 2: AGGREGATING CRIME CATEGORIES")
    print("="*70)
    
    master_df = aggregate_crime_categories(master_df)
    
    # Save master dataset
    output_path = OUTPUT_DIR / "master_crime_data.csv"
    master_df.to_csv(output_path, index=False)
    print(f"\n✓ Master dataset saved to: {output_path}")
    print(f"  Shape: {master_df.shape}")
    
    # Display summary statistics
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print("\nRecords per group:")
    print(master_df['protected_group'].value_counts())
    
    print("\nRecords per year:")
    print(master_df['year'].value_counts().sort_index())
    
    print("\nTop 10 states by total crimes:")
    state_crimes = master_df.groupby('state_name')['total_crimes'].sum().sort_values(ascending=False)
    print(state_crimes.head(10))
    
    print("\nCrime category summary:")
    category_cols = ['violent_crimes', 'sexual_crimes', 'property_crimes', 
                     'kidnapping_crimes', 'total_crimes']
    print(master_df[category_cols].describe())
    
    return master_df

if __name__ == "__main__":
    master_df = create_master_dataset()
    print("\n✓ Data consolidation complete!")

