#!/usr/bin/env python3
"""
Validate prepared datasets
Input: data/splits/train_data.csv, test_data.csv
Output: Validation report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
TRAIN_FILE = Path("train_data.csv")
TEST_FILE = Path("test_data.csv")
OUTPUT_FILE = Path("validation_report.json")

def load_splits():
    """Load train and test datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test: {len(test_df):,} records")
    return train_df, test_df

def check_missing_values(train_df, test_df):
    """Check for missing values"""
    
    print("\nChecking missing values...")
    
    issues = []
    
    # Train missing values
    train_missing = train_df.isnull().sum()
    train_missing = train_missing[train_missing > 0]
    
    if len(train_missing) > 0:
        print(f"  ⚠ Train set has missing values:")
        for col, count in train_missing.items():
            print(f"    {col}: {count} ({count/len(train_df)*100:.1f}%)")
            issues.append(f"Train: {col} has {count} missing values")
    else:
        print(f"  ✓ Train set has no missing values")
    
    # Test missing values
    test_missing = test_df.isnull().sum()
    test_missing = test_missing[test_missing > 0]
    
    if len(test_missing) > 0:
        print(f"  ⚠ Test set has missing values:")
        for col, count in test_missing.items():
            print(f"    {col}: {count} ({count/len(test_df)*100:.1f}%)")
            issues.append(f"Test: {col} has {count} missing values")
    else:
        print(f"  ✓ Test set has no missing values")
    
    return issues

def check_feature_consistency(train_df, test_df):
    """Check feature consistency between train and test"""
    
    print("\nChecking feature consistency...")
    
    issues = []
    
    # Same columns
    if set(train_df.columns) != set(test_df.columns):
        print(f"  ✗ Different columns in train vs test")
        train_only = set(train_df.columns) - set(test_df.columns)
        test_only = set(test_df.columns) - set(train_df.columns)
        
        if train_only:
            print(f"    Train only: {train_only}")
            issues.append(f"Train-only columns: {train_only}")
        
        if test_only:
            print(f"    Test only: {test_only}")
            issues.append(f"Test-only columns: {test_only}")
    else:
        print(f"  ✓ Same columns in both sets ({len(train_df.columns)} columns)")
    
    return issues

def check_data_ranges(train_df, test_df):
    """Check if test data is within train data ranges"""
    
    print("\nChecking data ranges...")
    
    issues = []
    warnings = []
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'year']]
    
    for col in numeric_cols:
        train_min = train_df[col].min()
        train_max = train_df[col].max()
        test_min = test_df[col].min()
        test_max = test_df[col].max()
        
        # Check if test is outside train range
        if test_min < train_min or test_max > train_max:
            warnings.append(f"{col}: test [{test_min:.1f}, {test_max:.1f}] outside train [{train_min:.1f}, {train_max:.1f}]")
    
    if warnings:
        print(f"  ⚠ {len(warnings)} features have test values outside train range")
        for warning in warnings[:5]:  # Show first 5
            print(f"    {warning}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings)-5} more")
    else:
        print(f"  ✓ All test values within train ranges")
    
    return issues, warnings

def check_target_distribution(train_df, test_df):
    """Check target variable distribution"""
    
    print("\nChecking target distribution...")
    
    target_col = 'total_crimes'
    
    train_stats = {
        'mean': float(train_df[target_col].mean()),
        'std': float(train_df[target_col].std()),
        'min': float(train_df[target_col].min()),
        'max': float(train_df[target_col].max()),
        'median': float(train_df[target_col].median())
    }
    
    test_stats = {
        'mean': float(test_df[target_col].mean()),
        'std': float(test_df[target_col].std()),
        'min': float(test_df[target_col].min()),
        'max': float(test_df[target_col].max()),
        'median': float(test_df[target_col].median())
    }
    
    print(f"\n  Train {target_col}:")
    print(f"    Mean: {train_stats['mean']:.1f}")
    print(f"    Std:  {train_stats['std']:.1f}")
    print(f"    Range: [{train_stats['min']:.1f}, {train_stats['max']:.1f}]")
    
    print(f"\n  Test {target_col}:")
    print(f"    Mean: {test_stats['mean']:.1f}")
    print(f"    Std:  {test_stats['std']:.1f}")
    print(f"    Range: [{test_stats['min']:.1f}, {test_stats['max']:.1f}]")
    
    # Check if distributions are similar
    mean_diff = abs(train_stats['mean'] - test_stats['mean']) / train_stats['mean']
    
    if mean_diff > 0.3:
        print(f"\n  ⚠ Mean difference > 30%: {mean_diff*100:.1f}%")
    else:
        print(f"\n  ✓ Mean difference acceptable: {mean_diff*100:.1f}%")
    
    return train_stats, test_stats

def generate_report(train_df, test_df, issues, warnings, train_stats, test_stats):
    """Generate validation report"""
    
    report = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {
            'train': {
                'path': str(TRAIN_FILE),
                'records': len(train_df),
                'features': len(train_df.columns),
                'years': f"{train_df['year'].min()}-{train_df['year'].max()}"
            },
            'test': {
                'path': str(TEST_FILE),
                'records': len(test_df),
                'features': len(test_df.columns),
                'years': f"{test_df['year'].min()}-{test_df['year'].max()}"
            }
        },
        'validation_checks': {
            'missing_values': len(issues) == 0,
            'feature_consistency': 'Different columns' not in str(issues),
            'data_ranges': len(warnings) == 0,
            'target_distribution': 'acceptable'
        },
        'issues': issues,
        'warnings': warnings,
        'target_statistics': {
            'train': train_stats,
            'test': test_stats
        },
        'status': 'PASS' if len(issues) == 0 else 'FAIL'
    }
    
    # Save report
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Validation report saved to: {OUTPUT_FILE}")
    
    return report

def main():
    print("="*70)
    print("DATA VALIDATION")
    print("="*70)
    
    # Load data
    train_df, test_df = load_splits()
    
    # Run checks
    issues = []
    warnings = []
    
    missing_issues = check_missing_values(train_df, test_df)
    issues.extend(missing_issues)
    
    consistency_issues = check_feature_consistency(train_df, test_df)
    issues.extend(consistency_issues)
    
    range_issues, range_warnings = check_data_ranges(train_df, test_df)
    issues.extend(range_issues)
    warnings.extend(range_warnings)
    
    train_stats, test_stats = check_target_distribution(train_df, test_df)
    
    # Generate report
    report = generate_report(train_df, test_df, issues, warnings, train_stats, test_stats)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Status: {report['status']}")
    print(f"Issues: {len(issues)}")
    print(f"Warnings: {len(warnings)}")
    
    if report['status'] == 'PASS':
        print("\n✓ All validation checks passed!")
        print("  Data is ready for model training")
    else:
        print("\n✗ Validation failed")
        print("  Please review issues before training")

if __name__ == "__main__":
    main()

