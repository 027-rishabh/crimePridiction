#!/usr/bin/env python3
"""
Export baseline results in format ready for React dashboard
"""

import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "react_dashboard_data.json"

def load_all_predictions():
    """Load predictions from all models"""
    pred_dir = RESULTS_DIR / "model_predictions"
    
    all_predictions = {}
    model_files = {
        'SARIMA': 'sarima_predictions.json',
        'Prophet': 'prophet_predictions.json',
        'Random Forest': 'random_forest_predictions.json',
        'XGBoost': 'xgboost_predictions.json',
        'CNN-LSTM': 'cnn_lstm_predictions.json',
        'Transformer': 'transformer_predictions.json'
    }
    
    for model_name, filename in model_files.items():
        filepath = pred_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                all_predictions[model_name] = json.load(f)
    
    return all_predictions

def load_all_metrics():
    """Load fairness metrics from all models"""
    metrics_dir = RESULTS_DIR / "fairness_metrics"
    
    all_metrics = {}
    model_files = {
        'SARIMA': 'sarima_fairness.json',
        'Prophet': 'prophet_fairness.json',
        'Random Forest': 'random_forest_fairness.json',
        'XGBoost': 'xgboost_fairness.json',
        'CNN-LSTM': 'cnn_lstm_fairness.json',
        'Transformer': 'transformer_fairness.json'
    }
    
    for model_name, filename in model_files.items():
        filepath = metrics_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                all_metrics[model_name] = json.load(f)
    
    return all_metrics

def create_comparison_data(all_metrics):
    """Create model comparison data"""
    comparison = []
    
    for model_name, metrics in all_metrics.items():
        comparison.append({
            'model': model_name,
            'mae': metrics['overall']['mae'],
            'rmse': metrics['overall']['rmse'],
            'r2': metrics['overall']['r2'],
            'fairness_gap': metrics['fairness_gap'],
            'fairness_ratio': metrics['fairness_ratio'],
            'training_time': metrics.get('training_time_minutes', 0)
        })
    
    # Sort by MAE
    comparison.sort(key=lambda x: x['mae'])
    
    return comparison

def create_fairness_data(all_metrics):
    """Create fairness breakdown data"""
    fairness_data = {}
    
    for model_name, metrics in all_metrics.items():
        by_group = metrics.get('by_group', {})
        fairness_data[model_name] = {
            group: {
                'mae': data['mae'],
                'rmse': data['rmse'],
                'r2': data['r2'],
                'count': data['count']
            }
            for group, data in by_group.items()
        }
    
    return fairness_data

def create_geographic_data(all_predictions):
    """Create geographic distribution data"""
    # Use first available model's predictions
    if not all_predictions:
        return []
    
    first_model = list(all_predictions.values())[0]
    
    # Aggregate by state
    df = pd.DataFrame(first_model)
    
    state_data = df.groupby('state_name').agg({
        'actual': 'sum',
        'district_name': 'nunique'
    }).reset_index()
    
    state_data.columns = ['state', 'total_crimes', 'num_districts']
    
    return state_data.to_dict('records')

def main():
    print("="*70)
    print("EXPORTING RESULTS FOR REACT DASHBOARD")
    print("="*70)
    
    print("\nLoading predictions...")
    all_predictions = load_all_predictions()
    print(f"  Loaded {len(all_predictions)} model predictions")
    
    print("\nLoading metrics...")
    all_metrics = load_all_metrics()
    print(f"  Loaded {len(all_metrics)} model metrics")
    
    print("\nCreating comparison data...")
    comparison_data = create_comparison_data(all_metrics)
    
    print("Creating fairness breakdown...")
    fairness_data = create_fairness_data(all_metrics)
    
    print("Creating geographic data...")
    geographic_data = create_geographic_data(all_predictions)
    
    # Combine all data
    dashboard_data = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'num_models': len(all_metrics),
            'test_year': 2022
        },
        'model_comparison': comparison_data,
        'fairness_breakdown': fairness_data,
        'geographic_distribution': geographic_data,
        'predictions': all_predictions,
        'detailed_metrics': all_metrics
    }
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"\n✓ Dashboard data exported to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*70)
    print("✓ EXPORT COMPLETE")
    print("="*70)
    print("\nData structure:")
    print("  - model_comparison: Performance metrics for all models")
    print("  - fairness_breakdown: Per-group performance")
    print("  - geographic_distribution: State-level aggregates")
    print("  - predictions: Individual predictions per model")
    print("  - detailed_metrics: Full metrics including training time")
    
    print("\nReady for React dashboard integration!")

if __name__ == "__main__":
    main()