#!/usr/bin/env python3
"""
Master script to train all 6 baseline models
"""

import subprocess
import time
from pathlib import Path
import json

# Model scripts
MODELS = [
    ("SARIMA", "models/baseline_01_sarima.py"),
    ("Prophet", "models/baseline_02_prophet.py"),
    ("Random Forest", "models/baseline_03_random_forest.py"),
    ("XGBoost", "models/baseline_04_xgboost.py"),
    ("CNN-LSTM", "models/baseline_05_cnn_lstm.py"),
    ("Transformer", "models/baseline_06_transformer.py"),
]

def run_model(name, script_path):
    """Run a single model script"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python3', script_path],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {name} completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False, elapsed

def load_metrics(results_dir):
    """Load all model metrics"""
    metrics_dir = Path(results_dir) / "fairness_metrics"
    
    all_metrics = {}
    for model_name, _ in MODELS:
        filename = model_name.lower().replace(" ", "_").replace("-", "_") + "_fairness.json"
        filepath = metrics_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                all_metrics[model_name] = json.load(f)
    
    return all_metrics

def print_comparison_table(all_metrics):
    """Print comparison table of all models"""
    print("\n" + "="*90)
    print("BASELINE MODELS COMPARISON")
    print("="*90)
    
    print(f"\n{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Fair.Gap':<10} {'Time(min)':<10}")
    print("-" * 90)
    
    for model_name in [m[0] for m in MODELS]:
        if model_name in all_metrics:
            metrics = all_metrics[model_name]
            overall = metrics['overall']
            fairness_gap = metrics['fairness_gap']
            train_time = metrics.get('training_time_minutes', 0)
            
            print(f"{model_name:<15} {overall['mae']:<8.2f} {overall['rmse']:<8.2f} "
                  f"{overall['r2']:<8.4f} {fairness_gap:<10.2f} {train_time:<10.1f}")
    
    print("="*90)

def print_fairness_breakdown(all_metrics):
    """Print per-group fairness breakdown"""
    print("\n" + "="*90)
    print("FAIRNESS BREAKDOWN BY PROTECTED GROUP")
    print("="*90)
    
    # Updated to include all protected groups: SC, ST, General, Women, Children
    groups = ['SC', 'ST', 'General', 'Women', 'Children']
    
    for group in groups:
        print(f"\n{group} Group:")
        print(f"  {'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Count':<8}")
        print(f"  {'-'*60}")
        
        for model_name in [m[0] for m in MODELS]:
            if model_name in all_metrics:
                by_group = all_metrics[model_name].get('by_group', {})
                if group in by_group:
                    gm = by_group[group]
                    print(f"  {model_name:<15} {gm['mae']:<8.2f} {gm['rmse']:<8.2f} "
                          f"{gm['r2']:<8.4f} {gm['count']:<8}")

def main():
    print("="*70)
    print("TRAINING ALL BASELINE MODELS")
    print("="*70)
    print(f"\nTotal models: {len(MODELS)}")
    print("Estimated total time: ~60 minutes")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Track results
    results = {}
    total_start = time.time()
    
    # Run each model
    for model_name, script_path in MODELS:
        success, elapsed = run_model(model_name, script_path)
        results[model_name] = {
            'success': success,
            'time': elapsed
        }
    
    total_elapsed = time.time() - total_start
    
    # Load all metrics
    print("\n" + "="*70)
    print("LOADING RESULTS")
    print("="*70)
    
    all_metrics = load_metrics(results_dir)
    
    # Print comparison
    print_comparison_table(all_metrics)
    print_fairness_breakdown(all_metrics)
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    
    print(f"\nTotal execution time: {total_elapsed/60:.1f} minutes")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nModels completed: {successful}/{len(MODELS)}")
    
    for model_name, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {model_name:<15} ({result['time']/60:.1f} min)")
    
    # Save summary
    summary = {
        'total_time_minutes': total_elapsed / 60,
        'models_completed': successful,
        'models_total': len(MODELS),
        'individual_results': results,
        'metrics_summary': {
            name: {
                'mae': metrics['overall']['mae'],
                'r2': metrics['overall']['r2'],
                'fairness_gap': metrics['fairness_gap']
            }
            for name, metrics in all_metrics.items()
        }
    }
    
    summary_file = results_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("✓ ALL BASELINE TRAINING COMPLETE!")
    print("="*70)
    print("\nNext step: Export results for React dashboard")
    print("  Command: python3 export_results_for_react.py")

if __name__ == "__main__":
    main()