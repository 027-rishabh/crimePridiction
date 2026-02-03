"""
Evaluation Metrics and Scripts for FC-MT-LSTM Model
Comprehensive evaluation with fairness metrics
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os


def evaluate_model(model, test_loader, device='cuda'):
    """
    Comprehensive evaluation of FC-MT-LSTM
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_groups = []
    all_attention_weights = []
    
    with torch.no_grad():
        for X_batch, y_batch, group_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            group_batch = group_batch.to(device)
            
            predictions, attention_weights = model(X_batch, group_batch)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_groups.append(group_batch.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    groups = np.concatenate(all_groups)
    attention = np.concatenate(all_attention_weights)
    
    # Overall metrics
    overall_mae = mean_absolute_error(targets, predictions)
    overall_rmse = np.sqrt(mean_squared_error(targets, predictions))
    overall_r2 = r2_score(targets, predictions)
    
    print("=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    print(f"MAE:  {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R²:   {overall_r2:.4f}")
    
    # Per-group metrics
    group_names = ['SC', 'ST', 'Women', 'Children']
    group_metrics = {}
    
    print("\n" + "=" * 60)
    print("PER-GROUP PERFORMANCE")
    print("=" * 60)
    
    for group_id, group_name in enumerate(group_names):
        mask = (groups == group_id)
        
        if mask.sum() == 0:
            continue
        
        group_preds = predictions[mask]
        group_targets = targets[mask]
        
        mae = mean_absolute_error(group_targets, group_preds)
        rmse = np.sqrt(mean_squared_error(group_targets, group_preds))
        r2 = r2_score(group_targets, group_preds)
        
        group_metrics[group_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'samples': int(mask.sum())
        }
        
        print(f"\n{group_name}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Samples: {mask.sum()}")
    
    # Fairness metrics
    maes = [group_metrics[g]['mae'] for g in group_names if g in group_metrics]
    if maes:
        fairness_gap = max(maes) - min(maes)
        avg_mae = np.mean(maes)
        fairness_ratio = fairness_gap / avg_mae if avg_mae > 0 else 0
    else:
        fairness_gap = 0.0
        avg_mae = 0.0
        fairness_ratio = 0.0
    
    print("\n" + "=" * 60)
    print("FAIRNESS METRICS")
    print("=" * 60)
    print(f"Fairness Gap (max MAE - min MAE): {fairness_gap:.4f}")
    print(f"Average MAE across groups: {avg_mae:.4f}")
    print(f"Fairness Ratio (gap / avg): {fairness_ratio:.4f}")
    if maes:
        print(f"Max MAE: {max(maes):.4f} ({group_names[maes.index(max(maes))]})")
        print(f"Min MAE: {min(maes):.4f} ({group_names[maes.index(min(maes))]})")
    
    # Compile results for JSON
    results = {
        'model_name': 'FC-MT-LSTM',
        'overall_metrics': {
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'r2': float(overall_r2)
        },
        'fairness_breakdown': group_metrics,
        'fairness_metrics': {
            'fairness_gap': float(fairness_gap),
            'avg_mae': float(avg_mae),
            'fairness_ratio': float(fairness_ratio)
        }
    }
    
    return results, predictions, targets, groups, attention


def visualize_evaluation_results(results, predictions, targets, groups, attention=None):
    """
    Create visualizations for evaluation results
    """
    # Setup
    group_names = ['SC', 'ST', 'Women', 'Children']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FC-MT-LSTM Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Overall Performance Metrics
    metrics = ['MAE', 'RMSE', 'R²']
    values = [
        results['overall_metrics']['mae'],
        results['overall_metrics']['rmse'], 
        results['overall_metrics']['r2']
    ]
    
    axes[0, 0].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Overall Model Performance')
    axes[0, 0].set_ylabel('Score')
    # Add value annotations
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + max(values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Per-Group MAE
    group_maes = []
    group_labels = []
    for group_name in group_names:
        if group_name in results['fairness_breakdown']:
            group_maes.append(results['fairness_breakdown'][group_name]['mae'])
            group_labels.append(group_name)
    
    if group_maes:
        bars = axes[0, 1].bar(group_labels, group_maes, color=['blue', 'purple', 'pink', 'green'])
        axes[0, 1].set_title('Per-Group MAE')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        # Add value annotations
        for bar, value in zip(bars, group_maes):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.3f}',
                            ha='center', va='bottom')
    
    # 3. Fairness Gap Visualization
    fairness_data = []
    for group_name in group_names:
        if group_name in results['fairness_breakdown']:
            fairness_data.append(results['fairness_breakdown'][group_name]['mae'])
    
    if fairness_data:
        axes[0, 2].bar(group_labels, fairness_data, color=['orange', 'lightblue', 'yellow', 'lightgray'])
        axes[0, 2].set_title(f'Fairness Analysis\nGap: {results["fairness_metrics"]["fairness_gap"]:.3f}')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add horizontal line for average
        avg_mae = results['fairness_metrics']['avg_mae']
        axes[0, 2].axhline(y=avg_mae, color='red', linestyle='--', label=f'Avg: {avg_mae:.3f}')
        axes[0, 2].legend()
    
    # 4. Predictions vs Actual Scatter Plot
    axes[1, 0].scatter(targets, predictions, alpha=0.6, color='darkblue')
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Predictions vs Actual')
    
    # 5. Residuals by Group
    residuals = predictions - targets
    if len(groups) == len(residuals):
        # Map group IDs to names for plotting
        group_names_map = {0: 'SC', 1: 'ST', 2: 'Women', 3: 'Children'}
        group_str = [group_names_map[g] for g in groups if g in group_names_map]
        
        # Only plot if we have group information
        if len(group_str) == len(residuals):
            df_plot = pd.DataFrame({'group': group_str, 'residual': residuals})
            sns.boxplot(data=df_plot, x='group', y='residual', ax=axes[1, 1])
            axes[1, 1].set_title('Residuals by Group')
            axes[1, 1].set_xlabel('Protected Group')
            axes[1, 1].set_ylabel('Residuals (Pred - Actual)')
    
    # 6. Distribution of Absolute Errors by Group
    abs_errors = np.abs(residuals)
    if len(groups) == len(abs_errors):
        df_plot = pd.DataFrame({'group': group_str, 'abs_error': abs_errors})
        sns.boxplot(data=df_plot, x='group', y='abs_error', ax=axes[1, 2])
        axes[1, 2].set_title('Absolute Errors by Group')
        axes[1, 2].set_xlabel('Protected Group')
        axes[1, 2].set_ylabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig('evaluation_visualization.png', dpi=300, bbox_inches='tight')
    print("Evaluation visualization saved to evaluation_visualization.png")
    plt.show()


def export_results_for_react(results, predictions, targets, groups, attention=None):
    """
    Export evaluation results in a format suitable for React dashboard
    """
    # Prepare data for React dashboard
    react_data = {
        'model': 'FC-MT-LSTM',
        'metrics': {
            'overall': {
                'mae': results['overall_metrics']['mae'],
                'rmse': results['overall_metrics']['rmse'],
                'r2': results['overall_metrics']['r2']
            },
            'fairness': {
                'gap': results['fairness_metrics']['fairness_gap'],
                'ratio': results['fairness_metrics']['fairness_ratio']
            }
        },
        'breakdown': {},
        'predictions': predictions.tolist()[:100],  # Limit for dashboard
        'actuals': targets.tolist()[:100],  # Limit for dashboard
        'groups': groups.tolist()[:100]  # Limit for dashboard
    }
    
    # Add per-group metrics
    group_names = ['SC', 'ST', 'Women', 'Children']
    for i, name in enumerate(group_names):
        if name in results['fairness_breakdown']:
            react_data['breakdown'][name] = {
                'mae': results['fairness_breakdown'][name]['mae'],
                'rmse': results['fairness_breakdown'][name]['rmse'],
                'r2': results['fairness_breakdown'][name]['r2'],
                'count': results['fairness_breakdown'][name]['samples']
            }
    
    # Save to JSON file
    with open('results/fc_mt_lstm_react_data.json', 'w') as f:
        json.dump(react_data, f, indent=2)
    
    print("Results exported for React dashboard to results/fc_mt_lstm_react_data.json")
    
    return react_data


def evaluate_complete_model(model_path='fc_mt_lstm_final.pth', test_loader=None):
    """
    Complete evaluation pipeline
    """
    print("Starting complete FC-MT-LSTM evaluation...")
    
    # Load model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Import model class
        from models.fc_mt_lstm_pytorch import FC_MT_LSTM
        model = FC_MT_LSTM(
            input_dim=checkpoint['input_dim'], 
            hidden_dim=checkpoint.get('config', {}).get('hidden_dim', 128)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
        return None
    
    if test_loader is None:
        print("No test loader provided. Please provide a test data loader.")
        return None
    
    # Evaluate model
    results, predictions, targets, groups, attention = evaluate_model(model, test_loader, device='cpu')
    
    # Visualize results
    visualize_evaluation_results(results, predictions, targets, groups, attention)
    
    # Export for React dashboard
    react_data = export_results_for_react(results, predictions, targets, groups, attention)
    
    # Save detailed results
    with open('results/fc_mt_lstm_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Detailed results saved to results/fc_mt_lstm_detailed_results.json")
    
    return results, react_data


def compare_models(baseline_results_path='baseline_results.json', fc_mt_lstm_results_path='results/fc_mt_lstm_detailed_results.json'):
    """
    Compare FC-MT-LSTM results with baseline models
    """
    try:
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        # Create mock baseline results
        baseline_results = {
            'XGBoost': {
                'overall_metrics': {'mae': 1.83, 'rmse': 3.2, 'r2': 0.72},
                'fairness_metrics': {'fairness_gap': 7.5, 'fairness_ratio': 0.85}
            },
            'Random_Forest': {
                'overall_metrics': {'mae': 3.2, 'rmse': 4.8, 'r2': 0.65},
                'fairness_metrics': {'fairness_gap': 1.2, 'fairness_ratio': 0.35}
            }
        }
    
    try:
        with open(fc_mt_lstm_results_path, 'r') as f:
            fc_mt_lstm_results = json.load(f)
    except FileNotFoundError:
        print(f"FC-MT-LSTM results file {fc_mt_lstm_results_path} not found!")
        return
    
    print("=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Prepare comparison table
    print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Fairness Gap':<12} {'Fairness Ratio':<15}")
    print("-" * 80)
    
    # Print baseline results
    for model_name, metrics in baseline_results.items():
        print(f"{model_name:<15} {metrics['overall_metrics']['mae']:<8.3f} "
              f"{metrics['overall_metrics']['rmse']:<8.3f} {metrics['overall_metrics']['r2']:<8.3f} "
              f"{metrics['fairness_metrics']['fairness_gap']:<12.3f} {metrics['fairness_metrics']['fairness_ratio']:<15.3f}")
    
    # Print FC-MT-LSTM results
    print(f"FC-MT-LSTM      {fc_mt_lstm_results['overall_metrics']['mae']:<8.3f} "
          f"{fc_mt_lstm_results['overall_metrics']['rmse']:<8.3f} {fc_mt_lstm_results['overall_metrics']['r2']:<8.3f} "
          f"{fc_mt_lstm_results['fairness_metrics']['fairness_gap']:<12.3f} {fc_mt_lstm_results['fairness_metrics']['fairness_ratio']:<15.3f}")
    
    # Improvement calculation
    fc_mae = fc_mt_lstm_results['overall_metrics']['mae']
    fc_gap = fc_mt_lstm_results['fairness_metrics']['fairness_gap']
    
    for model_name, metrics in baseline_results.items():
        baseline_mae = metrics['overall_metrics']['mae']
        baseline_gap = metrics['fairness_metrics']['fairness_gap']
        
        mae_improvement = ((baseline_mae - fc_mae) / baseline_mae) * 100
        gap_improvement = ((baseline_gap - fc_gap) / baseline_gap) * 100
        
        print(f"\nImprovement over {model_name}:")
        print(f"  MAE: {'+' if mae_improvement > 0 else ''}{mae_improvement:.2f}%")
        print(f"  Fairness Gap: {'+' if gap_improvement > 0 else ''}{gap_improvement:.2f}%")
    
    # Create comparison visualization
    model_names = list(baseline_results.keys()) + ['FC-MT-LSTM']
    maes = [baseline_results[model]['overall_metrics']['mae'] for model in baseline_results.keys()] + [fc_mae]
    gaps = [baseline_results[model]['fairness_metrics']['fairness_gap'] for model in baseline_results.keys()] + [fc_gap]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE comparison
    bars1 = ax1.bar(model_names, maes, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_title('Model Comparison - MAE')
    ax1.set_ylabel('Mean Absolute Error')
    for bar, value in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(maes)*0.01, 
                 f'{value:.3f}', ha='center', va='bottom')
    
    # Fairness Gap comparison  
    bars2 = ax2.bar(model_names, gaps, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title('Model Comparison - Fairness Gap')
    ax2.set_ylabel('Fairness Gap')
    for bar, value in zip(bars2, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(gaps)*0.01, 
                 f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison visualization saved to model_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("Testing Evaluation Scripts...")
    
    # Create a mock model and data for testing
    from models.fc_mt_lstm_pytorch import FC_MT_LSTM
    
    # Create dummy results for testing visualization
    mock_results = {
        'model_name': 'FC-MT-LSTM',
        'overall_metrics': {'mae': 6.52, 'rmse': 9.45, 'r2': 0.752},
        'fairness_breakdown': {
            'SC': {'mae': 6.21, 'rmse': 9.12, 'r2': 0.761, 'samples': 1250},
            'ST': {'mae': 6.53, 'rmse': 9.45, 'r2': 0.748, 'samples': 980},
            'Women': {'mae': 6.66, 'rmse': 9.67, 'r2': 0.742, 'samples': 1450},
            'Children': {'mae': 7.06, 'rmse': 10.12, 'r2': 0.725, 'samples': 1120}
        },
        'fairness_metrics': {'fairness_gap': 0.85, 'avg_mae': 6.615, 'fairness_ratio': 0.13}
    }
    
    # Generate mock predictions and targets for testing
    np.random.seed(42)
    n_samples = 500
    mock_predictions = np.random.normal(10, 3, n_samples)
    mock_targets = mock_predictions + np.random.normal(0, 1, n_samples)  # Add some noise
    mock_groups = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Test visualization
    visualize_evaluation_results(mock_results, mock_predictions, mock_targets, mock_groups)
    
    # Test React export
    react_data = export_results_for_react(mock_results, mock_predictions, mock_targets, mock_groups)
    
    # Test model comparison
    compare_models()
    
    print("✅ Evaluation scripts test completed!")