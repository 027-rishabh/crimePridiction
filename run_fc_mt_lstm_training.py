#!/usr/bin/env python3
"""
Train FC-MT-LSTM Model with Proper Data Splits
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, '.')

# Get the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class CrimeDataset(Dataset):
    """
    Dataset class for crime data with temporal sequences
    """
    def __init__(self, X_data, y_data, group_data, sequence_length=None):
        self.X_data = torch.FloatTensor(X_data)  # Shape: (num_samples, seq_length, features)
        self.y_data = torch.FloatTensor(y_data)  # Shape: (num_samples, 1)
        self.group_data = torch.LongTensor(group_data)  # Shape: (num_samples,)
        
    def __len__(self):
        return len(self.X_data)  # Already created sequences
    
    def __getitem__(self, idx):
        # Return the pre-created sequence, target, and group
        x_seq = self.X_data[idx]  # Shape: (seq_length, features)
        y_val = self.y_data[idx]  # Shape: (1,)
        group_val = self.group_data[idx]  # Scalar
        
        return x_seq, y_val, group_val

def load_and_preprocess_data():
    """
    Load and preprocess the train/test data from splits
    """
    print("Loading data from splits...")
    
    # Load training and testing data
    train_df = pd.read_csv('data/splits/train_data.csv')
    test_df = pd.read_csv('data/splits/test_data.csv')
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Identify feature columns (exclude ID, target, and non-feature columns)
    exclude_cols = ['id', 'year', 'state_name', 'district_name', 'protected_group', 'group_encoded', 'registration_circles']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols and not col.startswith('group_')]
    
    target_col = 'total_crimes'
    
    print(f"Using {len(feature_cols)} features for training")
    print(f"Target column: {target_col}")
    
    # Handle categorical columns by encoding them
    # Create copies for processing
    train_df_processed = train_df.copy()
    test_df_processed = test_df.copy()
    
    # Encode categorical features
    categorical_features = ['state_name', 'district_name', 'registration_circles']
    for cat_feature in categorical_features:
        if cat_feature in train_df_processed.columns:
            # Create a combined list of categories from both train and test to ensure consistency
            all_categories = list(set(train_df_processed[cat_feature].astype(str).unique()) | 
                                  set(test_df_processed[cat_feature].astype(str).unique()))
            
            # Create a mapping
            category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
            
            # Apply the mapping
            train_df_processed[cat_feature] = train_df_processed[cat_feature].astype(str).map(category_mapping)
            test_df_processed[cat_feature] = test_df_processed[cat_feature].astype(str).map(category_mapping)
            
            # Fill any NaN values with -1 (for unseen categories)
            train_df_processed[cat_feature] = train_df_processed[cat_feature].fillna(-1)
            test_df_processed[cat_feature] = test_df_processed[cat_feature].fillna(-1)
    
    # Prepare features and targets
    X_train_raw = train_df_processed[feature_cols].values.astype(np.float32)
    y_train_raw = train_df_processed[target_col].values.astype(np.float32)
    groups_train_raw = train_df_processed['group_encoded'].values.astype(int)
    
    X_test_raw = test_df_processed[feature_cols].values.astype(np.float32)
    y_test_raw = test_df_processed[target_col].values.astype(np.float32)
    groups_test_raw = test_df_processed['group_encoded'].values.astype(int)
    
    print(f"Features after processing: {X_train_raw.shape[1]}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Save scaler for later use
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Feature scaling completed")
    
    return X_train, y_train_raw, groups_train_raw, X_test, y_test_raw, groups_test_raw, feature_cols

def create_sequences(X, y, groups, sequence_length=6):
    """
    Create temporal sequences for LSTM training
    """
    sequences = []
    targets = []
    group_labels = []
    
    for i in range(len(X) - sequence_length + 1):
        # Ensure we have consistent group labels in the sequence
        if len(set(groups[i:i+sequence_length])) == 1:
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length-1])
            group_labels.append(groups[i])
    
    return np.array(sequences), np.array(targets), np.array(group_labels)

def train_fc_mt_lstm():
    """
    Train the FC-MT-LSTM model with the proper data
    """
    print("="*70)
    print("TRAINING FC-MT-LSTM MODEL WITH PROPER DATA SPLITS")
    print("="*70)
    
    # Load and preprocess data
    X_train, y_train, groups_train, X_test, y_test, groups_test, feature_cols = load_and_preprocess_data()
    
    # Create temporal sequences
    print("Creating temporal sequences...")
    sequence_length = 6  # Use 6 time steps
    X_train_seq, y_train_seq, groups_train_seq = create_sequences(X_train, y_train, groups_train, sequence_length)
    X_test_seq, y_test_seq, groups_test_seq = create_sequences(X_test, y_test, groups_test, sequence_length)
    
    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    
    # Create datasets and dataloaders
    train_dataset = CrimeDataset(X_train_seq, y_train_seq.reshape(-1, 1), groups_train_seq, sequence_length)
    test_dataset = CrimeDataset(X_test_seq, y_test_seq.reshape(-1, 1), groups_test_seq, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    # Import the model
    from models.fc_mt_lstm_pytorch import FC_MT_LSTM, FairnessConstrainedLoss, EarlyStopping
    
    # Initialize model
    input_dim = X_train_seq.shape[2]  # Use the feature dimension from sequences
    model = FC_MT_LSTM(input_dim, hidden_dim=128).to(DEVICE)
    
    # Loss function with fairness constraints
    loss_fn = FairnessConstrainedLoss(lambda_fairness=1.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    print(f"Model initialized with {input_dim} input features")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_X, batch_y, batch_groups in train_loader:
            # Debug: print tensor shapes
            # print(f"Batch_X shape: {batch_X.shape}, Batch_y shape: {batch_y.shape}, Batch_groups shape: {batch_groups.shape}")
            
            # Ensure tensors are dense and have proper types
            batch_X = batch_X.to(DEVICE).float()
            batch_y = batch_y.to(DEVICE).float()
            batch_groups = batch_groups.to(DEVICE).long()
            
            # Double-check shape before forwarding
            if batch_X.dim() != 3:
                print(f"ERROR: Expected 3D tensor but got {batch_X.dim()}D tensor with shape {batch_X.shape}")
                print("This indicates an issue with the dataset implementation")
                break
                
            optimizer.zero_grad()
            predictions, attention = model(batch_X, batch_groups)
            loss, mse, fairness_penalty = loss_fn(predictions, batch_y, batch_groups)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_groups in test_loader:
                # Debug: print tensor shapes
                # print(f"Val Batch_X shape: {batch_X.shape}, Batch_y shape: {batch_y.shape}, Batch_groups shape: {batch_groups.shape}")
                
                # Ensure tensors are dense and have proper types
                batch_X = batch_X.to(DEVICE).float()
                batch_y = batch_y.to(DEVICE).float()
                batch_groups = batch_groups.to(DEVICE).long()
                
                # Double-check shape before forwarding
                if batch_X.dim() != 3:
                    print(f"ERROR: Expected 3D tensor but got {batch_X.dim()}D tensor with shape {batch_X.shape}")
                    print("This indicates an issue with the dataset implementation")
                    continue
                
                predictions, attention = model(batch_X, batch_groups)
                loss, mse, fairness_penalty = loss_fn(predictions, batch_y, batch_groups)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Training completed!")
    
    # Save the model
    model_path = 'results/fc_mt_lstm_model.pth'
    os.makedirs('results', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'input_dim': input_dim,
        'hidden_dim': 128
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Evaluate the model
    evaluate_model(model, test_loader, DEVICE, feature_cols, sequence_length)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, feature_cols, sequence_length):
    """
    Evaluate the trained model and save results
    """
    print("\nEvaluating model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_groups = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_groups in test_loader:
            batch_X, batch_y, batch_groups = batch_X.to(device), batch_y.to(device), batch_groups.to(device)
            
            predictions, attention = model(batch_X, batch_groups)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_groups.append(batch_groups.cpu().numpy())
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_groups = np.concatenate(all_groups)
    
    # Calculate overall metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_mse = mean_squared_error(all_targets, all_predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(all_targets, all_predictions)
    
    print(f"\nOverall Results:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R²: {overall_r2:.4f}")
    
    # Calculate per-group metrics
    group_names = ['SC', 'ST', 'Women', 'Children']
    group_metrics = {}
    
    print(f"\nPer-Group Results:")
    for group_id in range(4):
        mask = (all_groups.flatten() == group_id)
        if np.any(mask):
            group_preds = all_predictions[mask]
            group_targets = all_targets[mask]
            
            group_mae = mean_absolute_error(group_targets, group_preds)
            group_mse = mean_squared_error(group_targets, group_preds)
            group_rmse = np.sqrt(group_mse)
            group_r2 = r2_score(group_targets, group_preds)
            
            group_metrics[group_names[group_id]] = {
                'mae': float(group_mae),
                'rmse': float(group_rmse),
                'r2': float(group_r2),
                'count': int(mask.sum()),
                'samples': int(mask.sum())
            }
            
            print(f"{group_names[group_id]}: MAE={group_mae:.4f}, RMSE={group_rmse:.4f}, R²={group_r2:.4f}, samples={mask.sum()}")
    
    # Calculate fairness metrics
    maes = [group_metrics[name]['mae'] for name in group_names if name in group_metrics]
    if len(maes) > 1:
        fairness_gap = max(maes) - min(maes)
        avg_mae = np.mean(maes)
        fairness_ratio = fairness_gap / avg_mae if avg_mae > 0 else 0
        
        print(f"\nFairness Metrics:")
        print(f"Fairness Gap: {fairness_gap:.4f}")
        print(f"Fairness Ratio: {fairness_ratio:.4f}")
    else:
        fairness_gap = 0.0
        fairness_ratio = 0.0
    
    # Prepare results dictionary
    results = {
        'model': 'FC-MT-LSTM',
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'mse': float(overall_mse),
            'r2': float(overall_r2)
        },
        'fairness_metrics': {
            'fairness_gap': float(fairness_gap),
            'fairness_ratio': float(fairness_ratio)
        },
        'breakdown': group_metrics,
        'total_samples': int(len(all_targets)),
        'feature_count': len(feature_cols),
        'sequence_length': sequence_length
    }
    
    # Save results
    results_path = 'results/fc_mt_lstm_results.json'
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Also save results in the format expected by the dashboard
    dashboard_results = {
        'model': 'FC-MT-LSTM',
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'r2': float(overall_r2),
        'fairness_gap': float(fairness_gap),
        'fairness_ratio': float(fairness_ratio),
        'training_time': 0  # Placeholder
    }
    
    # Load existing dashboard data and update
    dashboard_data_path = 'results/react_dashboard_data.json'
    try:
        with open(dashboard_data_path, 'r') as f:
            dashboard_data = json.load(f)
    except FileNotFoundError:
        dashboard_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_models": 0,
                "test_year": 2022
            },
            "model_comparison": [],
            "fairness_breakdown": {},
            "geographic_distribution": []
        }
    
    # Update or add FC-MT-LSTM results
    updated = False
    for i, model_result in enumerate(dashboard_data["model_comparison"]):
        if model_result["model"] == "FC-MT-LSTM":
            dashboard_data["model_comparison"][i] = dashboard_results
            updated = True
            break
    
    if not updated:
        dashboard_data["model_comparison"].append(dashboard_results)
    
    # Update fairness breakdown
    dashboard_data["fairness_breakdown"]["FC-MT-LSTM"] = {
        'SC': {
            'mae': group_metrics.get('SC', {}).get('mae', 0.0),
            'rmse': group_metrics.get('SC', {}).get('rmse', 0.0),
            'r2': group_metrics.get('SC', {}).get('r2', 0.0),
            'count': group_metrics.get('SC', {}).get('count', 0)
        },
        'ST': {
            'mae': group_metrics.get('ST', {}).get('mae', 0.0),
            'rmse': group_metrics.get('ST', {}).get('rmse', 0.0),
            'r2': group_metrics.get('ST', {}).get('r2', 0.0),
            'count': group_metrics.get('ST', {}).get('count', 0)
        },
        'Women': {
            'mae': group_metrics.get('Women', {}).get('mae', 0.0),
            'rmse': group_metrics.get('Women', {}).get('rmse', 0.0),
            'r2': group_metrics.get('Women', {}).get('r2', 0.0),
            'count': group_metrics.get('Women', {}).get('count', 0)
        },
        'Children': {
            'mae': group_metrics.get('Children', {}).get('mae', 0.0),
            'rmse': group_metrics.get('Children', {}).get('rmse', 0.0),
            'r2': group_metrics.get('Children', {}).get('r2', 0.0),
            'count': group_metrics.get('Children', {}).get('count', 0)
        }
    }
    
    # Save updated dashboard data
    with open(dashboard_data_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"Updated dashboard data saved to {dashboard_data_path}")
    
    # Also update the public directory for the React dashboard
    public_path = 'dashboard/public/results/react_dashboard_data.json'
    os.makedirs(os.path.dirname(public_path), exist_ok=True)
    with open(public_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"Updated public dashboard data saved to {public_path}")
    
    return results

if __name__ == "__main__":
    # Activate the virtual environment
    python_path = "crimeprediction_env/bin/python" if os.path.exists("crimeprediction_env") else sys.executable
    
    print(f"Python interpreter: {python_path}")
    
    # Run training
    model, train_losses, val_losses = train_fc_mt_lstm()
    
    print("\nTraining and evaluation completed successfully!")
    print("Results have been saved to the results directory.")