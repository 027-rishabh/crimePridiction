"""
Training Loop for FC-MT-LSTM Model
Implements complete training with fairness constraints
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os
import json


# Configuration
CONFIG = {
    'sequence_length': 12,
    'hidden_dim': 128,
    'batch_size': 128,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'lambda_fairness': 1.0,
    'early_stopping_patience': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, loss_fn, optimizer, 
                scheduler, early_stopping, num_epochs=200, device='cuda'):
    """
    Complete training loop with validation and checkpointing
    """
    model.to(device)
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    fairness_gaps = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_fairness_loss = 0.0
        train_mae = 0.0
        train_batches = 0
        
        for batch_idx, (X_batch, y_batch, group_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            group_batch = group_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, attention_weights = model(X_batch, group_batch)
            
            # Calculate loss
            loss, mse, fairness_penalty = loss_fn(predictions, y_batch, group_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mse_loss += mse.item()
            train_fairness_loss += fairness_penalty.item()
            train_mae += torch.abs(predictions - y_batch).mean().item()
            train_batches += 1
        
        # Average training metrics
        if train_batches > 0:
            train_loss /= train_batches
            train_mse_loss /= train_batches
            train_fairness_loss /= train_batches
            train_mae /= train_batches
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []
        all_groups = []
        
        with torch.no_grad():
            for X_batch, y_batch, group_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                group_batch = group_batch.to(device)
                
                predictions, attention_weights = model(X_batch, group_batch)
                
                loss, mse, fairness_penalty = loss_fn(predictions, y_batch, group_batch)
                
                val_loss += loss.item()
                val_mae += torch.abs(predictions - y_batch).mean().item()
                val_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_groups.append(group_batch.cpu().numpy())
        
        # Average validation metrics
        if val_batches > 0:
            val_loss /= val_batches
            val_mae /= val_batches
        
        # Calculate fairness gap
        if all_predictions:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            all_groups = np.concatenate(all_groups)
            
            group_maes = []
            for group in range(4):
                mask = (all_groups == group)
                if mask.sum() > 0:
                    group_mae = np.abs(all_predictions[mask] - all_targets[mask]).mean()
                    group_maes.append(group_mae)
            
            fairness_gap = max(group_maes) - min(group_maes) if group_maes else 0.0
        else:
            fairness_gap = 0.0
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        fairness_gaps.append(fairness_gap)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} (MSE: {train_mse_loss:.4f}, Fairness: {train_fairness_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Fairness Gap: {fairness_gap:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'fairness_gap': fairness_gap
            }, 'best_fc_mt_lstm.pth')
            print("  ✓ Saved best model")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if os.path.exists('best_fc_mt_lstm.pth'):
        checkpoint = torch.load('best_fc_mt_lstm.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses, train_maes, val_maes, fairness_gaps


def plot_training_curves(train_losses, val_losses, train_maes, val_maes, fairness_gaps):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE curves
    axes[0, 1].plot(train_maes, label='Train MAE')
    axes[0, 1].plot(val_maes, label='Val MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Fairness gap
    axes[1, 0].plot(fairness_gaps, label='Fairness Gap', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Fairness Gap')
    axes[1, 0].set_title('Fairness Gap Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    # Since we don't track learning rate directly in the training loop,
    # we'll skip this plot for now
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Training curves saved to training_curves.png")


def run_training_pipeline(model, train_loader, val_loader, input_dim):
    """
    Complete training pipeline
    """
    print("Setting up training...")
    
    # Import our loss function and early stopping
    from models.fc_mt_lstm_pytorch import FairnessConstrainedLoss, EarlyStopping
    
    loss_fn = FairnessConstrainedLoss(lambda_fairness=CONFIG['lambda_fairness'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # Start training
    print("\nStarting training...\n")
    
    model, train_losses, val_losses, train_maes, val_maes, fairness_gaps = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, 
        scheduler, early_stopping, CONFIG['num_epochs'], CONFIG['device']
    )
    
    print("\nTraining completed!")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_maes, val_maes, fairness_gaps)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'input_dim': input_dim
    }, 'fc_mt_lstm_final.pth')
    
    print("Final model saved to fc_mt_lstm_final.pth")
    
    return model, train_losses, val_losses, train_maes, val_maes, fairness_gaps


def train_complete_model(data_path='data/crime_data_processed.csv'):
    """
    Complete training pipeline from data loading to model training
    """
    print("Starting complete FC-MT-LSTM training pipeline...")
    
    # Set random seed
    set_seed(CONFIG['random_seed'])
    
    from data.preparation_pipeline import prepare_crime_data_pipeline
    
    # Load and prepare data
    prepared_data = prepare_crime_data_pipeline(data_path, sequence_length=CONFIG['sequence_length'])
    
    if prepared_data['train_loader'] is None:
        print("Error: Could not create data loaders. Please check your data.")
        return
    
    # Initialize model
    input_dim = prepared_data['X_train'].shape[2] if prepared_data['X_train'].size > 0 else 20  # Default to 20 features
    
    from models.fc_mt_lstm_pytorch import FC_MT_LSTM
    model = FC_MT_LSTM(input_dim, CONFIG['hidden_dim'])
    
    print(f"Model initialized with {input_dim} input features")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {CONFIG['device']}")
    
    # Run training
    trained_model, train_losses, val_losses, train_maes, val_maes, fairness_gaps = run_training_pipeline(
        model, 
        prepared_data['train_loader'], 
        prepared_data['val_loader'], 
        input_dim
    )
    
    return trained_model, prepared_data


if __name__ == "__main__":
    print("Testing Training Pipeline...")
    
    # Create synthetic data for testing
    from data.preparation_pipeline import create_synthetic_data
    
    temp_file, df = create_synthetic_data(n_samples=2000, n_features=15)
    
    # Set up model and test training
    df = df.rename(columns={'crime_rate': 'crime_rate'})
    
    # Create a smaller training setup for testing
    from data.preparation_pipeline import prepare_crime_data_pipeline
    
    prepared_data = prepare_crime_data_pipeline(temp_file, sequence_length=6, target_col='crime_rate')
    
    if prepared_data['X_train'].size > 0:
        input_dim = prepared_data['X_train'].shape[2]
        
        from models.fc_mt_lstm_pytorch import FC_MT_LSTM, FairnessConstrainedLoss, EarlyStopping
        
        # Create a smaller model for testing
        model = FC_MT_LSTM(input_dim, hidden_dim=32)  # Smaller model for testing
        
        # Set up training components
        loss_fn = FairnessConstrainedLoss(lambda_fairness=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10)
        
        # Run a few epochs for testing
        print("Running test training (few epochs)...")
        model, train_losses, val_losses, train_maes, val_maes, fairness_gaps = train_model(
            model, 
            prepared_data['train_loader'], 
            prepared_data['val_loader'], 
            loss_fn, 
            optimizer, 
            scheduler, 
            early_stopping, 
            num_epochs=5,  # Just 5 epochs for testing
            device=CONFIG['device']
        )
        
        print("✅ Training pipeline test completed!")
    else:
        print("Could not prepare data for testing")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)