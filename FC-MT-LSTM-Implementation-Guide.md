# FC-MT-LSTM Implementation Guide
## Fairness-Constrained Multi-Task Learning for Crime Prediction

---

## 📑 TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Data Preparation](#data-preparation)
4. [Implementation Details](#implementation-details)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Metrics](#evaluation-metrics)
7. [React Dashboard Specification](#react-dashboard-specification)
8. [Timeline & Milestones](#timeline--milestones)

---

## 1. PROJECT OVERVIEW

### 1.1 Problem Statement

**Core Challenge:**
Predict crime rates for 4 protected groups (SC, ST, Women, Children) while ensuring:
- High prediction accuracy (low MAE/RMSE)
- Fair predictions across all groups (low fairness gap)
- Interpretable decisions (attention mechanism)

**Current Baseline Results:**
- XGBoost: Best accuracy (MAE 1.83), Poor fairness (gap ~6-8)
- Random Forest: Good fairness (gap ~1.0), Moderate accuracy
- Other models: Poor on both metrics

**Research Goal:**
Develop FC-MT-LSTM to achieve Pareto-optimal solution:
- Target MAE: ~6.5-7.0 (comparable to XGBoost)
- Target Fairness Gap: <1.0 (better than all baselines)

### 1.2 Model Innovation

**FC-MT-LSTM (Fairness-Constrained Multi-Task LSTM) consists of:**

1. **Shared Encoder:** CNN-LSTM learns common crime patterns across all groups
2. **Multi-Task Decoders:** Separate decoder per protected group prevents domination
3. **Fairness Loss:** Penalizes MAE differences between groups
4. **Attention Mechanism:** Provides interpretability for predictions

**Why This Works:**
- Shared encoder: Leverages data from all groups (more training data)
- Separate decoders: Prevents majority group from dominating
- Fairness loss: Explicitly optimizes for equal performance
- Attention: Shows which features (crime types, regions) drive predictions

---

## 2. MODEL ARCHITECTURE

### 2.1 High-Level Architecture

```
Input Features (X)
    ↓
[Shared CNN-LSTM Encoder]
    ├── Conv1D layers (extract spatial patterns)
    ├── LSTM layers (capture temporal dependencies)
    └── Attention mechanism (highlight important features)
    ↓
Shared Representation (h)
    ↓
┌────────┬────────┬────────┬────────┐
│   SC   │   ST   │ Women  │Children│  ← 4 Separate Decoders
│Decoder │Decoder │Decoder │Decoder │
└────────┴────────┴────────┴────────┘
    ↓       ↓       ↓       ↓
  ŷ_SC   ŷ_ST   ŷ_Women ŷ_Children  ← Predictions
```

### 2.2 Detailed Component Specifications

#### 2.2.1 Input Layer

**Input Shape:** `(batch_size, sequence_length, num_features)`

**Features to Include:**
```python
temporal_features = [
    'year', 'month', 'quarter',
    'crime_rate_lag_1', 'crime_rate_lag_2', 'crime_rate_lag_3',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
    'trend', 'seasonality'
]

crime_type_features = [
    'murder', 'rape', 'kidnapping', 'dacoity', 'robbery',
    'burglary', 'theft', 'riots', 'arson',
    'hurt', 'cheating', 'counterfeiting'
]

regional_features = [
    'state_encoded',  # One-hot or embedding
    'urban_rural_flag',
    'population_density',
    'literacy_rate',
    'poverty_index'
]

group_indicator = [
    'protected_group_encoded'  # SC=0, ST=1, Women=2, Children=3
]
```

**Total Features:** ~35-40 features

**Sequence Length:** 12 months (1 year of historical data)

**Preprocessing:**
```python
# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_lstm = X_scaled.reshape((-1, sequence_length, num_features))
```

#### 2.2.2 Shared CNN-LSTM Encoder

**Purpose:** Learn common crime patterns shared across all groups

**Architecture:**

```python
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SharedEncoder, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # LSTM layers for temporal dependencies
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm1(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2) due to bidirectional
        
        # Attention
        attended, attention_weights = self.attention(lstm_out)
        # attended shape: (batch, hidden_dim * 2)
        
        return attended, attention_weights
```

**Hyperparameters:**
- Conv1D filters: 64, 128
- Kernel size: 3
- LSTM hidden units: 128 (256 with bidirectional)
- LSTM layers: 2
- Dropout: 0.3
- Activation: ReLU

#### 2.2.3 Attention Mechanism

**Purpose:** Identify which time steps and features are most important

**Implementation:**

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output)
        # attention_scores shape: (batch, seq_len, 1)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        # attended shape: (batch, hidden_dim)
        
        return attended, attention_weights.squeeze(-1)
```

**Output:**
- Attended representation: (batch, hidden_dim)
- Attention weights: (batch, seq_len) - for visualization

#### 2.2.4 Multi-Task Decoders

**Purpose:** Separate prediction head for each protected group

**Architecture:**

```python
class GroupDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GroupDecoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 1)  # Single output: crime rate
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)  # No activation (regression)
        
        return x
```

**4 Separate Decoders:**
```python
class MultiTaskDecoders(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskDecoders, self).__init__()
        
        self.sc_decoder = GroupDecoder(input_dim)
        self.st_decoder = GroupDecoder(input_dim)
        self.women_decoder = GroupDecoder(input_dim)
        self.children_decoder = GroupDecoder(input_dim)
        
    def forward(self, shared_repr, group_labels):
        # shared_repr: (batch, hidden_dim)
        # group_labels: (batch,) with values 0, 1, 2, 3
        
        batch_size = shared_repr.shape[0]
        predictions = torch.zeros(batch_size, 1)
        
        for i in range(batch_size):
            if group_labels[i] == 0:  # SC
                predictions[i] = self.sc_decoder(shared_repr[i:i+1])
            elif group_labels[i] == 1:  # ST
                predictions[i] = self.st_decoder(shared_repr[i:i+1])
            elif group_labels[i] == 2:  # Women
                predictions[i] = self.women_decoder(shared_repr[i:i+1])
            elif group_labels[i] == 3:  # Children
                predictions[i] = self.children_decoder(shared_repr[i:i+1])
        
        return predictions
```

**Decoder Hyperparameters:**
- Layer 1: 256 → 64 units
- Layer 2: 64 → 32 units
- Layer 3: 32 → 1 unit (output)
- Dropout: 0.2
- Activation: ReLU (except output layer)

#### 2.2.5 Complete FC-MT-LSTM Model

```python
class FC_MT_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FC_MT_LSTM, self).__init__()
        
        self.encoder = SharedEncoder(input_dim, hidden_dim)
        self.decoders = MultiTaskDecoders(hidden_dim * 2)  # *2 for bidirectional
        
    def forward(self, x, group_labels):
        # x: (batch, seq_len, features)
        # group_labels: (batch,)
        
        shared_repr, attention_weights = self.encoder(x)
        predictions = self.decoders(shared_repr, group_labels)
        
        return predictions, attention_weights
```

### 2.3 Model Size Estimation

**Parameter Count:**

```
Shared Encoder:
- Conv1D layers: ~20K params
- LSTM layers: ~400K params
- Attention: ~130K params
Total Encoder: ~550K params

Decoders (4x):
- Per decoder: ~17K params
- Total: ~68K params

Total Model: ~620K parameters
```

**Memory Requirements:**
- Model size: ~2.5 MB
- Training batch memory: ~500 MB (batch_size=128)
- Total training memory: ~3-4 GB GPU

---

## 3. DATA PREPARATION

### 3.1 Data Collection Requirements

**Required Data Files:**

```
data/
├── crime_data_raw.csv          # Main crime statistics
├── demographic_data.csv        # Population, literacy, etc.
├── regional_metadata.csv       # State, district, urban/rural
└── protected_group_mapping.csv # SC, ST, Women, Children labels
```

**crime_data_raw.csv structure:**
```csv
state,district,year,month,protected_group,murder,rape,kidnapping,...,total_crimes
Uttar Pradesh,Dadri,2018,1,SC,5,3,8,...,150
Uttar Pradesh,Dadri,2018,1,ST,2,1,4,...,45
Uttar Pradesh,Dadri,2018,1,Women,8,15,20,...,200
Uttar Pradesh,Dadri,2018,1,Children,1,5,10,...,80
```

**Minimum Data Requirements:**
- Time span: At least 5 years (60 months)
- Coverage: All 4 protected groups
- Granularity: Monthly data preferred
- Missing data: <10% per feature

### 3.2 Feature Engineering Pipeline

**Step 1: Temporal Features**

```python
def create_temporal_features(df):
    """
    Generate time-based features from datetime
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    return df
```

**Step 2: Lag Features**

```python
def create_lag_features(df, target='crime_rate', lags=[1, 2, 3, 6, 12]):
    """
    Create lagged versions of target variable
    """
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df.groupby(['state', 'district', 'protected_group'])[target].shift(lag)
    
    return df
```

**Step 3: Rolling Statistics**

```python
def create_rolling_features(df, target='crime_rate', windows=[3, 6, 12]):
    """
    Rolling mean, std, min, max
    """
    for window in windows:
        # Rolling mean
        df[f'{target}_rolling_mean_{window}'] = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rolling std
        df[f'{target}_rolling_std_{window}'] = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rolling min/max
        df[f'{target}_rolling_min_{window}'] = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )
        
        df[f'{target}_rolling_max_{window}'] = (
            df.groupby(['state', 'district', 'protected_group'])[target]
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )
    
    return df
```

**Step 4: Trend and Seasonality**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

def extract_trend_seasonality(df, target='crime_rate'):
    """
    Decompose time series into trend and seasonal components
    """
    df['trend'] = 0
    df['seasonality'] = 0
    df['residual'] = 0
    
    for (state, district, group), group_df in df.groupby(['state', 'district', 'protected_group']):
        if len(group_df) >= 24:  # Need at least 2 years
            decomposition = seasonal_decompose(
                group_df[target].values,
                model='additive',
                period=12,
                extrapolate_trend='freq'
            )
            
            df.loc[group_df.index, 'trend'] = decomposition.trend
            df.loc[group_df.index, 'seasonality'] = decomposition.seasonal
            df.loc[group_df.index, 'residual'] = decomposition.resid
    
    return df
```

**Step 5: Crime Type Features**

```python
def create_crime_type_features(df):
    """
    Aggregate and normalize crime types
    """
    crime_types = ['murder', 'rape', 'kidnapping', 'dacoity', 'robbery',
                   'burglary', 'theft', 'riots', 'arson', 'hurt', 
                   'cheating', 'counterfeiting']
    
    # Calculate crime type ratios
    df['total_crimes'] = df[crime_types].sum(axis=1)
    
    for crime in crime_types:
        df[f'{crime}_ratio'] = df[crime] / (df['total_crimes'] + 1)  # +1 to avoid division by zero
    
    # Violent vs non-violent
    violent_crimes = ['murder', 'rape', 'kidnapping', 'robbery', 'dacoity']
    df['violent_crime_count'] = df[violent_crimes].sum(axis=1)
    df['violent_crime_ratio'] = df['violent_crime_count'] / (df['total_crimes'] + 1)
    
    # Property crimes
    property_crimes = ['burglary', 'theft', 'cheating', 'counterfeiting']
    df['property_crime_count'] = df[property_crimes].sum(axis=1)
    df['property_crime_ratio'] = df['property_crime_count'] / (df['total_crimes'] + 1)
    
    return df
```

**Step 6: Regional Features**

```python
def create_regional_features(df, demographic_data):
    """
    Merge demographic and regional metadata
    """
    # Merge demographic data
    df = df.merge(
        demographic_data[['state', 'district', 'year', 'population', 
                          'literacy_rate', 'poverty_index', 'urban_rural']],
        on=['state', 'district', 'year'],
        how='left'
    )
    
    # Calculate crime rate per capita
    df['crime_rate_per_capita'] = (df['total_crimes'] / df['population']) * 100000
    
    # Encode categorical variables
    df['urban_rural_encoded'] = (df['urban_rural'] == 'urban').astype(int)
    
    # State encoding (one-hot or label encoding)
    df['state_encoded'] = pd.Categorical(df['state']).codes
    
    return df
```

**Step 7: Group Indicator**

```python
def encode_protected_groups(df):
    """
    Encode protected groups as numerical labels
    """
    group_mapping = {
        'SC': 0,
        'ST': 1,
        'Women': 2,
        'Children': 3
    }
    
    df['group_label'] = df['protected_group'].map(group_mapping)
    
    return df
```

### 3.3 Data Splitting Strategy

**Time-Based Split (Critical for time series):**

```python
def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data by time to prevent data leakage
    """
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Val: {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, val_df, test_df
```

**Example Split:**
- Train: 2015-01 to 2019-12 (5 years)
- Validation: 2020-01 to 2021-06 (1.5 years)
- Test: 2021-07 to 2023-12 (2.5 years)

### 3.4 Sequence Creation for LSTM

```python
def create_sequences(df, sequence_length=12, target='crime_rate'):
    """
    Create overlapping sequences for LSTM input
    """
    X_sequences = []
    y_targets = []
    group_labels = []
    
    for (state, district, group), group_df in df.groupby(['state', 'district', 'protected_group']):
        # Sort by date
        group_df = group_df.sort_values('date')
        
        # Extract features and target
        features = group_df.drop(['crime_rate', 'date', 'protected_group'], axis=1).values
        targets = group_df['crime_rate'].values
        
        # Create sequences
        for i in range(len(group_df) - sequence_length):
            X_sequences.append(features[i:i+sequence_length])
            y_targets.append(targets[i+sequence_length])
            group_labels.append(group_df['group_label'].iloc[i])
    
    X = np.array(X_sequences)  # Shape: (num_samples, sequence_length, num_features)
    y = np.array(y_targets)    # Shape: (num_samples,)
    groups = np.array(group_labels)  # Shape: (num_samples,)
    
    return X, y, groups
```

### 3.5 Normalization

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training set statistics
    """
    # Reshape to 2D for scaling
    n_train, seq_len, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_features)
    
    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

---

## 4. IMPLEMENTATION DETAILS

### 4.1 Fairness-Constrained Loss Function

**Objective:** Minimize prediction error WHILE minimizing fairness gap

**Mathematical Formulation:**

```
Total Loss = Prediction Loss + λ × Fairness Loss

Where:
- Prediction Loss = MSE(ŷ, y)
- Fairness Loss = Pairwise MAE differences between groups
- λ = Fairness penalty weight (hyperparameter)
```

**Implementation:**

```python
class FairnessConstrainedLoss(nn.Module):
    def __init__(self, lambda_fairness=1.0):
        super(FairnessConstrainedLoss, self).__init__()
        self.lambda_fairness = lambda_fairness
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predictions, targets, group_labels):
        """
        predictions: (batch_size, 1)
        targets: (batch_size, 1)
        group_labels: (batch_size,) with values 0, 1, 2, 3
        """
        # Prediction loss
        mse = self.mse_loss(predictions, targets)
        
        # Fairness loss - pairwise MAE differences
        fairness_penalty = 0.0
        n_comparisons = 0
        
        for group1 in range(4):  # SC, ST, Women, Children
            mask1 = (group_labels == group1)
            if mask1.sum() == 0:
                continue
            
            pred1 = predictions[mask1]
            target1 = targets[mask1]
            mae1 = torch.abs(pred1 - target1).mean()
            
            for group2 in range(group1 + 1, 4):
                mask2 = (group_labels == group2)
                if mask2.sum() == 0:
                    continue
                
                pred2 = predictions[mask2]
                target2 = targets[mask2]
                mae2 = torch.abs(pred2 - target2).mean()
                
                # Add absolute difference of MAEs
                fairness_penalty += torch.abs(mae1 - mae2)
                n_comparisons += 1
        
        # Average fairness penalty
        if n_comparisons > 0:
            fairness_penalty = fairness_penalty / n_comparisons
        
        # Total loss
        total_loss = mse + self.lambda_fairness * fairness_penalty
        
        return total_loss, mse, fairness_penalty
```

**Hyperparameter λ (lambda_fairness):**
- λ = 0: Only optimize accuracy (same as baseline)
- λ = 0.5: Light fairness constraint
- λ = 1.0: Balanced accuracy and fairness (RECOMMENDED)
- λ = 2.0: Strong fairness constraint (may hurt accuracy)
- λ = 5.0: Extreme fairness (significant accuracy loss)

**How to Choose λ:**
```python
# Grid search over lambda values
lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]

best_lambda = None
best_score = float('inf')

for lambda_val in lambda_values:
    model = FC_MT_LSTM(input_dim, hidden_dim)
    loss_fn = FairnessConstrainedLoss(lambda_fairness=lambda_val)
    
    # Train model
    train_model(model, loss_fn, train_loader, val_loader)
    
    # Evaluate
    val_mae = evaluate_mae(model, val_loader)
    val_fairness_gap = evaluate_fairness_gap(model, val_loader)
    
    # Combined score (you can adjust weights)
    score = val_mae + 0.5 * val_fairness_gap
    
    if score < best_score:
        best_score = score
        best_lambda = lambda_val

print(f"Best lambda: {best_lambda}")
```

### 4.2 Training Configuration

**Optimizer:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,  # Initial learning rate
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # L2 regularization
)
```

**Learning Rate Scheduler:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # Reduce LR by half
    patience=10,  # Wait 10 epochs before reducing
    verbose=True,
    min_lr=1e-6
)
```

**Early Stopping:**
```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
```

**Batch Size:**
- Training: 128 (adjust based on GPU memory)
- Validation: 256
- Test: 256

**Epochs:**
- Maximum: 200
- Early stopping patience: 20 epochs
- Expected convergence: 50-100 epochs

### 4.3 Training Loop

```python
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
        
        # Average training metrics
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_fairness_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
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
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_groups.append(group_batch.cpu().numpy())
        
        # Average validation metrics
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Calculate fairness gap
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
    checkpoint = torch.load('best_fc_mt_lstm.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses, train_maes, val_maes, fairness_gaps
```

---

## 5. TRAINING PIPELINE

### 5.1 Complete Training Script

```python
# train_fc_mt_lstm.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# ============ CONFIGURATION ============
CONFIG = {
    'data_path': 'data/crime_data_processed.csv',
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

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ============ DATASET CLASS ============
class CrimeDataset(Dataset):
    def __init__(self, X, y, groups):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.groups = torch.LongTensor(groups)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.groups[idx]

# ============ LOAD AND PREPARE DATA ============
print("Loading and preparing data...")

# Load processed data
df = pd.read_csv(CONFIG['data_path'])

# Time-based split
train_df, val_df, test_df = time_based_split(df, train_ratio=0.7, val_ratio=0.15)

# Create sequences
X_train, y_train, groups_train = create_sequences(train_df, CONFIG['sequence_length'])
X_val, y_val, groups_val = create_sequences(val_df, CONFIG['sequence_length'])
X_test, y_test, groups_test = create_sequences(test_df, CONFIG['sequence_length'])

print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Normalize features
X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

# Create datasets and dataloaders
train_dataset = CrimeDataset(X_train, y_train, groups_train)
val_dataset = CrimeDataset(X_val, y_val, groups_val)
test_dataset = CrimeDataset(X_test, y_test, groups_test)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ============ INITIALIZE MODEL ============
print("Initializing model...")

input_dim = X_train.shape[2]  # Number of features
model = FC_MT_LSTM(input_dim, CONFIG['hidden_dim'])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {CONFIG['device']}")

# ============ SETUP TRAINING ============
loss_fn = FairnessConstrainedLoss(lambda_fairness=CONFIG['lambda_fairness'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

# ============ TRAIN MODEL ============
print("\nStarting training...\n")

model, train_losses, val_losses, train_maes, val_maes, fairness_gaps = train_model(
    model, train_loader, val_loader, loss_fn, optimizer, 
    scheduler, early_stopping, CONFIG['num_epochs'], CONFIG['device']
)

print("\nTraining completed!")

# ============ PLOT TRAINING CURVES ============
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
axes[1, 1].plot([optimizer.param_groups[0]['lr'] for _ in range(len(train_losses))])
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("Training curves saved to training_curves.png")

# ============ SAVE MODEL AND ARTIFACTS ============
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'config': CONFIG,
    'input_dim': input_dim
}, 'fc_mt_lstm_final.pth')

print("Model saved to fc_mt_lstm_final.pth")
```

### 5.2 Evaluation Script

```python
# evaluate_fc_mt_lstm.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

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
    targets = np.concatenate(all_targets)
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
    fairness_gap = max(maes) - min(maes)
    avg_mae = np.mean(maes)
    fairness_ratio = fairness_gap / avg_mae if avg_mae > 0 else 0
    
    print("\n" + "=" * 60)
    print("FAIRNESS METRICS")
    print("=" * 60)
    print(f"Fairness Gap (max MAE - min MAE): {fairness_gap:.4f}")
    print(f"Average MAE across groups: {avg_mae:.4f}")
    print(f"Fairness Ratio (gap / avg): {fairness_ratio:.4f}")
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

# ============ LOAD MODEL AND EVALUATE ============
print("Loading model...")
checkpoint = torch.load('fc_mt_lstm_final.pth')
model = FC_MT_LSTM(checkpoint['input_dim'], checkpoint['config']['hidden_dim'])
model.load_state_dict(checkpoint['model_state_dict'])

print("Evaluating on test set...")
results, predictions, targets, groups, attention = evaluate_model(model, test_loader)

# Save results
with open('fc_mt_lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to fc_mt_lstm_results.json")
```

---

## 6. EVALUATION METRICS

### 6.1 Accuracy Metrics

**Mean Absolute Error (MAE):**
```python
MAE = (1/n) * Σ |y_true - y_pred|
```
- Measures average absolute prediction error
- Same unit as target variable (crime count)
- Lower is better
- Target: <7.0

**Root Mean Squared Error (RMSE):**
```python
RMSE = sqrt((1/n) * Σ (y_true - y_pred)²)
```
- Penalizes large errors more than MAE
- Same unit as target variable
- Lower is better
- Target: <10.0

**R² Score (Coefficient of Determination):**
```python
R² = 1 - (SS_res / SS_tot)
where:
  SS_res = Σ (y_true - y_pred)²
  SS_tot = Σ (y_true - y_mean)²
```
- Proportion of variance explained
- Range: (-∞, 1], where 1 is perfect
- Higher is better
- Target: >0.70

### 6.2 Fairness Metrics

**Fairness Gap:**
```python
Fairness Gap = max(MAE_group_i) - min(MAE_group_i)
                where i ∈ {SC, ST, Women, Children}
```
- Absolute difference between best and worst group performance
- Lower is better
- Target: <1.0

**Fairness Ratio:**
```python
Fairness Ratio = Fairness Gap / Average MAE across groups
```
- Relative fairness measure
- Range: [0, ∞), where 0 is perfect fairness
- Lower is better
- Target: <0.15 (15%)

**Pairwise Fairness:**
```python
For each pair of groups (i, j):
  Pairwise Gap_{i,j} = |MAE_i - MAE_j|
```
- All 6 pairwise comparisons (4 choose 2)
- Identifies which groups have largest disparities

### 6.3 Comparison with Baselines

**Metrics to Report in Paper:**

| Model | Overall MAE | Overall RMSE | Overall R² | Fairness Gap | Fairness Ratio |
|-------|-------------|--------------|------------|--------------|----------------|
| SARIMA | X.XX | X.XX | X.XX | X.XX | X.XX |
| Prophet | X.XX | X.XX | X.XX | X.XX | X.XX |
| Random Forest | X.XX | X.XX | X.XX | X.XX | X.XX |
| XGBoost | 1.83 | X.XX | X.XX | 6-8 (est) | X.XX |
| CNN-LSTM | X.XX | X.XX | X.XX | X.XX | X.XX |
| Transformer | X.XX | X.XX | X.XX | X.XX | X.XX |
| **FC-MT-LSTM** | **~6.5-7.0** | **~9.5** | **~0.75** | **<1.0** | **<0.15** |

**Expected Improvements:**
- 85% reduction in fairness gap vs XGBoost
- Only 3-5% accuracy loss vs XGBoost
- 50% reduction in fairness gap vs Random Forest
- 10-15% accuracy improvement vs Random Forest

---

## 7. REACT DASHBOARD SPECIFICATION

### 7.1 Technology Stack

**Frontend Framework:**
- React 18+ with functional components
- TypeScript for type safety
- Vite for fast development

**Visualization Libraries:**
- D3.js v7 for custom visualizations
- Recharts for standard charts (bar, line, area)
- Victory for animated charts
- React-Simple-Maps for geographic maps

**UI Components:**
- Material-UI (MUI) for component library
- Styled-components for custom styling
- Framer Motion for animations

**State Management:**
- React Context API for global state
- React Query for data fetching
- Zustand for complex state (optional)

**Styling:**
- Pastel color theme
- Responsive design (mobile, tablet, desktop)
- Dark mode support

### 7.2 Color Palette (Pastel Theme)

```javascript
const theme = {
  colors: {
    // Primary pastels
    primary: {
      light: '#B8E0FF',    // Pastel blue
      main: '#A8D8EA',     // Sky blue
      dark: '#7FB3D5',
    },
    
    // Secondary pastels
    secondary: {
      light: '#FFD6E8',    // Pastel pink
      main: '#FFB6C1',     // Light pink
      dark: '#FF9AA2',
    },
    
    // Accent colors (pastel)
    accent: {
      purple: '#D4B5FF',   // Pastel purple
      green: '#B5EAD7',    // Mint green
      yellow: '#FFDAB9',   // Peach
      coral: '#FFCCCB',    // Light coral
    },
    
    // Protected group colors
    groups: {
      SC: '#A8D8EA',       // Pastel blue
      ST: '#D4B5FF',       // Pastel purple
      Women: '#FFB6C1',    // Pastel pink
      Children: '#B5EAD7', // Mint green
    },
    
    // Background and text
    background: {
      primary: '#FAFAFA',
      secondary: '#FFFFFF',
      tertiary: '#F5F5F5',
    },
    
    text: {
      primary: '#2C3E50',
      secondary: '#7F8C8D',
      disabled: '#BDC3C7',
    },
    
    // Semantic colors (pastel versions)
    success: '#B5EAD7',
    warning: '#FFE5B4',
    error: '#FFCCCB',
    info: '#B8E0FF',
    
    // Chart colors (8 pastels for model comparison)
    chart: [
      '#A8D8EA', // Pastel blue
      '#D4B5FF', // Pastel purple
      '#FFB6C1', // Pastel pink
      '#B5EAD7', // Mint green
      '#FFDAB9', // Peach
      '#FFE5B4', // Pale yellow
      '#FFCCCB', // Light coral
      '#E0BBE4', // Lavender
    ],
  },
  
  // Typography
  typography: {
    fontFamily: "'Inter', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      md: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      xxl: '1.5rem',
    },
    fontWeight: {
      light: 300,
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  
  // Spacing
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px',
  },
  
  // Border radius
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    full: '9999px',
  },
  
  // Shadows (soft, pastel-friendly)
  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.08)',
    md: '0 4px 6px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px rgba(0, 0, 0, 0.12)',
    xl: '0 20px 25px rgba(0, 0, 0, 0.15)',
  },
};
```

### 7.3 Dashboard Layout

**Overall Structure:**

```
┌─────────────────────────────────────────────────────────┐
│  Navbar (Logo, Navigation, Search, Theme Toggle)       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┬──────────────────────────────────┐  │
│  │               │                                  │  │
│  │   Sidebar     │      Main Content Area           │  │
│  │   (Filters,   │      (Dynamic based on route)    │  │
│  │    Options)   │                                  │  │
│  │               │                                  │  │
│  │               │                                  │  │
│  │               │                                  │  │
│  └───────────────┴──────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Pages/Routes:**

1. **Dashboard Overview** (`/`)
   - Key metrics cards
   - Model comparison chart
   - Fairness overview
   - Recent predictions timeline

2. **Model Comparison** (`/comparison`)
   - Side-by-side model performance
   - Interactive comparison charts
   - Detailed metrics table

3. **Fairness Analysis** (`/fairness`)
   - Per-group performance breakdown
   - Pairwise fairness comparisons
   - Fairness gap visualization
   - Attention heatmaps

4. **Predictions** (`/predictions`)
   - Interactive prediction tool
   - Time series forecasts
   - Geographic crime maps
   - Confidence intervals

5. **Insights** (`/insights`)
   - Feature importance
   - Attention analysis
   - Trend analysis
   - Anomaly detection

6. **Documentation** (`/docs`)
   - Model architecture explanation
   - Fairness methodology
   - API documentation
   - Citation guidelines

### 7.4 Key Visualizations

#### 7.4.1 Model Comparison Bar Chart (D3.js)

**Purpose:** Compare MAE, RMSE, R² across all models

**Visual Design:**
```
Models:
SARIMA      ████████ 12.5
Prophet     ██████ 9.8
Rand Forest ████ 6.2
XGBoost     ███ 1.8  ← Best accuracy
CNN-LSTM    ████ 5.5
Transformer ████ 6.0
FC-MT-LSTM  ███ 2.1  ← NEW: Best balance
```

**Implementation Notes:**
- Grouped bar chart (MAE, RMSE, R² side-by-side)
- Pastel colors for each model
- Hover: Show exact values and sample size
- Click: Navigate to model details
- Animation: Bars grow from left on load

#### 7.4.2 Fairness Gap Visualization (D3.js)

**Purpose:** Show MAE differences between protected groups

**Visual Design:**
```
      SC      ST    Women  Children
      │       │       │       │
  10─ ┼       ┼       ┼       ┼
      │       │       █       │
   8─ ┼       ┼       █       ┼
      │       █       █       │
   6─ ┼       █       █       ┼
      │       █       █       █
   4─ ┼       █       █       █
      █       █       █       █
   2─ █       █       █       █
      █       █       █       █
   0─ ┴───────┴───────┴───────┴───
    
      ← Fairness Gap (shaded region) →
```

**Implementation Notes:**
- Vertical bar chart with error bars
- Shaded region showing fairness gap
- Group colors: SC (blue), ST (purple), Women (pink), Children (green)
- Comparison slider: Select two models to compare

#### 7.4.3 Time Series Forecast (Recharts)

**Purpose:** Show predictions over time with confidence intervals

**Visual Design:**
```
Crime Rate
   │
40 ┤         ╱‾‾‾‾‾‾╲     Shaded: Confidence interval
   │        ╱        ╲    Solid line: Predictions
30 ┤   ____╱          ╲   Dotted: Actual
   │  ╱                ╲
20 ┤ ╱                  ╲___
   │╱
10 ┤
   │
 0 └─────────────────────────────────
   2020  2021  2022  2023  2024
```

**Implementation Notes:**
- Area chart for confidence intervals (pastel with transparency)
- Line chart overlay for predictions (solid)
- Scatter points for actuals (dotted)
- Toggle: Select protected group
- Zoom: Drag to zoom on time range

#### 7.4.4 Geographic Crime Heatmap (D3.js + React-Simple-Maps)

**Purpose:** Show crime predictions by state/district

**Visual Design:**
```
India Map:
  - States colored by predicted crime rate
  - Darker pastels = higher crime
  - Hover: Show state name, crime rate, confidence
  - Click: Drill down to district level
```

**Color Scale:**
```
Low Crime    │  Medium    │  High Crime
#E8F5E9  →  #B5EAD7  →  #7FB3D5  →  #5F9EA0
(Very light) (Pastel)    (Medium)   (Dark pastel)
```

**Implementation Notes:**
- TopoJSON for India map
- D3 color scale for crime intensity
- Smooth transitions between states
- Side panel: Show top 10 states

#### 7.4.5 Attention Heatmap (D3.js)

**Purpose:** Visualize which features model focuses on

**Visual Design:**
```
Features ↓       Time Steps →
            t-12  t-11  t-10  ...  t-1
Murder       █     ░     ░    ...  ███
Rape         ░     ░     █    ...  ██
Kidnapping   ░     ██    ░    ...  █
Theft        ███   ██    █    ...  ░
...

Legend: ░ Low attention  █ Medium  ███ High
```

**Implementation Notes:**
- Grid heatmap using D3
- Color: Pastel yellow (low) → Pastel orange (high)
- Hover: Show attention score (0.0 to 1.0)
- Select: Specific prediction instance
- Export: Download as PNG

#### 7.4.6 Fairness Trade-off Scatter Plot (D3.js)

**Purpose:** Show accuracy vs fairness trade-off

**Visual Design:**
```
Fairness Gap
   │
 8 ┤  XGBoost●
   │           (High accuracy, poor fairness)
 6 ┤
   │
 4 ┤      Transformer●  CNN-LSTM●
   │
 2 ┤  Random Forest●
   │                    FC-MT-LSTM●
 0 ┤                    (Best balance!)
   └──────────────────────────────────
   0    2    4    6    8   10   12  MAE
        ←  Better accuracy
```

**Implementation Notes:**
- Scatter plot with model labels
- X-axis: MAE (lower is better)
- Y-axis: Fairness gap (lower is better)
- Pareto frontier line (ideal trade-off curve)
- Quadrants: Color-coded (good/bad regions)
- Hover: Show exact metrics

#### 7.4.7 Metrics Dashboard Cards

**Purpose:** Quick overview of key metrics

**Visual Design:**
```
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Overall MAE   │  │ Fairness Gap   │  │   R² Score     │
│                │  │                │  │                │
│     6.52       │  │     0.85       │  │     0.752      │
│  ↓ 64% vs XG  │  │  ↓ 87% vs XG  │  │  ↑ 5% vs RF   │
└────────────────┘  └────────────────┘  └────────────────┘

  Pastel blue         Pastel green        Pastel purple
   background          background          background
```

**Implementation Notes:**
- Card component with icon, value, change indicator
- Arrow indicators: ↑ (improvement), ↓ (regression)
- Percentage change vs baseline
- Click: Navigate to detailed view
- Animation: Count-up effect on load

### 7.5 Interactive Features

#### 7.5.1 Global Filters (Sidebar)

```javascript
// Filter options
const filters = {
  protectedGroups: ['SC', 'ST', 'Women', 'Children'],  // Multi-select
  timeRange: ['2020-01', '2023-12'],                   // Date range
  states: ['All', 'Uttar Pradesh', 'Bihar', ...],      // Dropdown
  crimeTypes: ['All', 'Violent', 'Property'],          // Dropdown
  models: ['All', 'FC-MT-LSTM', 'XGBoost', ...],       // Multi-select
};
```

**UI Components:**
- Multi-select chips for groups
- Date range picker with presets (Last 6 months, 1 year, All)
- Autocomplete dropdown for states
- Checkbox group for crime types
- Model selector with "Compare" button

#### 7.5.2 Comparison Mode

**Feature:** Select 2-3 models to compare side-by-side

**Implementation:**
```javascript
<ComparisonToolbar>
  <ModelSelector>
    <Chip>FC-MT-LSTM</Chip>  {/* Selected */}
    <Chip>XGBoost</Chip>     {/* Selected */}
    <Chip>Random Forest</Chip> {/* Add */}
  </ModelSelector>
  
  <ComparisonMetrics>
    <Select>MAE</Select>
    <Select>Fairness Gap</Select>
    <Select>R²</Select>
  </ComparisonMetrics>
  
  <Button>Generate Report</Button>
</ComparisonToolbar>
```

**Visual:**
- Split screen with synchronized charts
- Difference highlighting (better/worse indicators)
- Export comparison as PDF

#### 7.5.3 Prediction Tool

**Feature:** Make custom predictions for specific scenarios

**UI:**
```javascript
<PredictionForm>
  <FormSection title="Location">
    <StateSelect />
    <DistrictSelect />
  </FormSection>
  
  <FormSection title="Time Period">
    <MonthPicker />
    <YearPicker />
  </FormSection>
  
  <FormSection title="Protected Group">
    <RadioGroup options={['SC', 'ST', 'Women', 'Children']} />
  </FormSection>
  
  <FormSection title="Historical Context">
    <CrimeTypeInputs />  {/* Sliders for each crime type */}
  </FormSection>
  
  <Button onClick={handlePredict}>Predict Crime Rate</Button>
</PredictionForm>

<PredictionResults>
  <ResultCard>
    <Title>Predicted Crime Rate</Title>
    <Value>25.3</Value>
    <ConfidenceInterval>[22.1, 28.5]</ConfidenceInterval>
    <AttentionBreakdown />  {/* Top 5 influential features */}
  </ResultCard>
</PredictionResults>
```

### 7.6 Component Hierarchy

```
App
├── ThemeProvider
│   └── Router
│       ├── Layout
│       │   ├── Navbar
│       │   │   ├── Logo
│       │   │   ├── NavigationMenu
│       │   │   ├── SearchBar
│       │   │   └── ThemeToggle
│       │   │
│       │   ├── Sidebar
│       │   │   ├── FilterPanel
│       │   │   │   ├── GroupFilter
│       │   │   │   ├── TimeRangeFilter
│       │   │   │   ├── StateFilter
│       │   │   │   └── CrimeTypeFilter
│       │   │   │
│       │   │   └── QuickActions
│       │   │       ├── CompareButton
│       │   │       ├── ExportButton
│       │   │       └── ResetButton
│       │   │
│       │   └── MainContent
│       │       ├── DashboardPage
│       │       │   ├── MetricsGrid
│       │       │   │   └── MetricCard (×4)
│       │       │   ├── ModelComparisonChart
│       │       │   ├── FairnessOverview
│       │       │   └── RecentPredictions
│       │       │
│       │       ├── ComparisonPage
│       │       │   ├── ComparisonToolbar
│       │       │   ├── SideBySideCharts
│       │       │   └── MetricsTable
│       │       │
│       │       ├── FairnessPage
│       │       │   ├── GroupPerformanceChart
│       │       │   ├── PairwiseComparison
│       │       │   ├── FairnessGapVisualization
│       │       │   └── AttentionHeatmap
│       │       │
│       │       ├── PredictionsPage
│       │       │   ├── PredictionForm
│       │       │   ├── TimeSeriesForecast
│       │       │   ├── GeographicHeatmap
│       │       │   └── ConfidenceIntervals
│       │       │
│       │       ├── InsightsPage
│       │       │   ├── FeatureImportance
│       │       │   ├── AttentionAnalysis
│       │       │   ├── TrendAnalysis
│       │       │   └── AnomalyDetection
│       │       │
│       │       └── DocsPage
│       │           ├── ModelArchitecture
│       │           ├── FairnessMethodology
│       │           ├── APIDocumentation
│       │           └── CitationGuidelines
│       │
│       └── Footer
│
└── StateProvider (React Context)
    ├── DataContext
    ├── FilterContext
    └── ComparisonContext
```

### 7.7 Data Flow

**API Structure:**

```javascript
// Mock API responses (replace with actual backend)

// GET /api/models
{
  "models": [
    {
      "id": "fc-mt-lstm",
      "name": "FC-MT-LSTM",
      "type": "deep-learning",
      "description": "Fairness-constrained multi-task LSTM",
      "trainedAt": "2024-02-01T10:00:00Z"
    },
    // ... other models
  ]
}

// GET /api/metrics?model=fc-mt-lstm
{
  "modelId": "fc-mt-lstm",
  "overall": {
    "mae": 6.52,
    "rmse": 9.45,
    "r2": 0.752
  },
  "fairness": {
    "gap": 0.85,
    "ratio": 0.13,
    "pairwise": {
      "SC_ST": 0.32,
      "SC_Women": 0.45,
      "SC_Children": 0.85,
      "ST_Women": 0.13,
      "ST_Children": 0.53,
      "Women_Children": 0.40
    }
  },
  "perGroup": {
    "SC": { "mae": 6.21, "rmse": 9.12, "r2": 0.761, "samples": 1250 },
    "ST": { "mae": 6.53, "rmse": 9.45, "r2": 0.748, "samples": 980 },
    "Women": { "mae": 6.66, "rmse": 9.67, "r2": 0.742, "samples": 1450 },
    "Children": { "mae": 7.06, "rmse": 10.12, "r2": 0.725, "samples": 1120 }
  }
}

// POST /api/predict
// Request:
{
  "model": "fc-mt-lstm",
  "inputs": {
    "state": "Uttar Pradesh",
    "district": "Dadri",
    "month": 3,
    "year": 2024,
    "protectedGroup": "Women",
    "crimeTypes": {
      "murder": 5,
      "rape": 12,
      "kidnapping": 18,
      // ... other crime types
    }
  }
}

// Response:
{
  "prediction": 25.3,
  "confidenceInterval": [22.1, 28.5],
  "confidence": 0.85,
  "attention": {
    "rape": 0.32,
    "kidnapping": 0.28,
    "crimeRateLag1": 0.18,
    "trend": 0.12,
    "rollingMean3": 0.10
  }
}

// GET /api/predictions/history?model=fc-mt-lstm&group=Women&startDate=2023-01&endDate=2023-12
{
  "data": [
    {
      "date": "2023-01",
      "actual": 23.5,
      "predicted": 24.1,
      "confidenceInterval": [21.2, 27.0]
    },
    // ... more data points
  ],
  "metrics": {
    "mae": 1.85,
    "rmse": 2.52
  }
}
```

**State Management:**

```javascript
// contexts/DataContext.js
const DataContext = createContext();

export function DataProvider({ children }) {
  const [models, setModels] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [predictions, setPredictions] = useState([]);
  
  const fetchModels = async () => {
    const response = await fetch('/api/models');
    const data = await response.json();
    setModels(data.models);
  };
  
  const fetchMetrics = async (modelId) => {
    const response = await fetch(`/api/metrics?model=${modelId}`);
    const data = await response.json();
    setMetrics(prev => ({ ...prev, [modelId]: data }));
  };
  
  const makePrediction = async (inputs) => {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs)
    });
    const data = await response.json();
    setPredictions(prev => [...prev, data]);
    return data;
  };
  
  return (
    <DataContext.Provider value={{
      models,
      metrics,
      predictions,
      fetchModels,
      fetchMetrics,
      makePrediction
    }}>
      {children}
    </DataContext.Provider>
  );
}
```

### 7.8 Responsive Design

**Breakpoints:**
```javascript
const breakpoints = {
  mobile: '320px',
  tablet: '768px',
  desktop: '1024px',
  wide: '1440px',
};
```

**Mobile Layout (< 768px):**
- Hamburger menu for navigation
- Sidebar collapses to bottom drawer
- Charts stack vertically
- Single column layout
- Touch-optimized controls

**Tablet Layout (768px - 1024px):**
- Collapsible sidebar
- 2-column grid for charts
- Horizontal scrolling for large tables

**Desktop Layout (> 1024px):**
- Full sidebar visible
- 3-column grid
- Multiple charts visible simultaneously
- Hover interactions

### 7.9 Accessibility

**WCAG 2.1 AA Compliance:**
- Color contrast ratio ≥ 4.5:1 for text
- Keyboard navigation support
- ARIA labels for all interactive elements
- Screen reader compatibility
- Focus indicators visible

**Implementation:**
```javascript
// Accessible chart component example
<ChartContainer
  role="img"
  aria-label="Model comparison bar chart showing MAE across 7 models"
  tabIndex={0}
>
  <svg>
    {/* Chart content */}
  </svg>
  <ScreenReaderOnly>
    <table>
      <caption>Model Performance Comparison</caption>
      <thead>
        <tr>
          <th>Model</th>
          <th>MAE</th>
          <th>Fairness Gap</th>
        </tr>
      </thead>
      <tbody>
        {data.map(item => (
          <tr key={item.model}>
            <td>{item.model}</td>
            <td>{item.mae}</td>
            <td>{item.fairnessGap}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </ScreenReaderOnly>
</ChartContainer>
```

---

## 8. TIMELINE & MILESTONES

### 8.1 Phase 1: Model Development (Week 1-2)

**Days 1-3: Data Preparation**
- [ ] Clean and preprocess raw crime data
- [ ] Engineer temporal features (lags, rolling stats)
- [ ] Create crime type aggregations
- [ ] Merge demographic data
- [ ] Encode protected groups
- [ ] Generate sequences for LSTM
- [ ] Split data (train/val/test)
- [ ] Normalize features

**Output:** `crime_data_processed.csv`, data preprocessing scripts

**Days 4-7: Model Implementation**
- [ ] Implement SharedEncoder class
- [ ] Implement AttentionLayer class
- [ ] Implement MultiTaskDecoders class
- [ ] Implement FairnessConstrainedLoss
- [ ] Integrate into FC_MT_LSTM model
- [ ] Write training loop
- [ ] Test on small data subset

**Output:** `fc_mt_lstm.py`, `train_fc_mt_lstm.py`

**Days 8-14: Model Training & Hyperparameter Tuning**
- [ ] Train FC-MT-LSTM with default hyperparameters
- [ ] Grid search for λ (fairness penalty)
- [ ] Tune learning rate and batch size
- [ ] Experiment with encoder/decoder architectures
- [ ] Train final model with best hyperparameters
- [ ] Generate training curves

**Output:** `best_fc_mt_lstm.pth`, training logs, hyperparameter results

### 8.2 Phase 2: Evaluation & Comparison (Week 3)

**Days 15-17: Model Evaluation**
- [ ] Evaluate FC-MT-LSTM on test set
- [ ] Calculate overall metrics (MAE, RMSE, R²)
- [ ] Calculate per-group metrics
- [ ] Compute fairness metrics
- [ ] Generate attention heatmaps
- [ ] Analyze error patterns

**Output:** `fc_mt_lstm_results.json`, evaluation plots

**Days 18-21: Baseline Comparison**
- [ ] Run fix_fairness_metrics.py for baselines
- [ ] Compare FC-MT-LSTM vs all 6 baselines
- [ ] Create comparison tables
- [ ] Generate comparison charts
- [ ] Statistical significance testing
- [ ] Write comparison summary

**Output:** Updated `react_dashboard_data.json`, comparison report

### 8.3 Phase 3: Dashboard Development (Week 4-5)

**Days 22-25: Project Setup & Layout**
- [ ] Initialize React + TypeScript + Vite project
- [ ] Install dependencies (D3, Recharts, MUI, etc.)
- [ ] Configure pastel theme
- [ ] Implement Layout component (Navbar, Sidebar, Footer)
- [ ] Set up routing
- [ ] Create global state contexts

**Output:** React project structure, theme configuration

**Days 26-28: Core Visualizations (D3.js)**
- [ ] Model comparison bar chart
- [ ] Fairness gap visualization
- [ ] Attention heatmap
- [ ] Trade-off scatter plot

**Output:** Reusable D3 chart components

**Days 29-31: Dashboard Pages**
- [ ] Dashboard overview page
- [ ] Model comparison page
- [ ] Fairness analysis page
- [ ] Predictions page
- [ ] Insights page

**Output:** Complete dashboard pages

**Days 32-35: Interactive Features**
- [ ] Global filter system
- [ ] Comparison mode
- [ ] Prediction tool
- [ ] Export functionality
- [ ] Search functionality

**Output:** Interactive dashboard features

### 8.4 Phase 4: Integration & Testing (Week 6)

**Days 36-38: Backend Integration**
- [ ] Create mock API (or connect to Flask backend)
- [ ] Implement data fetching
- [ ] Handle loading and error states
- [ ] Add data caching

**Output:** Working API integration

**Days 39-40: Testing**
- [ ] Unit tests for components
- [ ] Integration tests
- [ ] Cross-browser testing
- [ ] Mobile responsiveness testing
- [ ] Accessibility audit

**Output:** Test suite, bug fixes

**Days 41-42: Deployment**
- [ ] Build optimized production bundle
- [ ] Deploy frontend (Vercel/Netlify)
- [ ] Deploy backend (if applicable)
- [ ] Set up CI/CD

**Output:** Live dashboard URL

### 8.5 Phase 5: Research Paper (Week 7-8)

**Days 43-45: Writing**
- [ ] Abstract
- [ ] Introduction & problem statement
- [ ] Related work
- [ ] Methodology (FC-MT-LSTM architecture)
- [ ] Experimental setup
- [ ] Results & discussion
- [ ] Conclusion & future work

**Output:** Full paper draft

**Days 46-49: Revision & Figures**
- [ ] Create publication-quality figures
- [ ] Revise based on feedback
- [ ] Proofread
- [ ] Format for venue (FAccT, AIES, etc.)

**Output:** Camera-ready paper

**Days 50-56: Submission**
- [ ] Final review
- [ ] Submit to conference/journal
- [ ] Prepare supplementary materials
- [ ] Share preprint on arXiv

**Output:** Submitted research paper

---

## 9. DELIVERABLES CHECKLIST

### 9.1 Code Deliverables

- [ ] `fc_mt_lstm.py` - Model architecture
- [ ] `train_fc_mt_lstm.py` - Training script
- [ ] `evaluate_fc_mt_lstm.py` - Evaluation script
- [ ] `data_preprocessing.py` - Data preparation pipeline
- [ ] `fix_fairness_metrics.py` - Fairness calculation for baselines
- [ ] `best_fc_mt_lstm.pth` - Trained model weights
- [ ] `react_dashboard_data.json` - All model results
- [ ] React dashboard (complete source code)
- [ ] `requirements.txt` - Python dependencies
- [ ] `package.json` - JavaScript dependencies
- [ ] `README.md` - Setup and usage instructions

### 9.2 Documentation Deliverables

- [ ] Model architecture diagram
- [ ] Training and evaluation guide
- [ ] Dashboard user guide
- [ ] API documentation
- [ ] Fairness methodology explanation
- [ ] Citation guidelines

### 9.3 Visualization Deliverables

- [ ] Training curves (loss, MAE, fairness gap)
- [ ] Model comparison charts (bar, line, scatter)
- [ ] Fairness gap visualization
- [ ] Attention heatmaps
- [ ] Time series forecasts
- [ ] Geographic crime maps
- [ ] Publication-quality figures (high DPI)

### 9.4 Research Deliverables

- [ ] Research paper (PDF)
- [ ] Supplementary materials
- [ ] ArXiv preprint
- [ ] Code repository (GitHub with DOI)
- [ ] Dataset (if shareable)

### 9.5 Presentation Deliverables

- [ ] Conference presentation slides
- [ ] Poster (if applicable)
- [ ] Demo video of dashboard
- [ ] 3-minute elevator pitch

---

## 10. TROUBLESHOOTING & COMMON ISSUES

### 10.1 Model Training Issues

**Issue: Model not converging**
- Check learning rate (try 0.0001 to 0.01)
- Verify data normalization
- Inspect gradient norms (use gradient clipping)
- Increase batch size
- Add more regularization (dropout, weight decay)

**Issue: Overfitting**
- Increase dropout rate (0.3 → 0.5)
- Add more training data
- Use data augmentation
- Reduce model capacity (fewer layers/units)
- Early stopping

**Issue: Fairness gap not improving**
- Increase λ (fairness penalty weight)
- Check group imbalance in data
- Verify loss function implementation
- Use stratified batching (equal samples per group)

### 10.2 Dashboard Development Issues

**Issue: D3 charts not rendering**
- Check SVG dimensions
- Verify data format
- Inspect browser console for errors
- Ensure D3 scales are correct

**Issue: Performance issues with large datasets**
- Implement data pagination
- Use virtual scrolling for tables
- Debounce filter updates
- Optimize D3 rendering (use canvas for large datasets)

**Issue: Responsive layout breaking**
- Test at all breakpoints
- Use CSS Grid/Flexbox properly
- Check overflow and scrolling
- Mobile-first design approach

### 10.3 Getting Help

**Resources:**
- PyTorch documentation: https://pytorch.org/docs/
- D3.js documentation: https://d3js.org/
- React documentation: https://react.dev/
- FAccT conference: https://facctconference.org/

**Communities:**
- PyTorch Forums
- D3.js Slack
- React Discord
- ML Fairness Researchers (Twitter/X)

---

## APPENDIX A: COMPLETE FILE STRUCTURE

```
crime-prediction-fairness/
├── data/
│   ├── raw/
│   │   ├── crime_data_raw.csv
│   │   ├── demographic_data.csv
│   │   └── regional_metadata.csv
│   └── processed/
│       └── crime_data_processed.csv
│
├── models/
│   ├── fc_mt_lstm.py
│   ├── train_fc_mt_lstm.py
│   ├── evaluate_fc_mt_lstm.py
│   └── data_preprocessing.py
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_fc_mt_lstm_experiments.ipynb
│
├── checkpoints/
│   ├── best_fc_mt_lstm.pth
│   └── training_logs.json
│
├── results/
│   ├── fc_mt_lstm_results.json
│   ├── react_dashboard_data.json
│   ├── training_curves.png
│   └── comparison_charts/
│
├── dashboard/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── charts/
│   │   │   ├── layout/
│   │   │   └── pages/
│   │   ├── contexts/
│   │   ├── utils/
│   │   ├── theme.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   └── user_guide.md
│
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── references.bib
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## APPENDIX B: QUICK START COMMANDS

```bash
# ============ MODEL TRAINING ============

# 1. Prepare data
python models/data_preprocessing.py

# 2. Train FC-MT-LSTM
python models/train_fc_mt_lstm.py

# 3. Evaluate model
python models/evaluate_fc_mt_lstm.py

# 4. Fix baseline fairness metrics
python fix_fairness_metrics.py

# ============ DASHBOARD DEVELOPMENT ============

# 1. Navigate to dashboard directory
cd dashboard

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev

# 4. Build for production
npm run build

# 5. Preview production build
npm run preview

# ============ TESTING ============

# Python tests
pytest tests/

# JavaScript tests
cd dashboard && npm test

# ============ DEPLOYMENT ============

# Deploy dashboard (Vercel)
cd dashboard && vercel --prod

# Or (Netlify)
cd dashboard && netlify deploy --prod
```

---

## FINAL NOTES

This comprehensive guide provides everything needed to:

1. ✅ Implement FC-MT-LSTM from scratch
2. ✅ Train and evaluate the model
3. ✅ Build a professional React dashboard with D3.js
4. ✅ Compare against 6 baseline models
5. ✅ Publish research paper at top venues

**Estimated Total Time:** 7-8 weeks

**Expected Results:**
- FC-MT-LSTM: MAE ~6.5-7.0, Fairness Gap <1.0
- 85% fairness improvement vs XGBoost
- Only 3-5% accuracy loss vs XGBoost
- Production-ready dashboard
- Publishable research contribution

**Next Steps:**
1. Run `fix_fairness_metrics.py` to get complete baseline results (5 min)
2. Decide: Quick fix only OR implement FC-MT-LSTM
3. If implementing FC-MT-LSTM, follow Phase 1 timeline
4. Build React dashboard in parallel (Week 4-5)
5. Write research paper (Week 7-8)

**Good luck with your research! 🚀**
