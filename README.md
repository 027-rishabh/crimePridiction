# Crime Prediction Models

This project implements seven different machine learning models for predicting crime rates across different districts and protected groups in India. The models are designed to be fair and unbiased across different demographic groups.

## Implemented Models

1. **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) - Traditional statistical time series model
2. **Prophet** - Facebook's forecasting model for time series with seasonality
3. **Random Forest** - Ensemble method using decision trees
4. **XGBoost** - Gradient boosting algorithm for tabular data
5. **CNN-LSTM** - Convolutional neural network combined with Long Short-Term Memory
6. **Transformer** - Attention-based neural network model
7. **Additional models** - Other advanced techniques

## Project Structure

```
crimePrediction/
├── data/
│   ├── splits/
│   │   ├── train_data.csv
│   │   └── test_data.csv
├── models/
│   ├── baseline_01_sarima.py
│   ├── baseline_02_prophet.py
│   ├── baseline_03_random_forest.py
│   ├── baseline_04_xgboost.py
│   ├── baseline_05_cnn_lstm.py
│   ├── baseline_06_transformer.py
│   └── fairness_metrics.py
├── results/
│   ├── model_predictions/
│   ├── fairness_metrics/
│   └── feature_importance/
├── run_all_baselines.py
├── export_results_for_react.py
└── requirements.txt
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Prophet installation may require additional steps on some systems:
```bash
# On some systems, you might need to install pystan separately
pip install pystan==2.19.1.1
pip install prophet
```

## Usage

### Train All Models
```bash
python run_all_baselines.py
```

This will:
- Train all 6 baseline models sequentially
- Generate predictions on the test set
- Calculate fairness metrics across protected groups
- Save results to the `results/` directory

### Export Results for Dashboard
```bash
python export_results_for_react.py
```

This creates a JSON file with all results formatted for React dashboard integration.

## Fairness Evaluation

The models are evaluated not just for accuracy, but for fairness across protected groups (SC, ST, General). The fairness metrics include:

- **Overall Performance**: MAE, RMSE, R², MAPE
- **Per-group Performance**: Performance metrics for each protected group
- **Fairness Gap**: Max difference in MAE across groups
- **Fairness Ratio**: Max MAE / Min MAE across groups

## Model Architecture Details

### SARIMA
- Traditional statistical model for time series
- Accounts for trend, seasonality, and autocorrelation
- Separate models trained for each district-protected group combination

### Prophet
- Facebook's forecasting model
- Handles missing data and irregular seasonality well
- Automatic changepoint detection

### Random Forest
- Ensemble of decision trees
- Handles mixed data types naturally
- Provides feature importance rankings

### XGBoost
- Gradient boosting implementation
- Often achieves high accuracy on tabular data
- Regularization options to prevent overfitting

### CNN-LSTM
- Hybrid neural network combining convolutional and recurrent layers
- Good for learning temporal patterns in sequential data
- Captures both spatial and temporal patterns

### Transformer
- Attention mechanism-based model
- Excels at capturing long-range dependencies
- Parallel computation capability

## Key Features

- Fairness-aware evaluation across protected groups
- Comprehensive metrics reporting
- Automated model training pipeline
- Clean, modular code structure
- Detailed documentation