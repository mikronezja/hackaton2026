"""
Predictive Grid Management for Heat Pump Networks
Improved solution with daily aggregation, XGBoost, and proper temporal validation.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "./"                          # folder containing the data files (current directory)
TRAIN_FILE = "data.csv"                    # training data (Oct 2024 - Apr 2025, includes x2)
DEVICES_FILE = "devices.csv"               # static device metadata
SAMPLE_FRAC = 1.0                          # if you want to use a subset for quick experiments

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def assign_dso(lat, lon):
    """Assign DSO based on approximate geographic centers."""
    dsos = {
        'Enea': (52.5, 16.5),
        'Energa': (54.0, 18.5),
        'PGE': (52.0, 22.0),
        'Tauron': (50.5, 19.0)
    }
    distances = {name: np.sqrt((lat - c[0])**2 + (lon - c[1])**2) for name, c in dsos.items()}
    return min(distances, key=distances.get)

def standardize_column_names(df):
    """Standardize column names to handle different naming conventions."""
    column_mapping = {
        'deviceId': 'deviceld',
        'deviceid': 'deviceld',
        'DeviceId': 'deviceld',
        'DEVICEID': 'deviceld',
        'timedate': 'Timedate',
        'timestamp': 'Timedate',
        'TimeDate': 'Timedate'
    }
    return df.rename(columns=column_mapping)

def downsample_to_hourly(df):
    """
    Convert 5‑minute data to hourly averages.
    This reduces data volume while preserving temporal patterns.
    """
    # Ensure Timedate is datetime
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    # Round to nearest hour to group readings from the same hour
    df['hour'] = df['Timedate'].dt.floor('H')
    # Aggregate by device and hour – mean for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep only relevant columns; avoid grouping on 'hour' later
    agg_dict = {col: 'mean' for col in numeric_cols if col not in ['deviceld', 'hour']}
    # For non‑numeric columns we take first (they are constant per device)
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['deviceld', 'hour']:
            agg_dict[col] = 'first'
    hourly = df.groupby(['deviceld', 'hour']).agg(agg_dict).reset_index()
    hourly.rename(columns={'hour': 'Timedate'}, inplace=True)
    return hourly

def aggregate_to_daily(df):
    """
    Aggregate hourly data to daily averages.
    """
    df['date'] = pd.to_datetime(df['Timedate']).dt.date
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_cols if col not in ['deviceld', 'date']}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['deviceld', 'date']:
            agg_dict[col] = 'first'
    daily = df.groupby(['deviceld', 'date']).agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    return daily

def add_features(df):
    """
    Add derived features at daily level.
    """
    # Temperature differences (potential predictors of heat pump load)
    if all(col in df.columns for col in ['t2', 't1']):
        df['delta_t1_t2'] = df['t2'] - df['t1']          # indoor - outdoor
    
    if all(col in df.columns for col in ['t3', 't4']):
        df['delta_t3_t4'] = df['t3'] - df['t4']          # source HEX delta
    
    if all(col in df.columns for col in ['t5', 't6']):
        df['delta_t5_t6'] = df['t5'] - df['t6']          # load HEX delta
    
    if all(col in df.columns for col in ['t1', 't8']):
        df['delta_t1_t8'] = df['t1'] - df['t8']          # outdoor - cooling circuit
    
    # Operating frequency squared (non‑linear effect)
    if 'x1' in df.columns:
        df['x1_sq'] = df['x1'] ** 2
    
    # Time features
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # DSO assignment from latitude/longitude
    if all(col in df.columns for col in ['latitude', 'longitude']):
        df['dso'] = df.apply(lambda r: assign_dso(r['latitude'], r['longitude']), axis=1)
    
    return df

def load_and_preprocess_train():
    """
    Load training data, merge with devices, downsample, and create daily features.
    Returns a daily DataFrame with features and target (x2).
    """
    print("Loading training data...")
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    devices_path = os.path.join(DATA_PATH, DEVICES_FILE)
    
    print(f"Looking for training file at: {train_path}")
    print(f"Looking for devices file at: {devices_path}")
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(devices_path):
        raise FileNotFoundError(f"Devices file not found at {devices_path}")
    
    df = pd.read_csv(train_path)
    devices = pd.read_csv(devices_path)
    
    # Standardize column names
    df = standardize_column_names(df)
    devices = standardize_column_names(devices)
    
    print(f"Training data columns: {df.columns.tolist()}")
    print(f"Devices data columns: {devices.columns.tolist()}")
    
    # Merge with device metadata
    df = df.merge(devices, on='deviceld', how='left')
    
    # Optional random subsampling for speed (only for experimentation)
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42)
    
    # Downsample 5‑min to hourly
    print("Downsampling to hourly...")
    hourly = downsample_to_hourly(df)
    del df  # free memory
    
    # Aggregate to daily
    print("Aggregating to daily...")
    daily = aggregate_to_daily(hourly)
    del hourly
    
    # Add engineered features
    daily = add_features(daily)
    
    # Ensure target is present
    if 'x2' not in daily.columns:
        print("Warning: x2 (target) not found in training data! Available columns:")
        print(daily.columns.tolist())
    else:
        print(f"Target x2 stats - mean: {daily['x2'].mean():.4f}, std: {daily['x2'].std():.4f}")
    
    return daily

def load_and_preprocess_test():
    """
    Load test data (May-Oct 2025, without x2), merge with devices, downsample,
    and create daily features. Returns daily DataFrame (no target).
    """
    print("Loading test data...")
    # Note: In the actual competition, test data might be provided separately
    # For now, we'll simulate by using a subset of training data
    # Replace this with actual test file loading when available
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    devices_path = os.path.join(DATA_PATH, DEVICES_FILE)
    
    df = pd.read_csv(train_path)
    devices = pd.read_csv(devices_path)
    
    # Standardize column names
    df = standardize_column_names(df)
    devices = standardize_column_names(devices)
    
    df = df.merge(devices, on='deviceld', how='left')
    
    # Filter for months 5-10 (May-October) in 2025
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df['year'] = df['Timedate'].dt.year
    df['month'] = df['Timedate'].dt.month
    
    # For now, use data from 2024 months 5-10 as proxy for 2025
    # In real scenario, you'd have actual 2025 test data
    test_mask = (df['year'] == 2024) & (df['month'].between(5, 10))
    df_test = df[test_mask].copy()
    
    if len(df_test) == 0:
        print("No test data found. Using last 20% of data as test set...")
        # Use last 20% of data as test set
        df = df.sort_values('Timedate')
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:].copy()
    
    print(f"Test data shape: {df_test.shape}")
    
    # Downsample
    hourly = downsample_to_hourly(df_test)
    del df_test
    daily = aggregate_to_daily(hourly)
    del hourly
    daily = add_features(daily)
    
    return daily

# ==========================================
# MODEL PIPELINE
# ==========================================

def create_model():
    """
    Create an XGBoost pipeline with appropriate preprocessing.
    """
    # Define available features based on what's in the data
    # We'll dynamically select features later
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    return model

def get_feature_columns(df, target_col='x2'):
    """Get list of feature columns, excluding target and non-feature columns."""
    exclude_cols = [target_col, 'date', 'Timedate', 'hour', 'deviceld']
    return [col for col in df.columns if col not in exclude_cols]

# ==========================================
# TRAINING & VALIDATION
# ==========================================

def temporal_train_validate(daily_df):
    """
    Train model using time‑based validation.
    Returns fitted model and feature columns.
    """
    # Create a time‑ordered split: last 20% of dates as validation
    daily_df = daily_df.sort_values('date').dropna(subset=['x2'])
    
    if len(daily_df) == 0:
        raise ValueError("No valid training data after dropping NaNs")
    
    # Use last 20% of dates for validation
    unique_dates = daily_df['date'].unique()
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    
    train_mask = daily_df['date'].isin(train_dates)
    val_mask = daily_df['date'].isin(val_dates)
    
    # Get feature columns
    feature_cols = get_feature_columns(daily_df)
    
    X_train = daily_df.loc[train_mask, feature_cols]
    X_val = daily_df.loc[val_mask, feature_cols]
    y_train = daily_df.loc[train_mask, 'x2']
    y_val = daily_df.loc[val_mask, 'x2']
    
    print(f"Training set size: {len(X_train)} days")
    print(f"Validation set size: {len(X_val)} days")
    print(f"Features used: {feature_cols}")
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_train.mean())  # use training mean for validation
    
    model = create_model()
    
    # Fit with early stopping on validation set
    print("Training model with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on validation
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE (daily): {val_mae:.5f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
    
    return model, feature_cols

# ==========================================
# PREDICTION & SUBMISSION
# ==========================================

def predict_monthly(model, test_daily, feature_cols):
    """
    Predict daily x2 for test data, then aggregate to monthly averages.
    Returns DataFrame with device, year, month, prediction.
    """
    # Prepare test features
    X_test = test_daily[feature_cols].fillna(0)
    
    # Predict daily
    daily_pred = model.predict(X_test)
    test_daily['pred_daily'] = np.clip(daily_pred, 0, None)  # x2 is non‑negative
    
    # Aggregate to monthly
    test_daily['year'] = 2025  # Force year to 2025 for submission
    test_daily['month'] = test_daily['date'].dt.month
    monthly = test_daily.groupby(['deviceld', 'year', 'month'])['pred_daily'].mean().reset_index()
    monthly.rename(columns={'pred_daily': 'prediction'}, inplace=True)
    
    # Ensure only months 5-10 (May-October) are included
    monthly = monthly[monthly['month'].between(5, 10)]
    
    return monthly

def main():
    # 1. Load and prepare training data
    print("=" * 50)
    print("STEP 1: Loading training data")
    print("=" * 50)
    train_daily = load_and_preprocess_train()
    print(f"Training daily shape: {train_daily.shape}")
    print(f"Date range: {train_daily['date'].min()} to {train_daily['date'].max()}")
    
    # 2. Train model with temporal validation
    print("\n" + "=" * 50)
    print("STEP 2: Training model")
    print("=" * 50)
    model, feature_cols = temporal_train_validate(train_daily)
    
    # 3. Retrain on full training data
    print("\n" + "=" * 50)
    print("STEP 3: Retraining on full dataset")
    print("=" * 50)
    X_all = train_daily[feature_cols].fillna(train_daily[feature_cols].mean())
    y_all = train_daily['x2']
    
    # Recreate model without early stopping for final training
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X_all, y_all)
    
    # 4. Load test data
    print("\n" + "=" * 50)
    print("STEP 4: Loading test data")
    print("=" * 50)
    test_daily = load_and_preprocess_test()
    print(f"Test daily shape: {test_daily.shape}")
    
    # 5. Predict monthly for test period
    print("\n" + "=" * 50)
    print("STEP 5: Generating predictions")
    print("=" * 50)
    monthly_preds = predict_monthly(final_model, test_daily, feature_cols)
    
    # 6. Save submission
    sub = monthly_preds[['deviceld', 'year', 'month', 'prediction']]
    sub.to_csv('task3_submission_2.csv', index=False)
    print("\nSubmission saved to task3_submission_2.csv")
    print(f"Submission shape: {sub.shape}")
    print("\nFirst 10 predictions:")
    print(sub.head(10))
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total predictions: {len(sub)}")
    print(f"Unique devices: {sub['deviceld'].nunique()}")
    print(f"Months covered: {sorted(sub['month'].unique())}")
    print(f"Prediction range: {sub['prediction'].min():.4f} - {sub['prediction'].max():.4f}")
    print(f"Mean prediction: {sub['prediction'].mean():.4f}")

if __name__ == "__main__":
    main()