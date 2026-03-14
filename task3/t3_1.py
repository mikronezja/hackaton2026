import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. FUNKCJE POMOCNICZE (UTILITIES)
# ==========================================

def assign_dso(lat, lon):
    """Przypisuje operatora (DSO) na podstawie współrzędnych geograficznych."""
    dsos = {
        'Enea': (52.5, 16.5),    # Zachód
        'Energa': (54.0, 18.5),  # Północ
        'PGE': (52.0, 22.0),     # Wschód/Centrum
        'Tauron': (50.5, 19.0)   # Południe
    }
    # Obliczanie najkrótszego dystansu euklidesowego do centrum operatora
    distances = {name: np.sqrt((lat-c[0])**2 + (lon-c[1])**2) for name, c in dsos.items()}
    return min(distances, key=distances.get)

# ==========================================
# 2. PRZETWARZANIE DANYCH
# ==========================================

def load_and_preprocess(data_path, devices_path, sample_fraction=1.0):
    """Wczytuje dane, łączy je i przygotowuje ramy czasowe."""
    print("Wczytywanie i łączenie danych...")
    df = pd.read_csv(data_path)
    devices = pd.read_csv(devices_path)
    
    # Subsampling (Slicing) dla wydajności, jeśli wskazano
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
    
    df = df.merge(devices, on='deviceld', how='left')
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df['year'] = df['Timedate'].dt.year
    df['month'] = df['Timedate'].dt.month
    
    return df

def aggregate_to_monthly(df, is_train=True):
    """Agreguje dane 5-minutowe do średnich miesięcznych (Top-Down)."""
    print("Agregacja do poziomu miesiąca...")
    
    agg_dict = {
        't1': 'mean', 't2': 'mean',
        'latitude': 'first', 'longitude': 'first',
        'deviceType': 'first'
    }
    if is_train:
        agg_dict['x2'] = 'mean'
        
    monthly_df = df.groupby(['deviceld', 'year', 'month']).agg(agg_dict).reset_index()
    
    # Inżynieria cech
    monthly_df['dso'] = monthly_df.apply(lambda x: assign_dso(x['latitude'], x['longitude']), axis=1)
    monthly_df['delta_T'] = monthly_df['t2'] - monthly_df['t1']
    
    return monthly_df

# ==========================================
# 3. ARCHITEKTURA MODELU
# ==========================================

def create_model_pipeline(numeric_features, categorical_features):
    """Tworzy zaawansowany Pipeline z obsługą nieliniowości i kategorii."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)) # Nieliniowość
    ])
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])

# ==========================================
# 4. EKSPERYMENT I WALIDACJA
# ==========================================

def run_validation(df_monthly, numeric_features, categorical_features):
    """Przeprowadza walidację czasową (TimeSeriesSplit)."""
    df_monthly = df_monthly.sort_values(['year', 'month'])
    X = df_monthly[numeric_features + categorical_features]
    y = df_monthly['x2']
    
    tscv = TimeSeriesSplit(n_splits=3)
    pipeline = create_model_pipeline(numeric_features, categorical_features)
    
    print("\n--- Walidacja Czasowa (MAE) ---")
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        preds = np.clip(pipeline.predict(X_val), 0, None)
        
        score = mean_absolute_error(y_val, preds)
        scores.append(score)
        print(f"Fold MAE: {score:.5f}")
    
    print(f"Średni błąd walidacji: {np.mean(scores):.5f}\n")
    return pipeline

# ==========================================
# 5. GŁÓWNY PROCES (EXECUTION)
# ==========================================

def main():
    # Definicja cech
    num_cols = ['t1', 't2', 'delta_T']
    cat_cols = ['dso', 'deviceType']
    
    # 1. Dane Mock (zastąp load_and_preprocess dla prawdziwych danych)
    mock_data = pd.DataFrame({
        'deviceld': ['A']*100 + ['B']*100,
        'Timedate': pd.date_range(start='2024-10-01', periods=200, freq='D'),
        't1': np.random.normal(0, 5, 200), 't2': np.random.normal(21, 1, 200),
        'x2': np.random.uniform(0.1, 2.0, 200),
        'latitude': [52.0]*100 + [50.0]*100, 'longitude': [19.0]*100 + [18.0]*100,
        'deviceType': [1]*100 + [2]*100
    })
    mock_data['year'] = mock_data['Timedate'].dt.year
    mock_data['month'] = mock_data['Timedate'].dt.month

    # 2. Przetwarzanie i Walidacja
    train_monthly = aggregate_to_monthly(mock_data, is_train=True)
    model = run_validation(train_monthly, num_cols, cat_cols)
    
    # 3. Finalny trening na wszystkich danych zimowych
    model.fit(train_monthly[num_cols + cat_cols], train_monthly['x2'])
    
    # 4. Generowanie predykcji na lato (Maj - Październik 2025)
    test_devices = train_monthly['deviceld'].unique()
    test_months = [5, 6, 7, 8, 9, 10]
    
    prediction_rows = []
    for dev in test_devices:
        for m in test_months:
            # W prawdziwym zadaniu użyj średnich t1, t2 z danych testowych
            prediction_rows.append({
                'deviceld': dev, 'year': 2025, 'month': m,
                't1': 15.0, 't2': 22.0, 'delta_T': 7.0,
                'dso': assign_dso(52.0, 19.0), 'deviceType': 1
            })
            
    df_test = pd.DataFrame(prediction_rows)
    preds = model.predict(df_test[num_cols + cat_cols])
    df_test['prediction'] = np.clip(preds, 0, None)
    
    # 5. Zapis wyniku
    submission = df_test[['deviceld', 'year', 'month', 'prediction']]
    submission.to_csv('task3_submission.csv', index=False)
    print("Zapisano submission.csv. Podgląd:")
    print(submission.head())

if __name__ == "__main__":
    main()