"""
Task 3 – Predictive Grid Management for Heat Pump Networks
Ensemble: Ridge + LightGBM + Historical baseline  →  weighted average
Target  : average x2 per (deviceId, year, month)  for May–Oct 2025
Metric  : MAE (lower = better)

Directory layout expected:
    data/
        data.csv
        devices.csv
    solution_task3.py   ← this file

Output:
    submission.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = Path("")
OUTPUT_CSV = Path("submission.csv")

TRAIN_END_MONTH  = (2025, 4)   # inclusive  (Oct 2024 – Apr 2025)
VAL_MONTHS       = [(2025, 5), (2025, 6)]
TEST_MONTHS      = [(2025, m) for m in range(7, 11)]
FORECAST_MONTHS  = VAL_MONTHS + TEST_MONTHS   # May–Oct 2025

# Features available in ALL splits (x2 withheld in val/test)
BASE_FEATURES = [
    "t1_mean", "t1_min", "t1_max", "t1_std",
    "t2_mean",
    "t7_mean", "t7_min",
    "t9_mean",
    "t10_mean",
    "x1_mean", "x1_std",
    "month_sin", "month_cos",
    "n_readings",
]

RIDGE_ALPHA    = 10.0
LGBM_PARAMS    = dict(
    n_estimators      = 1000,
    learning_rate     = 0.03,
    num_leaves        = 31,
    min_child_samples = 3,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)
LGBM_EARLY_STOP = 30


# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data …")

    # ── Sniff column names before committing to dtypes ──────────────────────
    header = pd.read_csv(DATA_DIR / "data.csv", nrows=0)
    print(f"  Columns found: {list(header.columns)}")

    # Detect timestamp column (case-insensitive, common variants)
    ts_candidates = ["Timedate", "timedate", "Timestamp", "timestamp",
                     "datetime", "DateTime", "time", "Time", "date", "Date"]
    ts_col = next((c for c in ts_candidates if c in header.columns), None)
    if ts_col is None:
        ts_col = header.columns[0]
        print(f"  WARNING: timestamp column not found by name, using '{ts_col}'")
    else:
        print(f"  Timestamp column: '{ts_col}'")

    # Build dtype map
    float_cols = [c for c in [f"t{i}" for i in range(1, 14)] + ["x1", "x2", "x3"]
                  if c in header.columns]
    dtypes = {c: "float32" for c in float_cols}
    if "deviceId"   in header.columns: dtypes["deviceId"]   = "str"
    if "deviceType" in header.columns: dtypes["deviceType"] = "Int8"

    # Avoid parse_dates during load for speed; do it vectorized afterwards
    df = pd.read_csv(
        DATA_DIR / "data.csv",
        dtype=dtypes,
        low_memory=False,
    )

    if ts_col != "Timedate":
        df = df.rename(columns={ts_col: "Timedate"})
    
    # Vectorized fast parsing 
    df["Timedate"] = pd.to_datetime(df["Timedate"], cache=True)

    devices = pd.read_csv(DATA_DIR / "devices.csv", dtype={"deviceId": "str"})

    df = df.merge(devices, on="deviceId", how="left")
    df["year"]  = df["Timedate"].dt.year.astype("int16")
    df["month"] = df["Timedate"].dt.month.astype("int8")
    print(f"  Rows: {len(df):,}  |  Devices: {df['deviceId'].nunique()}")
    return df, devices


# ─────────────────────────────────────────────
# 2. MONTHLY AGGREGATION
# ─────────────────────────────────────────────
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating to monthly …")
    agg = df.groupby(["deviceId", "year", "month"]).agg(
        t1_mean  = ("t1", "mean"),
        t1_min   = ("t1", "min"),
        t1_max   = ("t1", "max"),
        t1_std   = ("t1", "std"),
        t2_mean  = ("t2", "mean"),
        t7_mean  = ("t7", "mean"),
        t7_min   = ("t7", "min"),
        t9_mean  = ("t9", "mean"),
        t10_mean = ("t10", "mean"),
        x1_mean  = ("x1", "mean"),
        x1_std   = ("x1", "std"),
        x2_mean  = ("x2", "mean"),
        n_readings = ("x2", "count"),
    ).reset_index()

    agg["month_sin"] = np.sin(2 * np.pi * agg["month"] / 12).astype("float32")
    agg["month_cos"] = np.cos(2 * np.pi * agg["month"] / 12).astype("float32")
    agg["n_readings"] = agg["n_readings"].astype("int32")
    
    return agg


# ─────────────────────────────────────────────
# 3. SPLITS
# ─────────────────────────────────────────────
def make_splits(monthly: pd.DataFrame):
    def _filter(ym_list):
        # Fast inner join instead of looping masks
        ym_df = pd.DataFrame(ym_list, columns=["year", "month"])
        return monthly.merge(ym_df, on=["year", "month"], how="inner")

    train_mask = (
        monthly["x2_mean"].notna()
        & (
            (monthly["year"] < TRAIN_END_MONTH[0])
            | (
                (monthly["year"] == TRAIN_END_MONTH[0])
                & (monthly["month"] <= TRAIN_END_MONTH[1])
            )
        )
    )
    train = monthly[train_mask].copy()
    val   = _filter(VAL_MONTHS)
    test  = _filter(TEST_MONTHS)

    print(f"  Train rows: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────
# 4. HISTORICAL BASELINE  (Model C)
# ─────────────────────────────────────────────
def build_historical_baseline(train: pd.DataFrame):
    dev_month = train.groupby(["deviceId", "month"])["x2_mean"].mean().rename("pred_hist").reset_index()
    global_month = train.groupby("month")["x2_mean"].mean().rename("pred_hist_global").reset_index()
    return dev_month, global_month


def predict_historical(df: pd.DataFrame, dev_month, global_month) -> np.ndarray:
    merged = df.merge(dev_month, on=["deviceId", "month"], how="left")
    merged = merged.merge(global_month, on="month", how="left")
    return merged["pred_hist"].fillna(merged["pred_hist_global"]).values.astype("float32")


# ─────────────────────────────────────────────
# 5. RIDGE REGRESSION  (Model A)
# ─────────────────────────────────────────────
def train_ridge(train: pd.DataFrame, val: pd.DataFrame):
    print("Training Ridge …")
    X_tr = train[BASE_FEATURES].values
    y_tr = train["x2_mean"].values
    X_v  = val[BASE_FEATURES].values

    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(X_tr, y_tr)

    return model, model.predict(X_v)


def predict_ridge(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[BASE_FEATURES].values).astype("float32")


# ─────────────────────────────────────────────
# 6. LIGHTGBM  (Model B)
# ─────────────────────────────────────────────
def train_lgbm(train: pd.DataFrame, val: pd.DataFrame):
    print("Training LightGBM …")
    X_tr = train[BASE_FEATURES]
    y_tr = train["x2_mean"].values
    X_v  = val[BASE_FEATURES]
    y_v  = val["x2_mean"].values  

    has_gt = ~np.isnan(y_v)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    if has_gt.sum() > 0:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_v[has_gt], y_v[has_gt])],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOP, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
    else:
        model.set_params(n_estimators=500)
        model.fit(X_tr, y_tr)

    return model, model.predict(X_v)


def predict_lgbm(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[BASE_FEATURES].values).astype("float32")


# ─────────────────────────────────────────────
# 7. ENSEMBLE WEIGHT OPTIMISATION
# ─────────────────────────────────────────────
def optimise_weights(preds_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(y_val)
    if mask.sum() == 0:
        print("  No val GT available – using equal weights")
        return np.array([1/3, 1/3, 1/3])

    pv = preds_val[mask]
    yv = y_val[mask]

    def obj(w):
        w = np.abs(w)
        w = w / w.sum()
        return np.mean(np.abs((pv @ w) - yv))

    best_result = None
    for w0 in [[1/3,1/3,1/3], [0.2,0.6,0.2], [0.1,0.7,0.2], [0.3,0.4,0.3]]:
        r = minimize(obj, w0, method="Nelder-Mead", options={"xatol":1e-6, "fatol":1e-6, "maxiter":5000})
        if best_result is None or r.fun < best_result.fun:
            best_result = r

    w_opt = np.abs(best_result.x)
    w_opt /= w_opt.sum()
    print(f"  Optimal weights  Ridge={w_opt[0]:.3f}  LGBM={w_opt[1]:.3f}  Hist={w_opt[2]:.3f}")
    print(f"  Val MAE (blended)= {best_result.fun:.6f}")
    return w_opt


# ─────────────────────────────────────────────
# 8. FORECAST FOR SUBMISSION MONTHS
# ─────────────────────────────────────────────
def build_forecast_frame(monthly: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    # Massively faster grid generation using cross merge
    ym_df = pd.DataFrame(FORECAST_MONTHS, columns=["year", "month"])
    forecast = devices[["deviceId"]].merge(ym_df, how="cross")

    forecast = forecast.merge(
        monthly.drop(columns=["x2_mean"], errors="ignore"),
        on=["deviceId", "year", "month"],
        how="left",
    )

    dev_feat_means = monthly.groupby("deviceId")[BASE_FEATURES].mean()

    # Fast vectorized imputation (No slow `.loc` masks)
    for feat in BASE_FEATURES:
        if feat in ("month_sin", "month_cos", "n_readings"):
            continue
        # Device-level fallback
        forecast[feat] = forecast[feat].fillna(forecast["deviceId"].map(dev_feat_means[feat]))
        # Global fallback
        forecast[feat] = forecast[feat].fillna(monthly[feat].mean())

    forecast["month_sin"] = np.sin(2 * np.pi * forecast["month"] / 12).astype("float32")
    forecast["month_cos"] = np.cos(2 * np.pi * forecast["month"] / 12).astype("float32")
    forecast["n_readings"] = forecast["n_readings"].fillna(monthly["n_readings"].median()).astype("int32")

    return forecast


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    df_raw, devices = load_raw()
    monthly = aggregate_monthly(df_raw)

    train, val, test = make_splits(monthly)
    dev_month, global_month = build_historical_baseline(train)

    forecast = build_forecast_frame(monthly, devices)
    print(f"Forecast frame shape: {forecast.shape}")

    val_mask = forecast["month"].isin([5, 6]) & (forecast["year"] == 2025)
    val_fc   = forecast[val_mask].reset_index(drop=True)

    model_a, pred_a_val = train_ridge(train, val_fc)
    model_b, pred_b_val = train_lgbm(train, val_fc)
    pred_c_val = predict_historical(val_fc, dev_month, global_month)

    y_val_gt = val_fc["x2_mean"].values if "x2_mean" in val_fc.columns else np.full(len(val_fc), np.nan)
    preds_val_stack = np.column_stack([pred_a_val, pred_b_val, pred_c_val])
    w_opt = optimise_weights(preds_val_stack, y_val_gt)

    mask_gt = ~np.isnan(y_val_gt)
    if mask_gt.sum() > 0:
        print(f"  Ridge MAE (val)  = {mean_absolute_error(y_val_gt[mask_gt], pred_a_val[mask_gt]):.6f}")
        print(f"  LGBM  MAE (val)  = {mean_absolute_error(y_val_gt[mask_gt], pred_b_val[mask_gt]):.6f}")
        print(f"  Hist  MAE (val)  = {mean_absolute_error(y_val_gt[mask_gt], pred_c_val[mask_gt]):.6f}")

    pred_a_all = predict_ridge(model_a, forecast)
    pred_b_all = predict_lgbm(model_b, forecast)
    pred_c_all = predict_historical(forecast, dev_month, global_month)

    preds_all  = np.column_stack([pred_a_all, pred_b_all, pred_c_all])
    final_pred = (preds_all @ w_opt).clip(0, 1)

    submission = forecast[["deviceId", "year", "month"]].copy()
    submission["prediction"] = final_pred.astype("float32")
    submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)

    expected = len(devices) * 6
    assert len(submission) == expected, f"Expected {expected} rows, got {len(submission)}"

    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}  ({len(submission):,} rows)")
    print(submission.head(12).to_string(index=False))


if __name__ == "__main__":
    main()