"""
train_models.py
================
Trains and evaluates water-demand forecasting models on the REAL Kaggle
dataset (region, date, consumption_liters):

    1. Baseline  - previous-day-value ("naive") forecast
    2. Random Forest Regressor
    3. Gradient Boosting Regressor

Methodology (defensible in a viva):
  - Features are calendar + time-series only (no invented weather):
    region (one-hot), month, day_of_week, is_weekend, lag_1, lag_7,
    rolling_mean_7, rolling_mean_14.
  - Lag/rolling features are computed per region and use ONLY past values
    (see src/features.py) - no data leakage.
  - Strict CHRONOLOGICAL 80/20 split (no shuffle): the model is trained on
    the first 80% of the timeline and evaluated on the last 20%, which is
    exactly how it would be deployed (predicting the future).

Saves trained models, the feature list, recent per-region history, and
honest evaluation metrics (MAE, RMSE, R2) into /models for reuse by app.py.

Run:
    python train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import src.features as features
from src.features import build_features, chronological_split, TARGET_COL, REGION_COL, DATE_COL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "processed", "water_demand_clean.csv")
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "R2": round(float(r2), 4)}


def main():
    print(f"Loading REAL cleaned dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  shape: {df.shape} | columns: {list(df.columns)}")
    print(f"  regions: {sorted(df[REGION_COL].unique())}")
    print(f"  date range: {df[DATE_COL].min()} -> {df[DATE_COL].max()}")

    print("\nBuilding features (leakage-safe, chronological, per-region lags) ...")
    feat_df = build_features(df, for_training=True)
    FEATURE_COLUMNS = features.FEATURE_COLUMNS
    print(f"  feature dataframe: {feat_df.shape[0]} rows, {len(FEATURE_COLUMNS)} features")
    print(f"  features: {FEATURE_COLUMNS}")

    train_df, test_df = chronological_split(feat_df, test_size=0.2)
    print(f"\nChronological 80/20 split -> train: {len(train_df)} rows, test: {len(test_df)} rows")
    print(f"  train period: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}")
    print(f"  test period : {test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL]

    results = {}

    # ------------------------------------------------------------------
    # 1. BASELINE: previous-day value forecast (naive forecast = lag_1)
    # ------------------------------------------------------------------
    print("\n[1/3] Evaluating baseline (previous-value naive forecast) ...")
    baseline_pred = X_test["lag_1"].values
    results["Baseline"] = evaluate(y_test, baseline_pred)
    print(f"  Baseline -> {results['Baseline']}")

    # ------------------------------------------------------------------
    # 2. RANDOM FOREST
    # ------------------------------------------------------------------
    print("\n[2/3] Training Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = evaluate(y_test, rf_pred)
    print(f"  Random Forest -> {results['RandomForest']}")

    # ------------------------------------------------------------------
    # 3. GRADIENT BOOSTING
    # ------------------------------------------------------------------
    print("\n[3/3] Training Gradient Boosting ...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    results["GradientBoosting"] = evaluate(y_test, gb_pred)
    print(f"  Gradient Boosting -> {results['GradientBoosting']}")

    # ------------------------------------------------------------------
    # Pick best model by R2 (only the two ML models compete; the baseline
    # is a reference point, not a candidate "best").
    # ------------------------------------------------------------------
    model_objs = {"RandomForest": rf, "GradientBoosting": gb}
    best_name = max(model_objs.keys(), key=lambda k: results[k]["R2"])
    print(f"\nBest model by R2: {best_name} ({results[best_name]})")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    joblib.dump(gb, os.path.join(MODELS_DIR, "gradient_boosting_model.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_list.joblib"))

    metrics_payload = {
        "results": results,
        "best_model": best_name,
        "target_column": TARGET_COL,
        "feature_columns": FEATURE_COLUMNS,
        "split": {"method": "chronological_80_20_no_shuffle", "test_size": 0.2},
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": [str(train_df[DATE_COL].min().date()), str(train_df[DATE_COL].max().date())],
        "test_date_range": [str(test_df[DATE_COL].min().date()), str(test_df[DATE_COL].max().date())],
        "dataset": {
            "source": "data/raw/water_consumption_forecasting.csv",
            "rows": int(len(df)),
            "regions": sorted(df[REGION_COL].unique().tolist()),
            "date_range": [str(pd.to_datetime(df[DATE_COL]).min().date()), str(pd.to_datetime(df[DATE_COL]).max().date())],
        },
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save a recent per-region history slice (needed by the Flask app to
    # compute lag_1 / lag_7 / rolling means for NEW predictions, per region).
    df_sorted = df.copy()
    df_sorted[DATE_COL] = pd.to_datetime(df_sorted[DATE_COL])
    df_sorted = df_sorted.sort_values([DATE_COL, REGION_COL])
    recent_history = (
        df_sorted.groupby(REGION_COL, sort=False)
        .tail(30)
        .sort_values([DATE_COL, REGION_COL])
        .reset_index(drop=True)
    )
    recent_history.to_csv(os.path.join(MODELS_DIR, "recent_history.csv"), index=False)

    print(f"\nSaved models + metrics + feature list + recent history -> {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()