"""
features.py
============
Reusable, leakage-safe feature engineering pipeline for SmartWater-AI
(built on the REAL Kaggle dataset: region, date, consumption_liters).

Import build_features() from anywhere (training script, Flask app) so the
exact same transformation is applied at train time and at inference time.

Feature set (all legitimate for this dataset — NO invented weather data):
    region_*          one-hot encoded region (North/South/East/West/Central)
    month             calendar month (1-12)
    day_of_week       weekday (0=Monday .. 6=Sunday)
    is_weekend        1 if Saturday/Sunday else 0
    lag_1             consumption 1 day ago (same region)
    lag_7             consumption 7 days ago (same region)
    rolling_mean_7    mean of previous 7 days (same region)
    rolling_mean_14   mean of previous 14 days (same region)

Leakage-safety notes:
  - All lag / rolling features are computed PER REGION (groupby) so a row
    never sees another region's values, and never its own or a future day's
    target (shift(1) before rolling).
  - Train/test split downstream must remain chronological (no shuffling)
    to reflect real forecasting conditions.
"""

import numpy as np
import pandas as pd

TARGET_COL = "consumption_liters"
DATE_COL = "date"
REGION_COL = "region"

# Calendar + time-series features (region one-hot columns are added
# dynamically by build_features() based on the regions actually present).
BASE_FEATURES = [
    "month",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
]

# Final feature columns (populated by build_features() once regions are known).
FEATURE_COLUMNS = []


def build_features(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Build the full feature set from a cleaned dataframe.

    Parameters
    ----------
    df : cleaned dataframe containing at least region, date and (if
         for_training) consumption_liters.
    for_training : if True, keeps the target column and drops the initial
         rows where lag/rolling features are undefined (NaN warm-up window).
         If False (inference), the caller must supply pre-computed
         lag_1/lag_7/rolling_mean_7/rolling_mean_14 (see src/predict.py).

    Returns
    -------
    DataFrame with [date, region] + FEATURE_COLUMNS (+ target if for_training).
    Also updates the module-level FEATURE_COLUMNS with the exact column order.
    """
    global FEATURE_COLUMNS

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([DATE_COL, REGION_COL]).reset_index(drop=True)

    # --- calendar features ---
    out["month"] = out[DATE_COL].dt.month
    out["day_of_week"] = out[DATE_COL].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # --- region one-hot encoding (deterministic column order) ---
    region_dummies = pd.get_dummies(out[REGION_COL], prefix="region").astype(int)
    region_dummies = region_dummies[sorted(region_dummies.columns)]
    out = pd.concat([out, region_dummies], axis=1)

    if for_training:
        # --- lag / rolling features computed PER REGION from PAST values ---
        # shift(1) BEFORE rolling so today's value never leaks into its own
        # 7/14-day rolling average; rolling is grouped per region so windows
        # never mix values from different regions.
        shifted_target = out.groupby(REGION_COL, sort=False)[TARGET_COL].shift(1)
        out["lag_1"] = shifted_target
        out["lag_7"] = out.groupby(REGION_COL, sort=False)[TARGET_COL].shift(7)
        out["rolling_mean_7"] = (
            shifted_target.groupby(out[REGION_COL])
            .rolling(window=7, min_periods=7)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["rolling_mean_14"] = (
            shifted_target.groupby(out[REGION_COL])
            .rolling(window=14, min_periods=14)
            .mean()
            .reset_index(level=0, drop=True)
        )

    FEATURE_COLUMNS = list(region_dummies.columns) + BASE_FEATURES

    if for_training:
        # drop warm-up rows where lag/rolling features are NaN (no leakage-prone fill)
        out = out.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]).reset_index(drop=True)
        keep_cols = [DATE_COL, REGION_COL] + FEATURE_COLUMNS + [TARGET_COL]
    else:
        keep_cols = [DATE_COL, REGION_COL] + FEATURE_COLUMNS

    return out[keep_cols]


def chronological_split(feat_df: pd.DataFrame, test_size: float = 0.2):
    """
    Chronological (no shuffle) train/test split — required for time series.

    The split is performed PER REGION: each region keeps its first (1-test_size)
    fraction of days for training and its most recent days for testing. This
    keeps every region's test set strictly AFTER its training set in time
    (true out-of-sample forecasting) and avoids overlapping a date boundary
    when regions are interleaved.
    """
    feat_df = feat_df.sort_values([DATE_COL, REGION_COL]).reset_index(drop=True)
    train_parts, test_parts = [], []
    for _, g in feat_df.groupby(REGION_COL, sort=False):
        split_idx = int(len(g) * (1 - test_size))
        train_parts.append(g.iloc[:split_idx])
        test_parts.append(g.iloc[split_idx:])
    train = pd.concat(train_parts).sort_values([DATE_COL, REGION_COL]).reset_index(drop=True)
    test = pd.concat(test_parts).sort_values([DATE_COL, REGION_COL]).reset_index(drop=True)
    return train, test