"""
prepare_data.py
================
DATA STAGE for SmartWater-AI.

PRIMARY DATA SOURCE (REAL):
    data/raw/water_consumption_forecasting.csv
    Columns: region, date, consumption_liters
    900 observations, 5 regions x 180 days (2023-01-01 -> 2023-06-29).

The real dataset contains ONLY region/date/consumption_liters. There is no
temperature, humidity or holiday column, so the pipeline does NOT invent
them. Downstream feature engineering uses only legitimate calendar +
time-series features (see src/features.py).

SYNTHETIC BACKUP:
The earlier synthetic generator (weather-driven demand with injected data
quality issues) is kept purely as a backup. Run with `--synthetic` to
regenerate it; it writes to data/processed/water_demand_synthetic_clean.csv
so it never overwrites the real cleaned dataset.

Usage:
    python data/prepare_data.py            # real Kaggle dataset -> processed
    python data/prepare_data.py --synthetic  # backup synthetic dataset

Outputs:
    data/processed/water_demand_clean.csv  (cleaned REAL dataset, used downstream)
    data/processed/water_demand_synthetic_clean.csv  (backup only)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
PROCESSED_DIR = os.path.join(HERE, "processed")

REAL_RAW_PATH = os.path.join(RAW_DIR, "water_consumption_forecasting.csv")
REAL_CLEAN_PATH = os.path.join(PROCESSED_DIR, "water_demand_clean.csv")
SYNTHETIC_CLEAN_PATH = os.path.join(PROCESSED_DIR, "water_demand_synthetic_clean.csv")

REAL_COLUMNS = ["region", "date", "consumption_liters"]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. CLEAN THE REAL KAGGLE DATASET
# --------------------------------------------------------------------------
def clean_real_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the real (region, date, consumption_liters) dataset."""
    df = df_raw.copy()

    # --- Parse dates robustly; drop unparsable rows ----------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # --- Drop rows with missing region or consumption ---------------------
    n_missing = int(df[REAL_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=REAL_COLUMNS)

    # --- Remove exact duplicate rows --------------------------------------
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    # --- Remove duplicate (region, date) keys, keep first chronologically --
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["region", "date"], keep="first")

    # --- Consumption must be a positive number ----------------------------
    n_invalid = int((df["consumption_liters"] <= 0).sum())
    df = df[df["consumption_liters"] > 0]

    # --- Standardize column order and types -------------------------------
    df = df[REAL_COLUMNS].reset_index(drop=True)
    df["consumption_liters"] = df["consumption_liters"].astype(float)

    report = {
        "unparsable_dates_dropped": n_bad_dates,
        "missing_value_rows_dropped": n_missing,
        "duplicate_rows_removed": n_dupes,
        "non_positive_consumption_dropped": n_invalid,
        "rows_after_cleaning": int(len(df)),
        "regions": sorted(df["region"].unique().tolist()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
    }
    return df, report


# --------------------------------------------------------------------------
# 2. SYNTHETIC BACKUP (kept only as a fallback dataset)
# --------------------------------------------------------------------------
def generate_synthetic_raw_dataset(n_days=1095, start="2021-01-01") -> pd.DataFrame:
    """Recreates the old synthetic weather-driven dataset (backup only)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    day_of_year = dates.dayofyear.values
    weekday = dates.dayofweek.values
    t = np.arange(n_days)

    temperature = 22 + 9 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) \
        + rng.normal(0, 1.8, n_days)
    humidity = 60 - 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) \
        + rng.normal(0, 5, n_days)
    humidity = np.clip(humidity, 20, 95)

    holiday_dates = set()
    for y in sorted(set(dates.year)):
        for md in [(1, 1), (1, 26), (5, 1), (8, 15), (10, 2), (12, 25)]:
            try:
                holiday_dates.add(pd.Timestamp(year=y, month=md[0], day=md[1]))
            except ValueError:
                pass
    holiday = dates.isin(holiday_dates).astype(int)

    seasonal_component = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    weekend_component = np.where(weekday >= 5, 6.0, 0.0)
    holiday_component = holiday * 4.5
    trend_component = t * 0.01
    temp_component = (temperature - 22) * 0.9
    humidity_component = (60 - humidity) * 0.05
    noise = rng.normal(0, 3.0, n_days)

    demand = (
        50 + seasonal_component + weekend_component + holiday_component
        + trend_component + temp_component + humidity_component + noise
    )
    demand = np.clip(demand, 5, None)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "is_holiday": holiday,
        "water_demand_ml": np.round(demand, 2),
    })
    return df


def build_synthetic_backup() -> pd.DataFrame:
    """Generate + clean the synthetic dataset (backup)."""
    raw = generate_synthetic_raw_dataset()
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"]).drop_duplicates()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    df.loc[(df["humidity_pct"] < 0) | (df["humidity_pct"] > 100), "humidity_pct"] = np.nan
    df.loc[(df["temperature_c"] < -40) | (df["temperature_c"] > 60), "temperature_c"] = np.nan
    df.loc[df["water_demand_ml"] < 0, "water_demand_ml"] = np.nan

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    for col in ["temperature_c", "humidity_pct", "water_demand_ml"]:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df.reset_index()


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        print("Regenerating SYNTHETIC BACKUP dataset (not used by the app)...")
        synthetic_df = build_synthetic_backup()
        synthetic_df.to_csv(SYNTHETIC_CLEAN_PATH, index=False)
        print(f"Synthetic backup saved -> {SYNTHETIC_CLEAN_PATH} ({len(synthetic_df)} rows)")
        print("NOTE: the app uses the REAL dataset. Synthetic is backup only.")
    else:
        print(f"Loading REAL Kaggle dataset from {REAL_RAW_PATH} ...")
        raw_df = pd.read_csv(REAL_RAW_PATH)
        clean_df, report = clean_real_dataset(raw_df)
        clean_df.to_csv(REAL_CLEAN_PATH, index=False)
        print(f"Cleaned dataset saved -> {REAL_CLEAN_PATH} ({len(clean_df)} rows)")
        print("\nCleaning report:")
        for k, v in report.items():
            print(f"  - {k}: {v}")
        print("\nPreview:")
        print(clean_df.head())