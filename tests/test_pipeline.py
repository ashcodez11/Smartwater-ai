"""
test_pipeline.py
================
Contract tests for the data + model pipeline.

These lock down the facts the project's reported results depend on, so that any
future edit which silently breaks them (a reordered feature list, a shuffled
split, drifted thresholds, a contaminated forecast buffer) fails the suite
instead of quietly changing the science.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

import src.features as features
from src.features import TARGET_COL, build_features, chronological_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "water_demand_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLASSES = ["LOW", "NORMAL", "HIGH", "VERY HIGH"]


@pytest.fixture(scope="module")
def raw_df():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@pytest.fixture(scope="module")
def feature_frames(raw_df):
    feat = build_features(raw_df, for_training=True)
    train, test = chronological_split(feat, test_size=0.2)
    return feat, train, test


# --------------------------------------------------------------------- data
def test_dataset_shape_and_columns(raw_df):
    assert raw_df.shape == (900, 3)
    assert list(raw_df.columns) == ["region", "date", "consumption_liters"]
    assert sorted(raw_df["region"].unique()) == ["Central", "East", "North", "South", "West"]


def test_dataset_is_clean(raw_df):
    assert int(raw_df.isna().sum().sum()) == 0
    assert int(raw_df.duplicated().sum()) == 0
    assert (raw_df["consumption_liters"] > 0).all()


def test_every_region_has_180_days(raw_df):
    counts = raw_df.groupby("region").size()
    assert set(counts.values) == {180}
    assert raw_df["date"].min() == pd.Timestamp("2023-01-01")
    assert raw_df["date"].max() == pd.Timestamp("2023-06-29")


# ----------------------------------------------------------------- features
def test_feature_frame_size_and_warmup_dropped(feature_frames):
    feat, _, _ = feature_frames
    # 14-day warm-up per region is dropped, not imputed: 900 - (14 x 5) = 830
    assert len(feat) == 830
    assert not feat[["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]].isna().any().any()


def test_feature_contract_matches_saved_artifact(feature_frames):
    """The persisted feature list must equal the order the pipeline produces,
    otherwise inference would feed the model columns in the wrong order."""
    feat, _, _ = feature_frames
    saved = list(joblib.load(os.path.join(MODELS_DIR, "feature_list.joblib")))
    assert saved == features.FEATURE_COLUMNS
    assert len(saved) == 12
    assert all(c in feat.columns for c in saved)


def test_lags_are_computed_per_region(raw_df, feature_frames):
    """A row must never see another region's values."""
    feat, _, _ = feature_frames
    ordered = raw_df.sort_values(["date", "region"]).reset_index(drop=True)
    for region in ["Central", "East"]:
        g = ordered[ordered["region"] == region].reset_index(drop=True)
        f = feat[feat["region"] == region].reset_index(drop=True)
        # first modelled day is raw day 15 (index 14); its lag_1 is day 14
        assert f.loc[0, "lag_1"] == pytest.approx(g.loc[13, "consumption_liters"])
        assert f.loc[0, "lag_7"] == pytest.approx(g.loc[7, "consumption_liters"])


def test_rolling_mean_excludes_the_current_day(raw_df, feature_frames):
    """shift(1) before rolling: rolling_mean_7 must be the mean of the SEVEN
    days strictly before the row, never including its own target."""
    feat, _, _ = feature_frames
    assert feat["rolling_mean_7"].notna().all()

    ordered = raw_df.sort_values(["date", "region"]).reset_index(drop=True)
    g = ordered[ordered["region"] == "Central"].reset_index(drop=True)
    f = feat[feat["region"] == "Central"].reset_index(drop=True)
    # the feature frame drops the 14-day warm-up, so it is 14 rows shorter
    assert len(g) - 14 == len(f)

    # row i of the feature frame corresponds to raw day i+14 of that region.
    # Correct (past-only) mean uses raw days i+7 .. i+13.
    past_only = g.loc[0 + 7:0 + 13, "consumption_liters"].mean()
    assert f.loc[0, "rolling_mean_7"] == pytest.approx(past_only)

    # ...and must NOT equal a window that includes the current day (i+14).
    including_current = g.loc[1 + 7:1 + 14, "consumption_liters"].mean()
    assert f.loc[0, "rolling_mean_7"] != pytest.approx(including_current)


# -------------------------------------------------------------------- split
def test_split_is_chronological_with_no_overlap(feature_frames):
    _, train, test = feature_frames
    assert len(train) == 660 and len(test) == 170
    assert train["date"].max() < test["date"].min()
    assert str(train["date"].min().date()) == "2023-01-15"
    assert str(train["date"].max().date()) == "2023-05-26"
    assert str(test["date"].min().date()) == "2023-05-27"
    assert str(test["date"].max().date()) == "2023-06-29"


def test_split_matches_stored_metrics(feature_frames):
    _, train, test = feature_frames
    stored = json.load(open(os.path.join(MODELS_DIR, "metrics.json")))
    assert stored["train_rows"] == len(train)
    assert stored["test_rows"] == len(test)
    assert stored["train_date_range"] == [str(train["date"].min().date()), str(train["date"].max().date())]
    assert stored["test_date_range"] == [str(test["date"].min().date()), str(test["date"].max().date())]


# ------------------------------------------------------------------ metrics
def test_regression_metrics_reproduce_from_saved_model(feature_frames):
    """The deployed model must reproduce the metrics stored in metrics.json."""
    _, _, test = feature_frames
    stored = json.load(open(os.path.join(MODELS_DIR, "metrics.json")))
    cols = list(joblib.load(os.path.join(MODELS_DIR, "feature_list.joblib")))
    X, y = test[cols], test[TARGET_COL]

    for name, fname in [("RandomForest", "random_forest_model.joblib"),
                        ("GradientBoosting", "gradient_boosting_model.joblib")]:
        model = joblib.load(os.path.join(MODELS_DIR, fname))
        pred = model.predict(X)
        ref = stored["results"][name]
        assert mean_absolute_error(y, pred) == pytest.approx(ref["MAE"], abs=1e-4)
        assert r2_score(y, pred) == pytest.approx(ref["R2"], abs=1e-4)


def test_classification_thresholds_and_metrics_reproduce(feature_frames):
    _, train, test = feature_frames
    stored = json.load(open(os.path.join(MODELS_DIR, "classification_metrics.json")))
    thr = train[TARGET_COL].quantile([0.25, 0.50, 0.75]).to_numpy()
    for key, val in zip(["q25", "q50", "q75"], thr):
        assert val == pytest.approx(stored["thresholds"]["values"][key], abs=1e-4)

    q25, q50, q75 = thr
    y = np.select([test[TARGET_COL] < q25, test[TARGET_COL] < q50, test[TARGET_COL] < q75],
                  [CLASSES[0], CLASSES[1], CLASSES[2]], default=CLASSES[3])
    cols = list(joblib.load(os.path.join(MODELS_DIR, "classification_feature_list.joblib")))
    clf = joblib.load(os.path.join(MODELS_DIR, "classification_model.joblib"))
    pred = clf.predict(test[cols])
    from sklearn.metrics import accuracy_score
    ref = stored["results"][stored["best_classifier"]]
    assert accuracy_score(y, pred) == pytest.approx(ref["accuracy"], abs=1e-4)


def test_best_model_selection_is_by_r2(feature_frames):
    stored = json.load(open(os.path.join(MODELS_DIR, "metrics.json")))
    learned = {k: v for k, v in stored["results"].items() if k != "Baseline"}
    assert stored["best_model"] == max(learned, key=lambda k: learned[k]["R2"])


# ----------------------------------------------------- signal-audit ceiling
def test_signal_audit_artifact_is_consistent_with_the_data(raw_df):
    """The white-noise finding and the ceiling must still describe the data."""
    path = os.path.join(PROJECT_ROOT, "analysis", "signal_audit.json")
    if not os.path.exists(path):
        pytest.skip("run `python analysis/signal_audit.py` to generate the audit")
    audit = json.load(open(path))

    assert audit["white_noise_summary"]["conclusion"] == "cannot reject white noise"

    train = raw_df[raw_df["date"] <= "2023-05-26"]
    test = raw_df[raw_df["date"] >= "2023-05-27"]
    const = np.full(len(test), train[TARGET_COL].mean())
    assert mean_absolute_error(test[TARGET_COL], const) == pytest.approx(
        audit["ceiling"]["regression"]["ceiling_MAE"], abs=1e-4)
    assert audit["ceiling"]["regression"]["gap_vs_ceiling_percent"] < 5.0


# --------------------------------------------------- forecast determinism
def test_forecast_does_not_contaminate_history(predictor):
    """Regression test for the buffer leak: a forecast must not change what a
    later single-day prediction returns."""
    before = predictor.predict(region="Central", date="2026-10-13")
    predictor.forecast(region="Central", days=7)
    after = predictor.predict(region="Central", date="2026-10-13")
    assert before == after


def test_forecast_is_seven_days_and_ordered(predictor):
    rows = predictor.forecast(region="West", days=7)
    assert len(rows) == 7
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)
    assert all(3_000 < r["predicted_consumption"] < 25_000 for r in rows)
    assert all(r["demand_level"] in CLASSES for r in rows)


def test_lag_features_use_actual_history_after_forecast(predictor):
    """lag_1 must always come from the dataset, never from a forecast."""
    history = predictor.history
    last_actual = float(
        history[history["region"] == "Central"].sort_values("date")["consumption_liters"].iloc[-1])
    predictor.forecast(region="Central", days=7)
    row = predictor.build_feature_row(region="Central", date="2026-10-13")
    assert float(row["lag_1"].iloc[0]) == pytest.approx(last_actual)
