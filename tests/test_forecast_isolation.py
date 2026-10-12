"""
test_forecast_isolation.py
==========================
Targeted regression tests for the forecast-history contamination defect and
the full verification matrix it requires:

  1. A multi-day forecast must not mutate the shared historical buffer.
  2. Single-day predictions must use real observed lag values.
  3. Repeated identical requests must return identical results.
  4. Results must be independent of the order requests are made in.
  5. Recursive multi-step forecast semantics must be preserved.
  6. Regions must be isolated: one region's forecast cannot move another's
     predictions.
  7. Training and inference must produce the same feature schema.
  8. Invalid requests must raise/return errors without corrupting state.

The committed (HEAD) implementation restored the shared buffer only in
`finally`, so a request interleaved *during* a forecast still read forecasted
values as lag_1. test_interleaved_prediction_never_reads_forecast_values
reproduces exactly that failure mode deterministically (no threads, no
sleeps): it re-enters predict() from inside each model call, which a
shared-buffer implementation cannot survive.
"""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import TARGET_COL, DATE_COL, REGION_COL, build_features
from src.predict import DemandPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "water_demand_clean.csv")
ALL_REGIONS = ["Central", "East", "North", "South", "West"]


def snapshot_buffers(predictor):
    return {r: list(v) for r, v in predictor._buffer.items()}


# ------------------------------------------------- 1. buffer non-mutation
def test_forecast_never_mutates_shared_buffers(predictor):
    """Stronger than output equality: the buffer itself must be identical
    before and after recursive forecasts, and hold real observations only."""
    before = snapshot_buffers(predictor)
    for region in predictor.regions:
        predictor.forecast(region=region, days=7)
    after = snapshot_buffers(predictor)
    assert before == after
    for region, values in after.items():
        actual = set(
            predictor.history[predictor.history[REGION_COL] == region][TARGET_COL]
        )
        assert all(v in actual for v in values)


# ------------------------------------------------------- 3. determinism
def test_repeated_requests_are_identical(predictor):
    preds = [predictor.predict("North", "2026-10-13") for _ in range(5)]
    assert len(set(preds)) == 1
    forecasts = [predictor.forecast("North", days=7) for _ in range(3)]
    assert all(f == forecasts[0] for f in forecasts)


# ---------------------------------------------- 4. request-order freedom
def test_prediction_is_independent_of_request_order(predictor):
    reference = {r: predictor.predict(r, "2026-10-13") for r in predictor.regions}
    for region in predictor.regions:
        predictor.forecast(region=region, days=7)
        predictor.predict(region, "2026-11-01")
    predictor.forecast("Central", days=14)
    predictor.predict("East", "2027-01-01")
    predictor.forecast("West", days=3)
    assert {r: predictor.predict(r, "2026-10-13") for r in predictor.regions} == reference


def test_api_order_permutations(client):
    """End-to-end order independence across every region, not just Central."""
    reference = {
        r: client.post("/api/predict", json={"region": r, "date": "2026-10-13"}).get_json()
        for r in ALL_REGIONS
    }
    for region in ["West", "North", "Central", "South", "East"]:
        client.get("/api/forecast?region=" + region)
        client.post("/api/predict", json={"region": region, "date": "2026-10-20"})
    for region in ALL_REGIONS:
        got = client.post(
            "/api/predict", json={"region": region, "date": "2026-10-13"}
        ).get_json()
        assert got["predicted_consumption"] == reference[region]["predicted_consumption"]
        assert got["demand_level"] == reference[region]["demand_level"]


# ------------------------------------ 5. recursive semantics + isolation
def test_interleaved_prediction_never_reads_forecast_values(predictor):
    """Deterministic mid-forecast interleaving probe.

    Re-entering predict() from inside the model call simulates a concurrent
    /api/predict request arriving while /api/forecast is running. The nested
    read must see the shared REAL history only. The committed implementation
    failed exactly here: forecast() appended predicted values to the shared
    buffer inside its loop and restored them only in `finally`, so any request
    interleaved mid-forecast read a predicted value as lag_1.
    """
    region = "Central"
    truth = predictor.predict(region, "2026-10-13")
    original_predict = predictor.model.predict
    nested = []
    # `armed` keeps the probe active for the forecast's own model calls; each
    # such call injects one independent single-day prediction (the "concurrent
    # request"), which must read the shared real history. `in_probe` only
    # guards the nested call itself from re-arming.
    state = {"armed": False, "in_probe": False}

    def spy(X):
        if state["armed"] and not state["in_probe"]:
            state["in_probe"] = True
            try:
                nested.append(predictor.predict(region, "2026-10-13"))
            finally:
                state["in_probe"] = False
        return original_predict(X)

    predictor.model.predict = spy
    state["armed"] = True
    try:
        predictor.forecast(region=region, days=7)
    finally:
        state["armed"] = False
        predictor.model.predict = original_predict

    assert nested, "the probe should have fired during the forecast"
    assert all(v == truth for v in nested), (
        f"interleaved predictions were contaminated by forecast values: {nested} != {truth}"
    )


def test_forecast_recursion_feeds_predictions_forward(predictor):
    """Recursive semantics must SURVIVE the isolation fix: day N+1's lag_1
    is day N's *predicted* value inside the same forecast."""
    region = "East"
    rows = predictor.forecast(region=region, days=5)

    working = {r: list(v) for r, v in predictor._buffer.items()}
    manual = []
    day = pd.to_datetime(rows[0]["date"])
    for _ in range(5):
        day_str = day.strftime("%Y-%m-%d")
        pred = predictor.predict(region, day_str, buffer=working)
        manual.append({"date": day_str, "predicted_consumption": round(pred, 2)})
        working[region].append(pred)
        day = day + pd.Timedelta(days=1)
    manual_rows = [{"date": m["date"], "predicted_consumption": m["predicted_consumption"]}
                   for m in manual]
    assert manual_rows == [{"date": r["date"], "predicted_consumption": r["predicted_consumption"]}
                           for r in rows]

    working2 = {r: list(v) for r, v in predictor._buffer.items()}
    day1 = predictor.predict(region, rows[0]["date"], buffer=working2)
    working2[region].append(day1)
    day2 = pd.to_datetime(rows[0]["date"]) + pd.Timedelta(days=1)
    row2 = predictor.build_feature_row(region, day2.strftime("%Y-%m-%d"), buffer=working2)
    assert float(row2["lag_1"].iloc[0]) == pytest.approx(day1)


# ------------------------------------------------------ 6. region isolation
def test_forecasts_are_isolated_across_regions(predictor):
    others = {r: predictor.predict(r, "2026-10-13") for r in predictor.regions}
    lengths = {r: len(v) for r, v in predictor._buffer.items()}
    predictor.forecast("Central", days=30)  # long forecast, lots of feedback
    assert {r: predictor.predict(r, "2026-10-13") for r in predictor.regions} == others
    assert {r: len(v) for r, v in predictor._buffer.items()} == lengths


# ---------------------------------------------- 7. train/serve schema match
def test_train_and_inference_feature_schema_is_identical():
    saved = list(joblib.load(os.path.join(MODELS_DIR, "feature_list.joblib")))
    clean = pd.read_csv(DATA_PATH)
    feat = build_features(clean, for_training=True)
    train_cols = [c for c in feat.columns if c not in (DATE_COL, REGION_COL, TARGET_COL)]
    assert train_cols == saved  # same columns, same order

    predictor = DemandPredictor(model_name="random_forest_model.joblib")
    row = predictor.build_feature_row("South", "2026-10-13")
    assert list(row.columns) == saved
    assert row.notna().all().all()
    assert all(np.issubdtype(dtype, np.number) for dtype in row.dtypes)


# ---------------------------------------------------- 8. failure cleanliness
def test_invalid_requests_do_not_corrupt_state(predictor):
    before = snapshot_buffers(predictor)
    truth = {r: predictor.predict(r, "2026-10-13") for r in predictor.regions}

    with pytest.raises(ValueError):
        predictor.forecast(region="Atlantis", days=7)
    with pytest.raises(ValueError):
        predictor.predict(region="Atlantis", date="2026-10-13")
    with pytest.raises(ValueError):
        predictor.build_feature_row("Atlantis", "2026-10-13")
    with pytest.raises((ValueError, TypeError)):
        predictor.predict("Central", "not-a-date")

    assert snapshot_buffers(predictor) == before
    assert {r: predictor.predict(r, "2026-10-13") for r in predictor.regions} == truth
