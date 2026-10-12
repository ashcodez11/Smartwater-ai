"""
predict.py
==========
Reusable prediction functions for SmartWater-AI (real Kaggle dataset).

Loads the trained model + feature list + recent per-region history once, and
exposes:

    DemandPredictor.predict(region, date)   -> single-day prediction
    DemandPredictor.forecast(region, days)  -> recursive multi-step forecast

Because lag_1 / lag_7 / rolling_mean_7 / rolling_mean_14 require recent
actual consumption values (which a brand-new future date does not have), the
predictor seeds its buffers from the last known readings per region
(models/recent_history.csv). For multi-day forecasts, each predicted value is
fed back into the buffer so later days use earlier predicted values (a
standard recursive multi-step forecast). A production version would maintain
a live rolling buffer of actual daily readings.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.features import TARGET_COL, DATE_COL, REGION_COL

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class DemandPredictor:
    def __init__(self, model_name="random_forest_model.joblib"):
        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.feature_list_path = os.path.join(MODELS_DIR, "feature_list.joblib")
        self.history_path = os.path.join(MODELS_DIR, "recent_history.csv")

        self.model = joblib.load(self.model_path)
        self.feature_columns = joblib.load(self.feature_list_path)
        self.history = pd.read_csv(self.history_path, parse_dates=[DATE_COL]).sort_values(DATE_COL)
        self.history = self.history.drop_duplicates(subset=[REGION_COL, DATE_COL], keep="last")

        # --- demand-level classifier (optional; loaded if present) ---
        self.classifier = None
        self.classifier_classes = []
        self.classifier_path = os.path.join(MODELS_DIR, "classification_model.joblib")
        if os.path.exists(self.classifier_path):
            self.classifier = joblib.load(self.classifier_path)
            self.classifier_classes = self.classifier.classes_.tolist()

        # Per-region rolling buffer of known consumption, used to derive
        # lag/rolling features for new dates. Starts from recent history and
        # grows with each prediction during multi-step forecasts.
        self._buffer = {region: list(g[TARGET_COL].values)
                        for region, g in self.history.groupby(REGION_COL, sort=False)}

    @property
    def regions(self):
        return sorted(self.history[REGION_COL].unique())

    def reset_buffer(self):
        self._buffer = {region: list(g[TARGET_COL].values)
                        for region, g in self.history.groupby(REGION_COL, sort=False)}

    def _known_values(self, region, buffer=None):
        """The list of known consumption values for a region.

        `buffer` overrides the internal history. A recursive forecast passes a
        private copy, so the shared state is never written to and concurrent
        requests cannot influence each other (see forecast()).
        """
        source = self._buffer if buffer is None else buffer
        if region not in source:
            raise ValueError(f"Unknown region '{region}'. Available: {self.regions}")
        return source[region]

    def _lag_features(self, region, buffer=None):
        """Derive lag_1, lag_7, rolling_mean_7, rolling_mean_14 for a region
        from its buffer of known (past-only) consumption values."""
        vals = self._known_values(region, buffer)
        if not vals:
            raise ValueError(f"No history available for region '{region}'")

        lag_1 = float(vals[-1])
        lag_7 = float(vals[-7]) if len(vals) >= 7 else lag_1
        rolling_mean_7 = float(np.mean(vals[-7:])) if len(vals) >= 7 else float(np.mean(vals))
        rolling_mean_14 = float(np.mean(vals[-14:])) if len(vals) >= 14 else float(np.mean(vals))
        return lag_1, lag_7, rolling_mean_7, rolling_mean_14

    def build_feature_row(self, region, date, buffer=None):
        """Feature vector for (region, date). Read-only: passing `buffer`
        evaluates the lags against a private history without touching the
        shared one."""
        date = pd.to_datetime(date)
        lag_1, lag_7, rolling_mean_7, rolling_mean_14 = self._lag_features(region, buffer)

        row = {
            "month": date.month,
            "day_of_week": date.dayofweek,
            "is_weekend": int(date.dayofweek >= 5),
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": rolling_mean_14,
        }
        # region one-hot columns (0/1) — must exactly match training columns
        for col in self.feature_columns:
            if col.startswith("region_"):
                row[col] = 1 if col == f"region_{region}" else 0
        return pd.DataFrame([row])[self.feature_columns]

    def predict(self, region, date, buffer=None):
        X = self.build_feature_row(region, date, buffer)
        pred = float(self.model.predict(X)[0])
        return round(pred, 3)

    def classify(self, region, date, buffer=None):
        """Predict the demand level (LOW/NORMAL/HIGH/VERY HIGH) for a date."""
        if self.classifier is None:
            raise RuntimeError("Classification model not loaded (train_classifier.py not run?)")
        X = self.build_feature_row(region, date, buffer)
        return str(self.classifier.predict(X)[0])

    def predict_with_level(self, region, date, buffer=None):
        """Return (predicted_consumption, demand_level) for a date."""
        consumption = self.predict(region, date, buffer)
        try:
            level = self.classify(region, date, buffer)
        except RuntimeError:
            level = None
        return consumption, level

    def forecast(self, region, start_date=None, days=7):
        """
        Recursive multi-step forecast: each predicted value feeds the lag
        buffer for the next day. Returns a list of {date, predicted_consumption}.

        The recursion uses a copy of the region's history, so the shared
        buffer is never left holding forecasted values. Without this, a
        single-day prediction asked after a forecast would treat a *predicted*
        day as if it were a real reading (lag_1 came from the forecast, not the
        dataset), and the same input would return different answers depending
        on whether a forecast had already run.
        """
        if region not in self._buffer:
            raise ValueError(f"Unknown region '{region}'. Available: {self.regions}")

        if start_date is None:
            last_date = self.history[self.history[REGION_COL] == region][DATE_COL].max()
            start_date = last_date + pd.Timedelta(days=1)
        else:
            start_date = pd.to_datetime(start_date)

        # A PRIVATE copy of the history is the only thing this method writes to.
        # Predicted days still feed forward inside this forecast (recursive
        # multi-step semantics), but the shared buffer keeps real readings only.
        # That makes the method a pure function of its arguments: two requests
        # — or two threads — can never affect each other's results.
        working = {r: list(v) for r, v in self._buffer.items()}

        results = []
        current = start_date
        for _ in range(days):
            pred = self.predict(region, current, buffer=working)
            row = {"date": current.strftime("%Y-%m-%d"), "predicted_consumption": round(pred, 2)}
            if self.classifier is not None:
                row["demand_level"] = self.classify(region, current, buffer=working)
            results.append(row)
            working[region].append(pred)  # feed prediction back for the next day
            current = current + pd.Timedelta(days=1)
        return results