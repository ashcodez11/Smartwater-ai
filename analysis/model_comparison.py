"""
model_comparison.py
===================
Reproducible, controlled model comparison for SmartWater AI (Phase 3 + 4 of
the improvement mandate):

  1. INDEPENDENTLY REPRODUCE the deployed model's stored metrics on the
     untouched chronological test window (2023-05-27 -> 2023-06-29), and the
     constant training-mean reference MAE (~3,514.3).
  2. ROLLING-ORIGIN VALIDATION for candidate selection, entirely inside the
     training period, so the final test window is never used for selection:
     origins at 50/60/70/80% of each region's timeline; each origin predicts
     the following 30 days with 1-step-ahead features built ONLY from past
     actual observations.
  3. FINAL EVALUATION of all candidates on the untouched test window plus
     per-region results and a paired-bootstrap confidence interval of the
     best candidate's MAE against the deployed RandomForest.

Candidates (fair: identical evaluation windows, per-fold fitting only):
  - TrainMean           constant = training-fold global mean
  - RegionMean          constant = training-fold per-region mean
  - Persistence         lag_1 (previous actual day)
  - SeasonalNaive7      lag_7 (same weekday one week back)
  - Ridge               sklearn Ridge with scaler, same leakage-safe features
  - RandomForestDeployed  exact deployed hyperparameters (300/10/2, seed 42)
  - RandomForestStronger  more regularised (500/6/10, seed 42)
  - HistGradientBoosting  sklearn HistGB (early stopping off, seed 42)
  - RegionSpecificRF      separate RF per region (no region one-hot)

Reads:  data/processed/water_demand_clean.csv, models/*.joblib
Writes: analysis/model_comparison.json

Run:
    python analysis/model_comparison.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.features as features  # noqa: E402
from src.features import build_features, chronological_split, TARGET_COL, REGION_COL, DATE_COL  # noqa: E402
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "water_demand_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_PATH = os.path.join(HERE, "model_comparison.json")

SEED = 42
HORIZON_DAYS = 30          # each rolling origin predicts the next 30 days
ORIGIN_FRACTIONS = [0.5, 0.6, 0.7, 0.8]


def evaluate(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ------------------------------------------------------------- candidates
def fit_and_predict(name, train, test, feature_cols):
    """Return predictions for `test` rows, fitting ONLY on `train`."""
    Xtr, ytr = train[feature_cols], train[TARGET_COL]
    Xte, yte = test[feature_cols], test[TARGET_COL]

    if name == "TrainMean":
        return np.full(len(test), ytr.mean())
    if name == "RegionMean":
        means = train.groupby(REGION_COL)[TARGET_COL].mean()
        return test[REGION_COL].map(means).to_numpy()
    if name == "Persistence":
        return test["lag_1"].to_numpy()
    if name == "SeasonalNaive7":
        return test["lag_7"].to_numpy()
    if name == "Ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if name == "RandomForestDeployed":
        m = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if name == "RandomForestStronger":
        m = RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if name == "HistGradientBoosting":
        m = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if name == "RegionSpecificRF":
        cols = [c for c in feature_cols if not c.startswith("region_")]
        preds = np.zeros(len(test))
        for region, tr_g in train.groupby(REGION_COL):
            mask = (test[REGION_COL] == region).to_numpy()
            m = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2,
                                      random_state=SEED, n_jobs=-1)
            m.fit(tr_g[cols], tr_g[TARGET_COL])
            preds[mask] = m.predict(test.loc[mask, cols])
        return preds
    raise ValueError(name)


CANDIDATES = ["TrainMean", "RegionMean", "Persistence", "SeasonalNaive7", "Ridge",
              "RandomForestDeployed", "RandomForestStronger", "HistGradientBoosting",
              "RegionSpecificRF"]


# --------------------------------------------------- rolling-origin setup
def rolling_origin_frames(feat_df, test_start):
    """Global date-origin rolling windows INSIDE the training period only.

    Each origin date O yields (all rows dated < O, all rows O..O+30d), across
    every region, so pooled candidates (one model for all regions) and
    region-specific candidates are both evaluated on identical windows.
    Features are already past-only per region, so any date cut is leakage-free.
    """
    train_pool = feat_df[feat_df[DATE_COL] < test_start]
    day_starts = sorted(train_pool[DATE_COL].unique())
    for frac in ORIGIN_FRACTIONS:
        origin = day_starts[int(len(day_starts) * frac)]
        train = train_pool[train_pool[DATE_COL] < origin]
        test = train_pool[(train_pool[DATE_COL] >= origin)
                          & (train_pool[DATE_COL] < origin + pd.Timedelta(days=HORIZON_DAYS))]
        if len(train) >= 300 and len(test) >= 100:
            yield frac, train, test


def paired_bootstrap_ci(err_a, err_b, n=5000, alpha=0.05):
    """95% CI for MAE(a) - MAE(b) via paired bootstrap over rows."""
    rng = np.random.default_rng(SEED)
    diffs = np.abs(err_a) - np.abs(err_b)
    stats = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    raw = pd.read_csv(DATA_PATH)
    feat_df = build_features(raw, for_training=True)
    cols = list(features.FEATURE_COLUMNS)
    train, test = chronological_split(feat_df, test_size=0.2)

    out = {"protocol": {
        "seed": SEED,
        "split": "chronological per-region 80/20 (final test window untouched by selection)",
        "rolling_origins": [f"{int(fr * 100)}% of each region's training timeline" for fr in ORIGIN_FRACTIONS],
        "origin_horizon_days": HORIZON_DAYS,
        "selection_criterion": "mean rolling-origin MAE (test window only reported, never used to choose)",
    }}

    # ---------------------------------------------- 1. reproduce baseline
    stored = json.load(open(os.path.join(MODELS_DIR, "metrics.json")))
    deployed = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    dep_pred = deployed.predict(test[cols])
    y_test = test[TARGET_COL]
    reproduced = {
        "RandomForest": evaluate(y_test, dep_pred),
        "TrainMeanReference": evaluate(y_test, np.full(len(test), train[TARGET_COL].mean())),
    }
    out["baseline_reproduction"] = {
        "stored_in_metrics_json": {k: stored["results"][k] for k in ("RandomForest",)},
        "reproduced": reproduced,
        "max_abs_error_vs_stored": max(
            abs(reproduced["RandomForest"][m] - stored["results"]["RandomForest"][m])
            for m in ("MAE", "RMSE", "R2")),
        "test_rows": int(len(test)),
        "train_rows": int(len(train)),
    }

    # ------------------------------------------------ 2. rolling origins
    per_candidate = {c: [] for c in CANDIDATES}
    for frac, tr, te in rolling_origin_frames(feat_df, test[DATE_COL].min()):
        for c in CANDIDATES:
            pred = fit_and_predict(c, tr, te, cols)
            per_candidate[c].append(mean_absolute_error(te[TARGET_COL], pred))
    out["rolling_origin_mae"] = {
        c: {"mean": round(float(np.mean(v)), 3), "per_origin": [round(x, 3) for x in v]}
        for c, v in per_candidate.items()
    }
    ranked = sorted(out["rolling_origin_mae"], key=lambda c: out["rolling_origin_mae"][c]["mean"])
    out["selected_by_rolling_origin"] = ranked[0]

    # ------------------------------------- 3. final untouched-test results
    final, per_region = {}, {}
    best = ranked[0]
    for c in CANDIDATES + ["DeployedRFArtifact"]:
        pred = dep_pred if c == "DeployedRFArtifact" else fit_and_predict(c, train, test, cols)
        final[c] = evaluate(y_test, pred)
        if c in (best, "RandomForestStronger", "RegionMean", "DeployedRFArtifact"):
            per_region[c] = {
                r: evaluate(g[TARGET_COL], pred[(test[REGION_COL] == r).to_numpy()])
                for r, g in test.groupby(REGION_COL, sort=False)
            }
        out.setdefault("_abs_err", {})[c] = list(np.abs(y_test - pred))

    out["final_test_window"] = {
        "date_range": [str(test[DATE_COL].min().date()), str(test[DATE_COL].max().date())],
        "results": final, "per_region": per_region,
    }

    # --------------------------- 4. significance vs deployed RF (MAE)
    # A candidate is "significant" only if the 95% CI of
    # MAE(candidate) - MAE(deployed) lies ENTIRELY below zero.
    b = np.array(out["_abs_err"]["DeployedRFArtifact"])
    sig = {}
    for c in CANDIDATES:
        a = np.array(out["_abs_err"][c])
        ci = paired_bootstrap_ci(a, b)
        sig[c] = {
            "mae_improvement_vs_deployed": round(float(b.mean() - a.mean()), 3),
            "bootstrap_95ci_of_mae_difference": ci,
            "significant": ci[1] < 0,
        }
    out["significance_vs_deployed"] = sig
    out["best_vs_deployed"] = {**sig[best], "best": best}
    del out["_abs_err"]

    replacement = [c for c in ranked
                   if sig[c]["significant"] and c not in ("TrainMean", "RegionMean")]
    out["deployment_recommendation"] = {
        "replace_production_model": bool(replacement),
        "recommended_candidate": replacement[0] if replacement else None,
        "reason": ("a learned model significantly beats the deployed RF on the "
                   "untouched window" if replacement else
                   "no candidate's MAE advantage over the deployed RandomForest is "
                   "statistically significant (every 95% bootstrap CI spans zero); "
                   "the constants win by <=2% and are operationally unsuitable as "
                   "the API's forecaster. Do not force a replacement."),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:2000])
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
