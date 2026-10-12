"""
train_classifier.py
====================
Trains and evaluates demand-level CLASSIFIERS on the REAL Kaggle dataset
(region, date, consumption_liters).

Target:  demand_level  in {LOW, NORMAL, HIGH, VERY HIGH}

Methodology (defensible in a viva):
  - Reuses the EXACT same leakage-safe feature pipeline as the regression
    models (src/features.py): region one-hot + calendar + per-region lags.
  - The four demand_level classes are derived from consumption QUANTILES
    (25th / 50th / 75th percentiles) computed on the TRAINING SET ONLY.
    The test set is never used to derive thresholds (no label leakage).
  - The same chronological 80/20 split as the regression task is used, so
    classification is evaluated on the exact same held-out future period.
  - Compares Logistic Regression vs Random Forest Classifier with:
      Accuracy, Precision, Recall, F1 (per class and macro), weighted F1,
      and the full 4x4 confusion matrix.
  - The best classifier is selected by weighted F1 on the test set.

Saves into /models:
    classification_model.joblib        best classifier
    classification_feature_list.joblib feature columns (same as regression)
    classification_thresholds.json     quantile thresholds (train-only fit)
    classification_metrics.json        full evaluation results

Run:
    python train_classifier.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import src.features as features
from src.features import build_features, chronological_split, TARGET_COL, REGION_COL, DATE_COL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "processed", "water_demand_clean.csv")
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASSES = ["LOW", "NORMAL", "HIGH", "VERY HIGH"]
QUANTILES = [0.25, 0.50, 0.75]


def demand_levels(values, thresholds):
    """Map continuous consumption to demand_level using quantile thresholds.

    thresholds = [q25, q50, q75]  ->  <q25 LOW, <q50 NORMAL, <q75 HIGH, else VERY HIGH
    """
    q25, q50, q75 = thresholds
    return np.select(
        [values < q25, values < q50, values < q75],
        [CLASSES[0], CLASSES[1], CLASSES[2]],
        default=CLASSES[3],
    )


def evaluate_classifier(y_true, y_pred):
    per_class = {}
    p, r, f = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0), \
              recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0), \
              f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
        }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def main():
    print(f"Loading REAL cleaned dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print("\nBuilding features (same leakage-safe pipeline as regression) ...")
    feat_df = build_features(df, for_training=True)
    FEATURE_COLUMNS = features.FEATURE_COLUMNS
    print(f"  feature dataframe: {feat_df.shape[0]} rows, {len(FEATURE_COLUMNS)} features")

    train_df, test_df = chronological_split(feat_df, test_size=0.2)
    print(f"\nChronological 80/20 split -> train: {len(train_df)} rows, test: {len(test_df)} rows")
    print(f"  train period: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}")
    print(f"  test period : {test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()}")

    # ------------------------------------------------------------------
    # Derive demand_level thresholds from the TRAINING SET ONLY
    # (test data plays no role in threshold fitting - no label leakage).
    # ------------------------------------------------------------------
    thresholds = train_df[TARGET_COL].quantile(QUANTILES).values
    print(f"\nTraining-set consumption quantiles -> thresholds: "
          f"q25={thresholds[0]:.2f}, q50={thresholds[1]:.2f}, q75={thresholds[2]:.2f}")
    print("  class rule: <q25 LOW | <q50 NORMAL | <q75 HIGH | >=q75 VERY HIGH")

    y_train = demand_levels(train_df[TARGET_COL].values, thresholds)
    y_test = demand_levels(test_df[TARGET_COL].values, thresholds)

    print("\nClass distribution (train):", {c: int((y_train == c).sum()) for c in CLASSES})
    print("Class distribution (test) :", {c: int((y_test == c).sum()) for c in CLASSES})

    X_train, X_test = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]

    results = {}

    # ------------------------------------------------------------------
    # 1. LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    print("\n[1/2] Training Logistic Regression (standardized features) ...")
    # StandardScaler improves convergence given mixed scales (one-hot 0/1,
    # month 1-12, lags in the thousands). Scaler is fit on TRAIN only.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["LogisticRegression"] = evaluate_classifier(y_test, lr_pred)
    print(f"  Logistic Regression -> acc={results['LogisticRegression']['accuracy']}, "
          f"f1_macro={results['LogisticRegression']['f1_macro']}, f1_weighted={results['LogisticRegression']['f1_weighted']}")

    # ------------------------------------------------------------------
    # 2. RANDOM FOREST CLASSIFIER
    # ------------------------------------------------------------------
    print("\n[2/2] Training Random Forest Classifier ...")
    rfc = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    results["RandomForest"] = evaluate_classifier(y_test, rfc_pred)
    print(f"  Random Forest Classifier -> acc={results['RandomForest']['accuracy']}, "
          f"f1_macro={results['RandomForest']['f1_macro']}, f1_weighted={results['RandomForest']['f1_weighted']}")

    # ------------------------------------------------------------------
    # Select the best classifier by weighted F1 on the test set
    # ------------------------------------------------------------------
    classifiers = {"LogisticRegression": lr, "RandomForest": rfc}
    best_name = max(classifiers.keys(), key=lambda k: results[k]["f1_weighted"])
    print(f"\nBest classifier by weighted F1: {best_name} "
          f"(f1_weighted={results[best_name]['f1_weighted']})")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    joblib.dump(classifiers[best_name], os.path.join(MODELS_DIR, "classification_model.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "classification_feature_list.joblib"))

    thresholds_payload = {
        "method": "training-set consumption quantiles (fit on TRAIN ONLY)",
        "quantiles": QUANTILES,
        "values": {f"q{int(q * 100)}": round(float(t), 4) for q, t in zip(QUANTILES, thresholds)},
        "classes": CLASSES,
        "rule": ["< q25 -> LOW", "q25 <= x < q50 -> NORMAL", "q50 <= x < q75 -> HIGH", ">= q75 -> VERY HIGH"],
    }
    with open(os.path.join(MODELS_DIR, "classification_thresholds.json"), "w") as f:
        json.dump(thresholds_payload, f, indent=2)

    metrics_payload = {
        "classes": CLASSES,
        "thresholds": thresholds_payload,
        "results": results,
        "best_classifier": best_name,
        "selection_criterion": "weighted F1 on chronological test set",
        "target_column": f"demand_level (derived from {TARGET_COL})",
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": [str(train_df[DATE_COL].min().date()), str(train_df[DATE_COL].max().date())],
        "test_date_range": [str(test_df[DATE_COL].min().date()), str(test_df[DATE_COL].max().date())],
        "class_distribution_test": {c: int((y_test == c).sum()) for c in CLASSES},
    }
    with open(os.path.join(MODELS_DIR, "classification_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved classifier + thresholds + metrics + feature list -> {MODELS_DIR}")
    print("\nConfusion matrix (rows=true, cols=predicted, order: LOW, NORMAL, HIGH, VERY HIGH):")
    print(np.array(results[best_name]["confusion_matrix"]))
    print("Done.")


if __name__ == "__main__":
    main()