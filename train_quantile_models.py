"""
train_quantile_models.py
========================
Trains lower (alpha=0.10) and upper (alpha=0.90) quantile regression models
with split-conformal calibration (CQR) over the 73 engineered C-MAPSS features
to provide calibrated 80% RUL prediction uncertainty intervals [Q10, Q90].
"""

import os
import json
import warnings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

from train_improved_model import load_all_datasets, engineer_unit

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATS_PATH = os.path.join(BASE_DIR, "improved_model_features.json")
SCALER_PATH = os.path.join(BASE_DIR, "improved_rul_scaler.pkl")
POINT_MODEL_PATH = os.path.join(BASE_DIR, "improved_rul_model.pkl")

OUT_Q10 = os.path.join(BASE_DIR, "improved_rul_q10.pkl")
OUT_Q90 = os.path.join(BASE_DIR, "improved_rul_q90.pkl")
OUT_META = os.path.join(BASE_DIR, "improved_rul_quantiles_meta.json")


def main():
    print("=" * 65)
    print("AeroGuard — Quantile Uncertainty Bounds (CQR 80% Interval)")
    print("=" * 65)

    with open(FEATS_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    point_model = joblib.load(POINT_MODEL_PATH)

    df = load_all_datasets()
    df_feat = df.groupby("unit_key", group_keys=False).apply(engineer_unit)

    X = df_feat[feats].values.astype(float)
    y = df_feat["RUL"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_qtr, X_qcal, y_qtr, y_qcal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    X_qtr_sc = scaler.transform(X_qtr)
    X_qcal_sc = scaler.transform(X_qcal)
    X_test_sc = scaler.transform(X_test)

    params = dict(
        loss="quantile",
        max_iter=200,
        max_leaf_nodes=31,
        learning_rate=0.08,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    q10_model = HistGradientBoostingRegressor(quantile=0.10, **params)
    q90_model = HistGradientBoostingRegressor(quantile=0.90, **params)

    print("\nTraining Q10 (lower bound) & Q90 (upper bound) regressors...")
    q10_model.fit(X_qtr_sc, y_qtr)
    q90_model.fit(X_qtr_sc, y_qtr)

    # Split-Conformalized Quantile Regression (CQR) margin for finite-sample 80% coverage
    p10_cal = np.ravel(q10_model.predict(X_qcal_sc))
    p90_cal = np.ravel(q90_model.predict(X_qcal_sc))
    cal_scores = np.maximum(p10_cal - y_qcal, y_qcal - p90_cal)
    q_level = min(1.0, 0.80 * (1.0 + 1.0 / len(cal_scores)))
    cqr_margin = max(0.0, float(np.quantile(cal_scores, q_level)))

    # Evaluate cleanly on unseen test split
    pt_test = np.maximum(1.0, np.ravel(point_model.predict(X_test_sc)))
    raw_lo = np.ravel(q10_model.predict(X_test_sc)) - cqr_margin
    raw_hi = np.ravel(q90_model.predict(X_test_sc)) + cqr_margin

    lo_test = np.maximum(1.0, np.minimum(raw_lo, pt_test))
    hi_test = np.maximum(pt_test, np.maximum(raw_hi, lo_test + 2.0))

    emp_cov = float(np.mean((y_test >= lo_test) & (y_test <= hi_test)))
    mean_width = float(np.mean(hi_test - lo_test))
    pb10 = float(mean_pinball_loss(y_test, np.ravel(q10_model.predict(X_test_sc)), alpha=0.10))
    pb90 = float(mean_pinball_loss(y_test, np.ravel(q90_model.predict(X_test_sc)), alpha=0.90))

    joblib.dump(q10_model, OUT_Q10, compress=3)
    joblib.dump(q90_model, OUT_Q90, compress=3)

    meta = {
        "target_coverage": 0.80,
        "alpha_low": 0.10,
        "alpha_high": 0.90,
        "cqr_margin": round(cqr_margin, 4),
        "empirical_test_coverage": round(emp_cov, 4),
        "mean_interval_width_cycles": round(mean_width, 2),
        "pinball_loss_q10": round(pb10, 3),
        "pinball_loss_q90": round(pb90, 3),
        "q10_size_kb": round(os.path.getsize(OUT_Q10) / 1024.0, 1),
        "q90_size_kb": round(os.path.getsize(OUT_Q90) / 1024.0, 1),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"CQR Margin: +{cqr_margin:.2f} cyc | Test Coverage: {emp_cov*100:.1f}% | Mean Width: {mean_width:.1f} cyc")
    print(f"Saved: {OUT_Q10} ({meta['q10_size_kb']} KB), {OUT_Q90} ({meta['q90_size_kb']} KB), {OUT_META}")


if __name__ == "__main__":
    main()
