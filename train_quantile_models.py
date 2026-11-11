"""
train_quantile_models.py
========================
Trains lower (alpha=0.10) and upper (alpha=0.90) GradientBoostingRegressor quantile
models with split-conformal calibration (CQR) over the 73 engineered C-MAPSS features,
and exports sklearn-version-independent pure-NumPy portable bundles (.npz) for both
point/baseline models and quantile bounds so Streamlit Cloud never fails unpickling.
"""

import os
import json
import warnings
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

from train_improved_model import load_all_datasets, engineer_unit

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATS_PATH = os.path.join(BASE_DIR, "improved_model_features.json")
SCALER_PATH = os.path.join(BASE_DIR, "improved_rul_scaler.pkl")
POINT_MODEL_PATH = os.path.join(BASE_DIR, "improved_rul_model.pkl")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "gradient_boosting_model.pkl")
BASE_SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

OUT_Q10 = os.path.join(BASE_DIR, "improved_rul_q10.pkl")
OUT_Q90 = os.path.join(BASE_DIR, "improved_rul_q90.pkl")
OUT_META = os.path.join(BASE_DIR, "improved_rul_quantiles_meta.json")

OUT_BASE_PORTABLE = os.path.join(BASE_DIR, "baseline_rul_portable.npz")
OUT_IMP_PORTABLE = os.path.join(BASE_DIR, "improved_rul_portable.npz")
OUT_Q10_PORTABLE = os.path.join(BASE_DIR, "improved_rul_q10_portable.npz")
OUT_Q90_PORTABLE = os.path.join(BASE_DIR, "improved_rul_q90_portable.npz")


def export_gbr_to_portable_npz(model, scaler, out_path: str):
    """Exports a GradientBoostingRegressor + StandardScaler into a pure-NumPy .npz bundle."""
    trees = [est.tree_ for est in np.asarray(model.estimators_).ravel()]
    offsets = [0]
    cl_list, cr_list, feat_list, thresh_list, val_list = [], [], [], [], []
    for tr in trees:
        cl_list.append(np.asarray(tr.children_left, dtype=np.int32))
        cr_list.append(np.asarray(tr.children_right, dtype=np.int32))
        feat_list.append(np.asarray(tr.feature, dtype=np.int32))
        thresh_list.append(np.asarray(tr.threshold, dtype=np.float64))
        val_list.append(np.asarray(tr.value[:, 0, 0], dtype=np.float64))
        offsets.append(offsets[-1] + tr.node_count)

    n_feats = len(scaler.mean_)
    init_const = float(model.init_.predict(np.zeros((1, n_feats)))[0])
    np.savez_compressed(
        out_path,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        init_const=np.array([init_const], dtype=np.float64),
        learning_rate=np.array([float(model.learning_rate)], dtype=np.float64),
        tree_offsets=np.asarray(offsets, dtype=np.int32),
        children_left=np.concatenate(cl_list),
        children_right=np.concatenate(cr_list),
        feature=np.concatenate(feat_list),
        threshold=np.concatenate(thresh_list),
        value=np.concatenate(val_list),
    )
    return round(os.path.getsize(out_path) / 1024.0, 1)


def main():
    print("=" * 65)
    print("AeroGuard — Quantile Uncertainty + Portable Bundle Exporter")
    print("=" * 65)

    with open(FEATS_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    point_model = joblib.load(POINT_MODEL_PATH)
    base_model = joblib.load(BASE_MODEL_PATH)
    base_scaler = joblib.load(BASE_SCALER_PATH)

    # Export point & baseline models to portable pure-NumPy bundles
    base_kb = export_gbr_to_portable_npz(base_model, base_scaler, OUT_BASE_PORTABLE)
    imp_kb = export_gbr_to_portable_npz(point_model, scaler, OUT_IMP_PORTABLE)
    print(f"Exported portable bundles: baseline ({base_kb} KB), improved ({imp_kb} KB)")

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
        n_estimators=75,
        max_depth=4,
        learning_rate=0.09,
        subsample=0.35,
        random_state=42,
    )
    q10_model = GradientBoostingRegressor(alpha=0.10, **params)
    q90_model = GradientBoostingRegressor(alpha=0.90, **params)

    print("\nTraining GBR Q10 (lower) & Q90 (upper) quantile regressors...")
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
    q10_port_kb = export_gbr_to_portable_npz(q10_model, scaler, OUT_Q10_PORTABLE)
    q90_port_kb = export_gbr_to_portable_npz(q90_model, scaler, OUT_Q90_PORTABLE)

    meta = {
        "target_coverage": 0.80,
        "alpha_low": 0.10,
        "alpha_high": 0.90,
        "cqr_margin": round(cqr_margin, 4),
        "empirical_test_coverage": round(emp_cov, 4),
        "mean_interval_width_cycles": round(mean_width, 2),
        "pinball_loss_q10": round(pb10, 3),
        "pinball_loss_q90": round(pb90, 3),
        "q10_portable_kb": q10_port_kb,
        "q90_portable_kb": q90_port_kb,
        "improved_portable_kb": imp_kb,
        "baseline_portable_kb": base_kb,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"CQR Margin: +{cqr_margin:.2f} cyc | Test Coverage: {emp_cov*100:.1f}% | Mean Width: {mean_width:.1f} cyc")
    print(f"Portable bundles saved: Q10 ({q10_port_kb} KB), Q90 ({q90_port_kb} KB)")


if __name__ == "__main__":
    main()
