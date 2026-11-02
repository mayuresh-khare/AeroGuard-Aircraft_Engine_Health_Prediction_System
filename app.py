import os
import glob
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


TEXT_DARK = "#10182c"


def styled_layout(**overrides):
    """Base Plotly layout; dynamically switches between light / dark theme.
    Pass any Plotly layout kwargs to override or extend the defaults."""
    _dm   = st.session_state.get("dark_mode", False)
    _text = "#e2e8f0" if _dm else TEXT_DARK
    _pbg  = "#1e293b" if _dm else "white"
    _grid = "#334155" if _dm else "#e5e9f2"
    layout = {
        "plot_bgcolor":  _pbg,
        "paper_bgcolor": _pbg,
        "font":   dict(color=_text),
        "xaxis":  dict(color=_text, gridcolor=_grid, title_font=dict(color=_text)),
        "yaxis":  dict(color=_text, gridcolor=_grid, title_font=dict(color=_text)),
        "legend": dict(font=dict(color=_text)),
    }
    layout.update(overrides)
    return layout

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroGuard | Aircraft Engine Intelligence",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# THEME STATE  (must come right after set_page_config)
# --------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
_DM = st.session_state.dark_mode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gradient_boosting_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_LABELS = {
    "sensor_1": "Fan Inlet Temp", "sensor_2": "LPC Temperature",
    "sensor_3": "HPC Temperature", "sensor_4": "LPT Temperature",
    "sensor_5": "Fan Inlet Pressure", "sensor_6": "Bypass-Duct Pressure",
    "sensor_7": "HPC Pressure", "sensor_8": "Physical Fan Speed",
    "sensor_9": "Physical Core Speed", "sensor_10": "Engine Pressure Ratio",
    "sensor_11": "Static HPC Pressure", "sensor_12": "Fuel Flow Ratio",
    "sensor_13": "Corr. Fan Speed", "sensor_14": "Corr. Core Speed",
    "sensor_15": "Bypass Ratio", "sensor_16": "Burner Fuel-Air Ratio",
    "sensor_17": "Bleed Enthalpy", "sensor_18": "Demanded Fan Speed",
    "sensor_19": "Demanded Corr. Fan Speed", "sensor_20": "HPT Coolant Bleed",
    "sensor_21": "LPT Coolant Bleed",
}

CATEGORY_MAP = {
    "Temperature": ["sensor_2", "sensor_3", "sensor_4"],
    "Pressure": ["sensor_7", "sensor_11"],
    "Rotational Speed": ["sensor_9", "sensor_13", "sensor_14"],
    "Vibration": ["sensor_1", "sensor_5"],
    "Fuel": ["sensor_12"],
    "Oil": ["sensor_6", "sensor_20", "sensor_21"],
    "Electrical": ["sensor_15", "sensor_17"],
    "Other": ["sensor_8", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
}
CATEGORY_COLORS = {
    "Temperature": "#2563eb", "Pressure": "#16a34a", "Rotational Speed": "#f59e0b",
    "Vibration": "#7c3aed", "Fuel": "#ef4444", "Oil": "#0d9488",
    "Electrical": "#eab308", "Other": "#64748b",
}

# --------------------------------------------------------------------------
# CUSTOM CSS  (light / dark — driven by _DM)
# --------------------------------------------------------------------------
_bg     = "#0f172a"        if _DM else "#f4f6fb"
_card   = "#1e293b"        if _DM else "white"
_txt    = "#e2e8f0"        if _DM else "#10182c"
_lbl    = "#94a3b8"        if _DM else "#6b7484"
_sub_c  = "#64748b"        if _DM else "#8a93a3"
_border = "#334155"        if _DM else "#eef1f8"
_shadow = "rgba(0,0,0,.32)" if _DM else "rgba(20,30,60,.08)"

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ===== Sidebar collapsed toggle: stay visible even when header is hidden ===== */
    [data-testid="stSidebarCollapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        background: #0b1730 !important;
        border-radius: 0 10px 10px 0 !important;
        box-shadow: 2px 0 10px rgba(0,0,0,.45) !important;
        padding: 8px 5px !important;
        top: 14px !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg {{
        fill: #e7ecf7 !important;
        stroke: #e7ecf7 !important;
    }}



    /* ===== App background ===== */
    .stApp {{ background-color: {_bg}; }}

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {{ background-color: #0b1730; }}
    section[data-testid="stSidebar"] * {{ color: #e7ecf7 !important; }}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background-color: #16233f; color: #e7ecf7;
    }}

    /* ===== Hero banner ===== */
    .hero-banner {{
        background: linear-gradient(120deg, #0d2c66 0%, #123a86 55%, #1b4fb8 100%);
        border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 22px;
    }}
    .hero-title {{ font-size: 28px; font-weight: 800; margin: 0; }}
    .hero-sub {{ opacity: 0.85; margin-top: 4px; font-size: 14px; }}

    /* ===== Cards ===== */
    .kpi-card, .panel-card {{
        background: {_card} !important; color: {_txt} !important;
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 1px 3px {_shadow}; border: 1px solid {_border};
        margin-bottom: 18px;
    }}
    .kpi-card table, .panel-card table {{ color: {_txt} !important; width: 100%; }}
    .kpi-card td, .panel-card td, .kpi-card th, .panel-card th {{ color: {_txt} !important; }}
    .kpi-label {{ color: {_lbl} !important; font-size: 13px; font-weight: 600; }}
    .kpi-value {{ font-size: 26px; font-weight: 800; color: {_txt} !important; margin: 2px 0; }}
    .kpi-sub {{ color: {_sub_c} !important; font-size: 12px; }}
    .kpi-up {{ color: #16a34a !important; font-weight: 700; font-size: 12px; }}
    .kpi-down {{ color: #dc2626 !important; font-weight: 700; font-size: 12px; }}
    .panel-title {{ font-size: 16px; font-weight: 800; color: {_txt} !important; }}
    .panel-sub {{ color: {_sub_c} !important; font-size: 12.5px; margin-bottom: 10px; }}
    .status-pill {{
        display:inline-block; padding:3px 10px; border-radius:20px;
        background:#e8f9ee !important; color:#159654 !important; font-size:12px; font-weight:700;
    }}
    .cat-box {{
        border-radius: 10px; padding: 14px; color: white !important; font-weight: 700;
        margin-bottom: 10px;
    }}
    .alert-box {{
        background:#fff4e0 !important; color:#7a4a00 !important; border:1px solid #ffe0a3;
        border-radius:12px; padding:14px 16px; margin-bottom: 14px;
    }}
    .alert-box b {{ color:#7a4a00 !important; }}
    .info-box {{
        background:#eaf3ff !important; color:#0b3d8a !important; border:1px solid #cfe4ff;
        border-radius:12px; padding:14px 16px;
    }}
    .info-box b {{ color:#0b3d8a !important; }}
    .ai-box {{
        background:#0b1730 !important; color:#e7ecf7 !important; border-radius:14px;
        padding:20px 22px; margin-top:6px;
    }}
    .ai-box b {{ color:#7dd3fc; }}
    .ai-box li {{ margin-bottom: 6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# DATA LOADING 
# --------------------------------------------------------------------------
def _synth_unit(unit_id: int, n_cycles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    drift = cycles / n_cycles
    scale = rng.uniform(0.88, 1.18)          # each engine gets a slightly different baseline
    noise_level = rng.uniform(0.7, 1.4)

    data = {"unit": unit_id, "cycle": cycles, "op1": 0.0, "op2": 0.0, "op3": 100.0}
    base_values = {
        "sensor_1": (520, 8), "sensor_2": (1180, 40), "sensor_3": (1240, 90),
        "sensor_4": (1400, 60), "sensor_5": (14.6, 0.2), "sensor_6": (21.6, 0.5),
        "sensor_7": (9.0, 0.6), "sensor_8": (2388, 15), "sensor_9": (2200, 300),
        "sensor_10": (1.3, 0.05), "sensor_11": (46, 3), "sensor_12": (520, 20),
        "sensor_13": (2380, 40), "sensor_14": (8100, 120), "sensor_15": (0.03, 0.02),
        "sensor_16": (0.03, 0.005), "sensor_17": (390, 20), "sensor_18": (2388, 5),
        "sensor_19": (100, 2), "sensor_20": (38, 3), "sensor_21": (23, 2),
    }
    for i in range(1, 22):
        key = f"sensor_{i}"
        base, spread = base_values.get(key, (100, 10))
        noise = rng.normal(0, spread * 0.15 * noise_level, n_cycles)
        data[key] = base * scale + drift * spread * 2 * scale + noise
    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def generate_demo_data(fd_choice: str, n_units: int = 8) -> pd.DataFrame:
    base_seed = abs(hash(fd_choice)) % (2**31)
    rng = np.random.default_rng(base_seed)
    frames = []
    for unit in range(1, n_units + 1):
        n_cycles = int(rng.integers(110, 210))
        unit_seed = base_seed + unit * 97
        frames.append(_synth_unit(unit, n_cycles, unit_seed))
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_dataset(fd_choice: str) -> pd.DataFrame:
    patterns = [
        f"test_{fd_choice}.txt", f"test_{fd_choice}.csv",
        f"test {fd_choice} descriptive.txt", f"test {fd_choice} descriptive.csv",
        f"*{fd_choice}*descriptive*", f"*{fd_choice}*.txt", f"*{fd_choice}*.csv",
    ]
    found = None
    if os.path.isdir(DATA_DIR):
        for pat in patterns:
            matches = glob.glob(os.path.join(DATA_DIR, pat))
            if matches:
                found = matches[0]
                break

    if found:
        try:
            if found.endswith(".csv"):
                df = pd.read_csv(found)
            else:
                df = pd.read_csv(found, sep=r"\s+", header=None, engine="python")
            df = df.dropna(axis=1, how="all")
            if df.shape[1] >= len(COLS):
                df = df.iloc[:, : len(COLS)]
                df.columns = COLS
            else:
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                rename = {c: n for c, n in zip(df.columns, COLS)}
                df = df.rename(columns=rename)
            return df
        except Exception:
            pass

    return generate_demo_data(fd_choice)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model, scaler = None, None
    if joblib is not None:
        try:
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
        try:
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None
    return model, scaler


def predict_rul_batch(model, scaler, frame: pd.DataFrame):
    """Vectorized RUL prediction for every row in `frame`."""
    feature_cols = [c for c in frame.columns if c not in ("unit", "cycle")]
    X = frame[feature_cols].values.astype(float)

    if model is not None:
        try:
            n_expected = getattr(scaler, "n_features_in_", None) or getattr(
                model, "n_features_in_", X.shape[1]
            )
            if X.shape[1] > n_expected:
                X = X[:, :n_expected]
            elif X.shape[1] < n_expected:
                X = np.pad(X, ((0, 0), (0, n_expected - X.shape[1])))
            if scaler is not None:
                X = scaler.transform(X)
            preds = model.predict(X)
            return np.ravel(preds).astype(float), True
        except Exception:
            pass

    cycles = frame["cycle"].values if "cycle" in frame else np.arange(1, len(frame) + 1)
    heuristic = np.maximum(5.0, 220 - cycles * 0.9)
    return heuristic, False


def predict_rul_row(model, scaler, row: pd.Series):
    preds, used = predict_rul_batch(model, scaler, pd.DataFrame([row]))
    return float(preds[0]), used


def ai_recommendation(rul_pred, health_label, hpc_change, risk_label, cycle_now):
    if rul_pred < 30:
        headline = "⚠️ Immediate inspection recommended"
        tips = [
            f"Predicted RUL has dropped to ~{rul_pred:,.0f} cycles — schedule a borescope "
            f"inspection within the next {max(3, int(rul_pred * 0.25))} cycles.",
            "Compare current HPC/LPC temperature trend against past failure signatures for this engine family.",
            "Flag spare-parts and downtime planning given the high-risk status.",
        ]
    elif rul_pred < 80:
        headline = "🟡 Plan maintenance in the near term"
        tips = [
            f"RUL is moderate (~{rul_pred:,.0f} cycles) — slot this engine into the next scheduled maintenance window.",
            "Keep a closer watch on temperature and pressure drift over the next 10–15 cycles.",
            "No grounding needed yet, but move this engine up the priority review list.",
        ]
    else:
        headline = "✅ Engine operating normally"
        tips = [
            f"RUL looks healthy (~{rul_pred:,.0f} cycles) — continue the standard monitoring cadence.",
            "Keep tracking sensor drift and re-check this recommendation after the next batch of cycles.",
            "No maintenance action required at this time.",
        ]
    if hpc_change > 1.5:
        tips.append(
            f"HPC temperature rose {hpc_change:.1f}% since the last cycle — worth a closer look even though overall risk is '{risk_label}'."
        )
    return headline, tips


def render_ai_recommendation_panel(rul_pred, health_label, hpc_change, risk_label, cycle_now):
    headline, tips = ai_recommendation(rul_pred, health_label, hpc_change, risk_label, cycle_now)
    bullets = "".join(f"<li>{t}</li>" for t in tips)
    st.markdown(
        f"""<div class="ai-box">
        <div style="font-size:17px;font-weight:800;">🤖 AI Assistant Recommendation</div>
        <div style="margin:6px 0 12px 0;opacity:.85;font-size:13px;">
            Based on current cycle ({cycle_now}), RUL prediction and sensor trend.
        </div>
        <div style="font-size:15px;font-weight:700;margin-bottom:8px;">{headline}</div>
        <div style="font-weight:700;margin-bottom:4px;">What to do next:</div>
        <ul style="font-size:13.5px;">{bullets}</ul>
        </div>""",
        unsafe_allow_html=True,
    )


def compute_category_shares(engine_df: pd.DataFrame):
    """Category % is computed live from this engine's actual data,
    so it changes whenever the selection changes."""
    scores = {}
    for cat, sensors in CATEGORY_MAP.items():
        cols = [c for c in sensors if c in engine_df.columns]
        if not cols:
            scores[cat] = 1.0
            continue
        cvs = []
        for c in cols:
            series = engine_df[c]
            mean_abs = max(abs(series.mean()), 1e-6)
            cvs.append(series.std() / mean_abs)
        scores[cat] = max(float(np.mean(cvs)), 1e-4)
    total = sum(scores.values())
    return {cat: round(v / total * 100, 1) for cat, v in scores.items()}


# --------------------------------------------------------------------------
# RISK DETECTION HELPERS
# --------------------------------------------------------------------------
def compute_sensor_anomalies(engine_df: pd.DataFrame) -> pd.DataFrame:
    """For every sensor column, compute z-score of the last reading vs
    the engine's own history.  Returns a DataFrame with columns:
    sensor, label, current, mean, std, z_score, deviation_pct, status."""
    sensor_cols = [c for c in engine_df.columns if c.startswith("sensor_")]
    rows = []
    latest_row = engine_df.iloc[-1]
    for col in sensor_cols:
        series = engine_df[col]
        mean_val = series.mean()
        std_val  = series.std()
        current  = float(latest_row[col])
        z_score  = (current - mean_val) / max(std_val, 1e-9)
        dev_pct  = (current - mean_val) / max(abs(mean_val), 1e-9) * 100
        if abs(z_score) > 2.5:
            status = "Critical"
        elif abs(z_score) > 1.5:
            status = "Warning"
        else:
            status = "Normal"
        rows.append({
            "sensor": col,
            "label": SENSOR_LABELS.get(col, col),
            "current": round(current, 3),
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "z_score": round(z_score, 2),
            "deviation_pct": round(dev_pct, 2),
            "status": status,
        })
    return pd.DataFrame(rows)


def compute_failure_risk_score(rul_pred: float, anomaly_df: pd.DataFrame) -> float:
    """Returns a failure risk score in [0, 100].
    70 % from RUL proximity, 30 % from sensor anomaly severity."""
    # RUL component: 0 cycles → 100 risk, 150+ cycles → 0 risk
    rul_capped = min(max(rul_pred, 0), 150)
    rul_score  = (1 - rul_capped / 150) * 100

    # Anomaly component: average |z-score| across all sensors, clipped to [0,3]
    if anomaly_df is not None and not anomaly_df.empty:
        mean_abs_z    = anomaly_df["z_score"].abs().mean()
        anomaly_score = min(mean_abs_z / 3, 1) * 100
    else:
        anomaly_score = 0

    return round(0.70 * rul_score + 0.30 * anomaly_score, 1)


def classify_risk(rul_pred: float):
    """Returns (label, color, bg_color, icon, description)."""
    if rul_pred < 30:
        return ("CRITICAL", "#dc2626", "#fef2f2", "🔴",
                "Immediate inspection required — engine is at high risk of failure.")
    elif rul_pred < 80:
        return ("WARNING", "#d97706", "#fffbeb", "🟡",
                "Schedule maintenance soon — engine is showing signs of wear.")
    else:
        return ("NORMAL", "#16a34a", "#f0fdf4", "🟢",
                "Engine is operating within normal parameters.")


@st.cache_data(show_spinner=False)
def get_fleet_risk_summary(_df: pd.DataFrame, fd_key: str) -> pd.DataFrame:
    """Compute failure risk score for every engine unit in the dataset.
    `fd_key` is only used as a cache-buster string."""
    rows = []
    model_, scaler_ = load_model_and_scaler()
    for uid in sorted(_df["unit"].unique()):
        eng = _df[_df["unit"] == uid].reset_index(drop=True)
        if eng.empty:
            continue
        rul_arr, _ = predict_rul_batch(model_, scaler_, eng)
        rul_val    = float(rul_arr[-1])
        anoms      = compute_sensor_anomalies(eng)
        risk       = compute_failure_risk_score(rul_val, anoms)
        label, color, _, icon, _ = classify_risk(rul_val)
        rows.append({"engine": f"#{int(uid):02d}", "rul": round(rul_val, 1),
                     "risk_score": risk, "status": label,
                     "color": color, "icon": icon})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;padding:6px 0 18px 0;'>"
        "<span style='font-size:26px;'>🛩️</span>"
        "<div><div style='font-size:20px;font-weight:800;'>AeroGuard</div>"
        "<div style='font-size:11px;opacity:.7;'>Aircraft Engine Intelligence</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📈 Engine Monitoring", "🗄️ Sensor Data", "🔧 Maintenance", "⚠️ Risk Detection"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("---")
    st.markdown("**Engine Selection**")
    fd_choice = st.selectbox(
        "Dataset", ["FD001", "FD002", "FD003", "FD004"], label_visibility="collapsed", key="fd_choice"
    )

    df = load_dataset(fd_choice)
    unit_ids = sorted(df["unit"].unique().tolist()) if "unit" in df else [1]
    engine_id = st.selectbox(
        "Engine #", unit_ids, format_func=lambda u: f"Engine #{int(u):02d}", key="engine_id"
    )

    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown("🟢 **System Online**")
    st.caption(f"Last updated: {dt.datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")
    st.markdown("**Display Settings**")
    _dm_lbl = "☀️ Light Mode" if _DM else "🌙 Dark Mode"
    _dm_help = "Switch to light theme" if _DM else "Switch to dark theme"
    if st.button(_dm_lbl, use_container_width=True, key="theme_toggle_btn", help=_dm_help):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# --------------------------------------------------------------------------
# SHARED COMPUTATIONS (based on selected engine/dataset, used across all pages)
# --------------------------------------------------------------------------
engine_df = df[df["unit"] == engine_id].reset_index(drop=True) if "unit" in df else df
if engine_df.empty:
    engine_df = df
latest = engine_df.iloc[-1]
prev = engine_df.iloc[-2] if len(engine_df) > 1 else latest

model, scaler = load_model_and_scaler()
rul_pred, model_used = predict_rul_row(model, scaler, latest)

hpc_col = "sensor_3" if "sensor_3" in engine_df else engine_df.columns[5]
lpc_col = "sensor_2" if "sensor_2" in engine_df else engine_df.columns[6]
press_col = "sensor_7" if "sensor_7" in engine_df else engine_df.columns[7]
rpm_col = "sensor_9" if "sensor_9" in engine_df else engine_df.columns[8]

hpc_temp = float(latest[hpc_col])
lpc_temp = float(latest[lpc_col])
pressure = float(latest[press_col])
rpm = float(latest[rpm_col])
cycle_now = int(latest["cycle"]) if "cycle" in latest else len(engine_df)

hpc_change = (hpc_temp - float(prev[hpc_col])) / max(abs(float(prev[hpc_col])), 1e-6) * 100
cycle_change = (cycle_now - int(prev["cycle"])) / max(cycle_now, 1) * 100 if "cycle" in prev else 0.0

if rul_pred < 30:
    health_label, health_color, risk_label = "Critical", "#dc2626", "High Risk"
elif rul_pred < 80:
    health_label, health_color, risk_label = "Fair", "#d97706", "Moderate"
else:
    health_label, health_color, risk_label = "Healthy", "#16a34a", "Low Risk"

category_shares = compute_category_shares(engine_df)

# --------------------------------------------------------------------------
# HERO BANNER (shown at the top of every page)
# --------------------------------------------------------------------------
page_titles = {
    "🏠 Dashboard": ("Aircraft Engine Monitoring Center", "Real-time sensor monitoring and predictive maintenance dashboard"),
    "📈 Engine Monitoring": ("Engine Monitoring", "Deep-dive sensor trends for the selected engine"),
    "🗄️ Sensor Data": ("Sensor Data", "Raw sensor readings and descriptive statistics"),
    "🔧 Maintenance": ("Maintenance Center", "Health status, RUL trend and AI-driven recommendations"),
    "⚠️ Risk Detection": ("Engine Failure Risk Detection", "ML-powered failure risk classification · Normal · Warning · Critical"),
}
title_text, sub_text = page_titles[page]
st.markdown(
    f"""
    <div class="hero-banner">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <p class="hero-title">{title_text}</p>
                <p class="hero-sub">{sub_text}</p>
            </div>
            <div style="text-align:right;font-size:13px;opacity:.9;">
                🟢 System Online &nbsp;|&nbsp; {dt.datetime.now().strftime('%b %d, %Y')} &nbsp;{dt.datetime.now().strftime('%H:%M')}
                <br/>👤 Engine Operator &nbsp;·&nbsp; Engine #{int(engine_id):02d} &nbsp;·&nbsp; {fd_choice}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# PAGE 1 — DASHBOARD
# ==========================================================================
if page == "🏠 Dashboard":
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⚙️ Engine Cycle</div>
            <div class="kpi-value">{cycle_now}</div>
            <div class="kpi-sub">Current operating cycle</div>
            </div>""", unsafe_allow_html=True)
    with k2:
        arrow, cls = ("↑", "kpi-up") if hpc_change >= 0 else ("↓", "kpi-down")
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🌡️ HPC Temperature</div>
            <div class="kpi-value">{hpc_temp:,.1f} °C</div>
            <div class="kpi-sub">Outlet temperature &nbsp;<span class="{cls}">{arrow} {abs(hpc_change):.1f}%</span></div>
            </div>""", unsafe_allow_html=True)
    with k3:
        src = "Gradient Boosting model" if model_used else "Estimated (model not loaded)"
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📊 RUL Prediction</div>
            <div class="kpi-value">{rul_pred:,.0f} cycles</div>
            <div class="kpi-sub">{src}</div>
            </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">❤️ Engine Health</div>
            <div class="kpi-value" style="color:{health_color};">{health_label}</div>
            <div class="kpi-sub">Based on current RUL prediction</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">📈 Sensor Trends</div>
            <div class="panel-sub">Historical sensor behavior for the selected engine.</div>
            </div>""", unsafe_allow_html=True)
        # Real sensors have very different units/scales (temp in the 1000s,
        # pressure in the 100s-1000s, RPM in the 1000s-10000s) — plotting raw
        # values on one axis lets the biggest one (usually pressure) flatten
        # everything else near zero. So we plot each sensor normalized to its
        # own 0-1 range here and show the real value on hover.
        trend_specs = [
            (hpc_col, "HPC Temperature", "#2563eb", "°C"),
            (lpc_col, "LPC Temperature", "#16a34a", "°C"),
            (press_col, "Pressure", "#f59e0b", "psia"),
            (rpm_col, "RPM", "#7c3aed", "rpm"),
        ]
        fig = go.Figure()
        flat_sensors = []
        for col, label, color, unit_ in trend_specs:
            raw = engine_df[col]
            span = raw.max() - raw.min()
            if span < max(abs(raw.mean()), 1e-6) * 0.002:
                flat_sensors.append(label)
            norm = (raw - raw.min()) / span if span > 1e-9 else raw * 0 + 0.5
            fig.add_trace(go.Scatter(
                x=engine_df["cycle"], y=norm, name=label,
                line=dict(color=color, width=2), customdata=raw,
                hovertemplate=f"{label}: %{{customdata:,.2f}} {unit_}<br>Cycle: %{{x}}<extra></extra>",
            ))
        fig.update_layout(**styled_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color=TEXT_DARK), orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Engine Cycle", yaxis_title="Normalized value (hover for real reading)"))
        st.plotly_chart(fig, use_container_width=True, key="dash_trend_chart")
        if flat_sensors:
            st.caption(
                f"Note: {', '.join(flat_sensors)} barely change for this engine in the "
                f"{fd_choice} dataset — that's a real characteristic of this sensor here, "
                f"not a rendering issue. Try the Engine Monitoring page to inspect any single sensor closely."
            )

    with c2:
        st.markdown(
            f"""<div class="panel-card">
            <div class="panel-title">🖼️ Engine Overview</div>
            <div style="text-align:center;font-size:70px;margin:8px 0;">🛩️</div>
            <table style="width:100%;font-size:14px;">
                <tr><td>⚙️ Engine Cycle</td><td style="text-align:right;font-weight:700;">{cycle_now}</td></tr>
                <tr><td>🔄 Rotational Speed</td><td style="text-align:right;font-weight:700;">{rpm:,.0f} RPM</td></tr>
                <tr><td>📟 Pressure</td><td style="text-align:right;font-weight:700;">{pressure:,.2f} psia</td></tr>
                <tr><td>🌡️ HPC Temperature</td><td style="text-align:right;font-weight:700;">{hpc_temp:,.1f} °C</td></tr>
                <tr><td>🌡️ LPC Temperature</td><td style="text-align:right;font-weight:700;">{lpc_temp:,.1f} °C</td></tr>
            </table>
            </div>""", unsafe_allow_html=True)

    c3, c4 = st.columns([2, 1])
    with c3:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">🩺 Current Sensor Status</div>
            <div class="panel-sub">Live values from engine sensors.</div>
            </div>""", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        for col, label, val, unit_ in [
            (s1, "HPC Temperature", hpc_temp, "°C"), (s2, "LPC Temperature", lpc_temp, "°C"),
            (s3, "Temp. Difference", hpc_temp - lpc_temp, "°C"), (s4, "Pressure", pressure, "psia"),
        ]:
            with col:
                st.markdown(
                    f"""<div class="panel-card" style="text-align:center;">
                    <div style="font-size:13px;color:#6b7484;font-weight:600;">{label}</div>
                    <div style="font-size:22px;font-weight:800;margin:6px 0;">{val:,.1f} {unit_}</div>
                    <span class="status-pill">Normal</span>
                    </div>""", unsafe_allow_html=True)

    with c4:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=rul_pred,
            number={"suffix": " cyc", "font": {"size": 26, "color": TEXT_DARK}},
            gauge={"axis": {"range": [0, max(200, rul_pred * 1.2)], "tickcolor": TEXT_DARK},
                   "bar": {"color": health_color}, "bgcolor": "#eef1f8"},
            domain={"x": [0, 1], "y": [0, 1]}))
        gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=0),
                             paper_bgcolor="white", font=dict(color=TEXT_DARK))
        st.markdown('<div class="panel-card"><div class="panel-title">❤️ Engine Health Status</div></div>', unsafe_allow_html=True)
        st.plotly_chart(gauge, use_container_width=True, key="dash_gauge_chart")
        st.markdown(
            f"""<div class="panel-card">
            Health Level: <b style="color:{health_color};">{health_label}</b><br/>
            Risk Status: <b>{risk_label}</b>
            </div>""", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">🧩 Category-wise Sensor Distribution</div>
            <div class="panel-sub">Computed live from this engine's sensor variability.</div>
            """, unsafe_allow_html=True)
        cc = st.columns(2)
        for i, (name, pct) in enumerate(category_shares.items()):
            with cc[i % 2]:
                st.markdown(
                    f"""<div class="cat-box" style="background:{CATEGORY_COLORS[name]};">
                    {name}<br/><span style="font-size:20px;">{pct}%</span>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">🎯 Parameter Performance</div>
            <div class="panel-sub">Current vs normal range for key parameters.</div>
            """, unsafe_allow_html=True)
        radar_labels = ["Temperature", "Pressure", "Speed", "Efficiency", "Vibration"]
        current_vals = [min(100, hpc_temp / 15), min(100, pressure * 10), min(100, rpm / 25),
                         max(10, 100 - (rul_pred / 2)), min(100, category_shares.get("Vibration", 20) * 3)]
        normal_vals = [70, 65, 75, 80, 30]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=normal_vals + [normal_vals[0]], theta=radar_labels + [radar_labels[0]],
                                         name="Normal Range", line=dict(color="#c7d2fe")))
        radar.add_trace(go.Scatterpolar(r=current_vals + [current_vals[0]], theta=radar_labels + [radar_labels[0]],
                                         name="Current", line=dict(color="#2563eb")))
        radar.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10),
                             paper_bgcolor="white", font=dict(color=TEXT_DARK),
                             polar=dict(bgcolor="white",
                                        radialaxis=dict(visible=True, range=[0, 100], color=TEXT_DARK,
                                                         gridcolor="#e5e9f2"),
                                        angularaxis=dict(color=TEXT_DARK)),
                             showlegend=True, legend=dict(font=dict(color=TEXT_DARK), orientation="h", y=-0.1))
        st.plotly_chart(radar, use_container_width=True, key="dash_radar_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with c7:
        st.markdown('<div class="panel-title">🔧 Maintenance Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Recommendations based on engine condition.</div>', unsafe_allow_html=True)
        if rul_pred < 30:
            st.markdown(
                f"""<div class="alert-box"><b>⚠️ Maintenance Alert</b><br/>
                Predicted RUL is only <b>{rul_pred:,.0f} cycles</b>. Schedule an inspection soon.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div class="alert-box"><b>✅ No Urgent Action Needed</b><br/>
                Predicted RUL is <b>{rul_pred:,.0f} cycles</b>. Parameters are within expected bounds.
                </div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class="info-box"><b>💡 Why it matters?</b><br/>
            Predictive maintenance helps reduce downtime, lower costs and improve engine life.
            </div>""", unsafe_allow_html=True)

    st.markdown("### ")
    render_ai_recommendation_panel(rul_pred, health_label, hpc_change, risk_label, cycle_now)

# ==========================================================================
# PAGE 2 — ENGINE MONITORING
# ==========================================================================
elif page == "📈 Engine Monitoring":
    sensor_options = [c for c in engine_df.columns if c.startswith("sensor_")]
    chosen_sensor = st.selectbox(
        "Choose a sensor to inspect",
        sensor_options,
        format_func=lambda c: f"{c} — {SENSOR_LABELS.get(c, c)}",
    )
    series = engine_df[chosen_sensor]

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [
        (m1, "Current", series.iloc[-1]), (m2, "Mean", series.mean()),
        (m3, "Min", series.min()), (m4, "Max", series.max()),
    ]:
        with col:
            st.markdown(
                f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value">{val:,.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">📈 {SENSOR_LABELS.get(chosen_sensor, chosen_sensor)} — full trend</div>
        <div class="panel-sub">Engine #{int(engine_id):02d} · {fd_choice} · {len(engine_df)} cycles recorded</div>
        </div>""", unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=engine_df["cycle"], y=series, mode="lines",
                               line=dict(color="#2563eb", width=2), fill="tozeroy",
                               fillcolor="rgba(37,99,235,0.08)"))
    fig1.update_layout(**styled_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Engine Cycle", yaxis_title=SENSOR_LABELS.get(chosen_sensor, chosen_sensor)))
    st.plotly_chart(fig1, use_container_width=True, key="mon_single_sensor_chart")

    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">🔬 Multi-sensor comparison</div>
        <div class="panel-sub">Pick a few sensors to overlay (normalized 0–1 so scales line up).</div>
        </div>""", unsafe_allow_html=True)
    default_multi = [hpc_col, lpc_col, press_col, rpm_col]
    multi_sensors = st.multiselect(
        "Sensors to compare", sensor_options,
        default=[s for s in default_multi if s in sensor_options],
        format_func=lambda c: f"{c} — {SENSOR_LABELS.get(c, c)}",
    )
    if multi_sensors:
        fig2 = go.Figure()
        for s in multi_sensors:
            vals = engine_df[s]
            norm = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
            fig2.add_trace(go.Scatter(x=engine_df["cycle"], y=norm, name=SENSOR_LABELS.get(s, s)))
        fig2.update_layout(**styled_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Engine Cycle", yaxis_title="Normalized value",
            legend=dict(font=dict(color=TEXT_DARK), orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1)))
        st.plotly_chart(fig2, use_container_width=True, key="mon_multi_sensor_chart")
    else:
        st.info("Select at least one sensor above to compare.")

# ==========================================================================
# PAGE 3 — SENSOR DATA
# ==========================================================================
elif page == "🗄️ Sensor Data":
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🗄️ Raw Sensor Data</div>
        <div class="panel-sub">Engine #{int(engine_id):02d} · {fd_choice} · {len(engine_df)} rows</div>
        </div>""", unsafe_allow_html=True)

    show_cols = st.multiselect(
        "Columns to show", list(engine_df.columns), default=list(engine_df.columns)[:10]
    )
    st.dataframe(engine_df[show_cols] if show_cols else engine_df, use_container_width=True, height=380)

    csv_bytes = engine_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download this engine's data as CSV", data=csv_bytes,
        file_name=f"{fd_choice}_engine_{int(engine_id):02d}.csv", mime="text/csv",
    )

    st.markdown('<div class="panel-title" style="margin-top:18px;">📊 Descriptive statistics</div>', unsafe_allow_html=True)
    st.dataframe(engine_df.describe().T, use_container_width=True)

# ==========================================================================
# PAGE 4 — MAINTENANCE
# ==========================================================================
elif page == "🔧 Maintenance":
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">📊 Predicted RUL</div>
            <div class="kpi-value">{rul_pred:,.0f} cycles</div>
            <div class="kpi-sub">{'Model prediction' if model_used else 'Heuristic estimate'}</div></div>""",
            unsafe_allow_html=True)
    with m2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">❤️ Health Level</div>
            <div class="kpi-value" style="color:{health_color};">{health_label}</div></div>""",
            unsafe_allow_html=True)
    with m3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">🚨 Risk Status</div>
            <div class="kpi-value" style="color:{health_color};">{risk_label}</div></div>""",
            unsafe_allow_html=True)

    rul_series, _ = predict_rul_batch(model, scaler, engine_df)
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">📉 RUL Trend Across Cycles</div>
        <div class="panel-sub">How the predicted remaining useful life changes cycle by cycle.</div>
        </div>""", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=engine_df["cycle"], y=rul_series, mode="lines",
                               line=dict(color="#dc2626", width=2), fill="tozeroy",
                               fillcolor="rgba(220,38,38,0.08)"))
    fig3.add_hline(y=30, line_dash="dash", line_color="#f59e0b", annotation_text="Critical threshold")
    fig3.update_layout(**styled_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Engine Cycle", yaxis_title="Predicted RUL"))
    st.plotly_chart(fig3, use_container_width=True, key="maint_rul_trend_chart")

    if rul_pred < 30:
        st.markdown(
            f"""<div class="alert-box"><b>⚠️ Maintenance Alert</b><br/>
            Predicted RUL is only <b>{rul_pred:,.0f} cycles</b> for Engine #{int(engine_id):02d}.
            Schedule an inspection soon to avoid unplanned downtime.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""<div class="alert-box"><b>✅ No Urgent Action Needed</b><br/>
            Predicted RUL is <b>{rul_pred:,.0f} cycles</b>. Engine parameters are within expected operating bounds.
            </div>""", unsafe_allow_html=True)

    render_ai_recommendation_panel(rul_pred, health_label, hpc_change, risk_label, cycle_now)

# ==========================================================================
# PAGE 5 — RISK DETECTION
# ==========================================================================
elif page == "⚠️ Risk Detection":

    risk_label_full, risk_color, risk_bg, risk_icon, risk_desc = classify_risk(rul_pred)
    anomaly_df = compute_sensor_anomalies(engine_df)
    failure_risk_score = compute_failure_risk_score(rul_pred, anomaly_df)

    # ------------------------------------------------------------------
    # 1. RISK CLASSIFICATION BANNER
    # ------------------------------------------------------------------
    banner_border = risk_color
    st.markdown(
        f"""
        <div style="
            background:{risk_bg};
            border:2.5px solid {banner_border};
            border-radius:18px;
            padding:28px 36px;
            margin-bottom:22px;
            display:flex;
            align-items:center;
            gap:28px;
        ">
            <div style="font-size:64px;line-height:1;">{risk_icon}</div>
            <div style="flex:1;">
                <div style="font-size:13px;font-weight:600;color:#64748b;letter-spacing:.08em;text-transform:uppercase;">
                    Engine #{int(engine_id):02d} &nbsp;·&nbsp; {fd_choice} &nbsp;·&nbsp; Cycle {cycle_now}
                </div>
                <div style="font-size:38px;font-weight:900;color:{risk_color};margin:4px 0 6px;">
                    {risk_label_full}
                </div>
                <div style="font-size:15px;color:#374151;">{risk_desc}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:12px;color:#64748b;font-weight:600;">FAILURE RISK SCORE</div>
                <div style="font-size:52px;font-weight:900;color:{risk_color};line-height:1;">{failure_risk_score:.0f}<span style="font-size:24px;">%</span></div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">Higher = more urgent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 2. KPI ROW
    # ------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    critical_count = int((anomaly_df["status"] == "Critical").sum())
    warning_count  = int((anomaly_df["status"] == "Warning").sum())
    normal_count   = int((anomaly_df["status"] == "Normal").sum())

    with k1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📊 Predicted RUL</div>
            <div class="kpi-value">{rul_pred:,.0f} cycles</div>
            <div class="kpi-sub">{'Model prediction' if model_used else 'Heuristic estimate'}</div>
            </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🔴 Critical Sensors</div>
            <div class="kpi-value" style="color:#dc2626;">{critical_count}</div>
            <div class="kpi-sub">Z-score &gt; 2.5 deviation</div>
            </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🟡 Warning Sensors</div>
            <div class="kpi-value" style="color:#d97706;">{warning_count}</div>
            <div class="kpi-sub">Z-score 1.5 – 2.5 deviation</div>
            </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🟢 Normal Sensors</div>
            <div class="kpi-value" style="color:#16a34a;">{normal_count}</div>
            <div class="kpi-sub">Within expected range</div>
            </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. RISK SCORE GAUGE  +  SENSOR ANOMALY HEATMAP
    # ------------------------------------------------------------------
    gc, hc = st.columns([1, 2])

    with gc:
        st.markdown('<div class="panel-card"><div class="panel-title">📉 Risk Score Gauge</div></div>',
                    unsafe_allow_html=True)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=failure_risk_score,
            delta={"reference": 50, "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#16a34a"}},
            number={"suffix": "%", "font": {"size": 30, "color": TEXT_DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_DARK, "tickwidth": 1},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#f1f5f9",
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 30],  "color": "#dcfce7"},
                    {"range": [30, 70], "color": "#fef9c3"},
                    {"range": [70, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#7c3aed", "width": 3},
                    "thickness": 0.8,
                    "value": failure_risk_score,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        gauge_fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="white",
            font=dict(color=TEXT_DARK),
        )
        st.plotly_chart(gauge_fig, use_container_width=True, key="risk_gauge")

        # Risk level legend
        st.markdown(
            """
            <div style="display:flex;gap:8px;justify-content:center;margin-top:-10px;">
                <span style="background:#dcfce7;color:#166534;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:700;">0–30 Normal</span>
                <span style="background:#fef9c3;color:#854d0e;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:700;">30–70 Warning</span>
                <span style="background:#fee2e2;color:#991b1b;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:700;">70–100 Critical</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hc:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">🌡️ Sensor Anomaly Heatmap</div>'
            '<div class="panel-sub">Z-score of each sensor vs. this engine\'s own baseline across all cycles (last 40 cycles shown).</div></div>',
            unsafe_allow_html=True,
        )
        # Build z-score matrix: rows = sensors, cols = cycles (last 40)
        sensor_cols_heat = [c for c in engine_df.columns if c.startswith("sensor_")]
        window_df = engine_df.tail(40)
        z_matrix = []
        y_labels  = []
        for col in sensor_cols_heat:
            series = engine_df[col]          # use full history for stats
            mu  = series.mean()
            sig = series.std()
            z   = ((window_df[col] - mu) / max(sig, 1e-9)).clip(-3, 3)
            z_matrix.append(z.values)
            y_labels.append(SENSOR_LABELS.get(col, col))
        z_matrix = np.array(z_matrix)
        _dm_heat = st.session_state.get("dark_mode", False)
        _heat_text = "#e2e8f0" if _dm_heat else TEXT_DARK
        _heat_bg   = "#1e293b" if _dm_heat else "white"
        heat_fig = go.Figure(go.Heatmap(
            z=z_matrix,
            x=window_df["cycle"].tolist(),
            y=y_labels,
            colorscale=[
                [0.0,  "#16a34a"],
                [0.33, "#86efac"],
                [0.5,  "#fef9c3"],
                [0.67, "#fbbf24"],
                [1.0,  "#dc2626"],
            ],
            zmid=0,
            colorbar=dict(
                title=dict(text="Z-score", font=dict(color=_heat_text)),
                tickfont=dict(color=_heat_text),
            ),
            hovertemplate="Sensor: %{y}<br>Cycle: %{x}<br>Z-score: %{z:.2f}<extra></extra>",
        ))
        heat_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=_heat_bg,
            plot_bgcolor=_heat_bg,
            font=dict(color=_heat_text),
            xaxis=dict(title="Engine Cycle", color=_heat_text, gridcolor="#334155" if _dm_heat else "#e5e9f2"),
            yaxis=dict(title="Sensor", color=_heat_text, autorange="reversed",
                       gridcolor="#334155" if _dm_heat else "#e5e9f2"),
        )
        st.plotly_chart(heat_fig, use_container_width=True, key="risk_heatmap")

    # ------------------------------------------------------------------
    # 4. PER-SENSOR RISK TABLE
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="panel-card"><div class="panel-title">🔬 Per-Sensor Anomaly Analysis</div>'
        '<div class="panel-sub">Current reading vs. engine\'s historical baseline. Sorted by |Z-score| descending.</div></div>',
        unsafe_allow_html=True,
    )

    display_df = anomaly_df.sort_values("z_score", key=lambda s: s.abs(), ascending=False).copy()

    def _status_badge(s):
        if s == "Critical":
            return "🔴 Critical"
        elif s == "Warning":
            return "🟡 Warning"
        return "🟢 Normal"

    display_df["Risk Level"] = display_df["status"].apply(_status_badge)
    display_df["Deviation"]  = display_df["deviation_pct"].apply(lambda x: f"{x:+.1f}%")
    display_df["Z-Score"]    = display_df["z_score"].apply(lambda x: f"{x:+.2f}")
    show_cols = ["label", "current", "mean", "std", "Z-Score", "Deviation", "Risk Level"]
    rename_map = {"label": "Sensor", "current": "Current Value", "mean": "Baseline Mean", "std": "Std Dev"}
    st.dataframe(
        display_df[show_cols].rename(columns=rename_map),
        use_container_width=True,
        height=340,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # 5. MAINTENANCE ALERT PANEL
    # ------------------------------------------------------------------
    st.markdown("### 🚨 Maintenance Action")
    if risk_label_full == "CRITICAL":
        st.markdown(
            f"""
            <div style="background:#fef2f2;border:2px solid #dc2626;border-radius:14px;padding:22px 26px;margin-bottom:16px;">
                <div style="font-size:20px;font-weight:900;color:#dc2626;">🔴 CRITICAL ALERT — Immediate Action Required</div>
                <div style="margin-top:10px;font-size:14px;color:#7f1d1d;">
                    <b>Engine #{int(engine_id):02d}</b> has a predicted RUL of only <b>{rul_pred:,.0f} cycles</b>
                    and a failure risk score of <b>{failure_risk_score:.0f}%</b>.
                </div>
                <ul style="margin-top:10px;font-size:13.5px;color:#7f1d1d;">
                    <li>Ground the engine immediately if safety margins cannot be confirmed.</li>
                    <li>Schedule a borescope inspection within the next <b>{max(1, int(rul_pred * 0.2))} cycles</b>.</li>
                    <li>Activate spare-parts logistics and downtime planning.</li>
                    <li>Review HPC/LPC temperature anomalies against past failure signatures.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif risk_label_full == "WARNING":
        st.markdown(
            f"""
            <div style="background:#fffbeb;border:2px solid #d97706;border-radius:14px;padding:22px 26px;margin-bottom:16px;">
                <div style="font-size:20px;font-weight:900;color:#d97706;">🟡 WARNING — Schedule Maintenance Soon</div>
                <div style="margin-top:10px;font-size:14px;color:#78350f;">
                    <b>Engine #{int(engine_id):02d}</b> has a predicted RUL of <b>{rul_pred:,.0f} cycles</b>
                    and a failure risk score of <b>{failure_risk_score:.0f}%</b>.
                </div>
                <ul style="margin-top:10px;font-size:13.5px;color:#78350f;">
                    <li>Slot this engine into the next scheduled maintenance window.</li>
                    <li>Increase sensor monitoring frequency over the next 10–15 cycles.</li>
                    <li>No grounding needed yet — continue standard operations with caution.</li>
                    <li>Move this engine up the priority review list.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:14px;padding:22px 26px;margin-bottom:16px;">
                <div style="font-size:20px;font-weight:900;color:#16a34a;">🟢 NORMAL — No Immediate Action Required</div>
                <div style="margin-top:10px;font-size:14px;color:#14532d;">
                    <b>Engine #{int(engine_id):02d}</b> has a predicted RUL of <b>{rul_pred:,.0f} cycles</b>
                    and a failure risk score of <b>{failure_risk_score:.0f}%</b>.
                </div>
                <ul style="margin-top:10px;font-size:13.5px;color:#14532d;">
                    <li>Continue standard monitoring cadence.</li>
                    <li>Re-run risk assessment after the next batch of cycles.</li>
                    <li>No maintenance action required at this time.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # 6. FLEET RISK OVERVIEW
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="panel-card"><div class="panel-title">🌐 Fleet Risk Overview</div>'
        f'<div class="panel-sub">Failure risk scores for all engines in the {fd_choice} dataset.</div></div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Computing fleet risk scores…"):
        fleet_df = get_fleet_risk_summary(df, fd_choice)

    if not fleet_df.empty:
        # Color each bar by its risk status
        bar_colors = fleet_df["color"].tolist()
        fleet_fig = go.Figure()
        fleet_fig.add_trace(go.Bar(
            x=fleet_df["engine"],
            y=fleet_df["risk_score"],
            marker_color=bar_colors,
            text=fleet_df.apply(lambda r: f"{r['icon']} {r['status']}<br>{r['risk_score']:.0f}%", axis=1),
            textposition="outside",
            hovertemplate=(
                "Engine: %{x}<br>"
                "Risk Score: %{y:.1f}%<br>"
                "RUL: %{customdata:.0f} cycles<extra></extra>"
            ),
            customdata=fleet_df["rul"].tolist(),
            cliponaxis=False,
        ))
        # Highlight selected engine
        selected_label = f"#{int(engine_id):02d}"
        if selected_label in fleet_df["engine"].values:
            sel_idx  = fleet_df[fleet_df["engine"] == selected_label].index[0]
            sel_risk = fleet_df.loc[sel_idx, "risk_score"]
            fleet_fig.add_annotation(
                x=selected_label, y=sel_risk + 5,
                text="◀ Selected", showarrow=False,
                font=dict(color="#7c3aed", size=11, family="sans-serif"),
            )
        fleet_fig.add_hline(y=70, line_dash="dash", line_color="#dc2626",
                             annotation_text="Critical threshold (70%)",
                             annotation_font_color="#dc2626")
        fleet_fig.add_hline(y=30, line_dash="dot",  line_color="#d97706",
                             annotation_text="Warning threshold (30%)",
                             annotation_font_color="#d97706")
        fleet_fig.update_layout(**styled_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Engine",
            yaxis_title="Failure Risk Score (%)",
            yaxis=dict(range=[0, 115], color=TEXT_DARK, gridcolor="#e5e9f2"),
            bargap=0.35,
        ))
        st.plotly_chart(fleet_fig, use_container_width=True, key="fleet_risk_chart")

        # Summary table
        summary_cols = ["engine", "rul", "risk_score", "status"]
        summary_rename = {"engine": "Engine", "rul": "Predicted RUL (cycles)",
                          "risk_score": "Risk Score (%)", "status": "Risk Level"}
        st.dataframe(
            fleet_df[summary_cols].rename(columns=summary_rename),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

st.caption(
    f"AeroGuard · Safer Skies, Smarter Engines · Dataset: {fd_choice} · "
    f"Model: {'loaded' if model is not None else 'not found — showing heuristic RUL'}"
)