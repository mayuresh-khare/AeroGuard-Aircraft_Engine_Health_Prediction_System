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
    footer    {{visibility: hidden;}}

    /* ===== Header: make TRANSPARENT (NOT hidden) so sidebar toggle stays clickable ===== */
    header[data-testid="stHeader"] {{
        background:    transparent !important;
        border-bottom: none        !important;
        box-shadow:    none        !important;
    }}
    /* Hide the coloured decoration stripe */
    [data-testid="stDecoration"]    {{ display:     none    !important; }}
    /* Hide right-side toolbar (deploy button, kebab menu) */
    [data-testid="stToolbarActions"]{{ visibility:  hidden  !important; }}
    [data-testid="stStatusWidget"]  {{ visibility:  hidden  !important; }}
    /* Hide the top-right main-menu entirely */
    [data-testid="stMainMenuPopover"]{{ display:    none    !important; }}

    /* ===== Sidebar collapsed toggle — styled as a dark pill button ===== */
    [data-testid="stSidebarCollapsedControl"] {{
        visibility:    visible      !important;
        opacity:       1            !important;
        background:    #0b1730     !important;
        border-radius: 0 10px 10px 0 !important;
        box-shadow:    2px 0 12px rgba(0,0,0,.5) !important;
        padding:       10px 6px    !important;
        top:           50%         !important;
        transform:     translateY(-50%) !important;
        transition:    background .2s  !important;
    }}
    [data-testid="stSidebarCollapsedControl"]:hover {{
        background: #1e3a5f !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg {{
        fill:   #e7ecf7 !important;
        stroke: #e7ecf7 !important;
        width:  20px   !important;
        height: 20px   !important;
    }}


    /* ===== App background ===== */
    .stApp {{ background-color: {_bg}; }}

    /* ===== Sidebar base ===== */
    section[data-testid="stSidebar"] {{
        background-color: #0b1730 !important;
        /* Override Streamlit CSS custom properties so native widgets stop using white */
        --background-color: #0b1730;
        --secondary-background-color: #16233f;
        --text-color: #e7ecf7;
    }}

    /* Force ALL sidebar text to be light — deepest wildcard */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {{
        color: #e7ecf7 !important;
    }}

    /* Clear white/grey backgrounds from ALL intermediate Streamlit wrapper elements */
    section[data-testid="stSidebar"] .element-container,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        background-color: transparent !important;
        color: #e7ecf7 !important;
    }}

    /* ---- The actual select INPUT box ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: #1a2f54 !important;
        border: 1.5px solid #3b5ea6 !important;
        border-radius: 8px !important;
    }}

    /* ---- Every layer inside the select (value, arrow, search) ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div > div > div {{
        background-color: transparent !important;
        color: #e7ecf7 !important;
    }}

    /* ---- Spans that hold the displayed value text ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: #e7ecf7 !important;
        background-color: transparent !important;
    }}

    /* ---- Search / type-ahead input ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"] input {{
        color:       #e7ecf7 !important;
        caret-color: #e7ecf7 !important;
        background:  transparent !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {{
        color: #7a9fd4 !important;
    }}

    /* ---- Chevron ▾ arrow SVG ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill:   #e7ecf7 !important;
        stroke: #e7ecf7 !important;
    }}

    /* ---- Dropdown list (rendered as portal OUTSIDE sidebar) ---- */
    [data-baseweb="popover"] {{ background-color: #16233f !important; }}
    [data-baseweb="popover"] * {{ color: #e7ecf7 !important; }}
    [data-baseweb="menu"] {{ background-color: #16233f !important; }}
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] [role="option"] {{
        background-color: #16233f !important;
        color:            #e7ecf7 !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {{
        background-color: #1e3a5f !important;
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
        ["🏠 Dashboard", "📈 Engine Monitoring", "🗄️ Sensor Data", "🔧 Maintenance", "⚠️ Risk Detection", "🔍 Anomaly Detection"],
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
    "🔍 Anomaly Detection": ("Sensor Anomaly Detection", "Identify abnormal sensor readings · ML-based z-score & trend analysis"),
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

# ==========================================================================
# PAGE 6 — SENSOR ANOMALY DETECTION
# ==========================================================================
elif page == "🔍 Anomaly Detection":

    # -----------------------------------------------------------------------
    # HELPER: rolling z-score anomaly detection
    # -----------------------------------------------------------------------
    def rolling_anomaly(series: pd.Series, window: int = 20):
        """
        Compute a rolling z-score for each point using a backward window.
        Returns a Series of z-scores (clipped to ±4).
        """
        roll_mean = series.rolling(window, min_periods=5).mean()
        roll_std  = series.rolling(window, min_periods=5).std().fillna(1e-9)
        z = (series - roll_mean) / roll_std.clip(lower=1e-9)
        return z.clip(-4, 4)

    def classify_sensor_anomaly(z: float) -> tuple:
        """Return (status, color, icon) based on z-score magnitude."""
        if abs(z) > 2.5:
            return "Critical", "#dc2626", "🔴"
        elif abs(z) > 1.5:
            return "Warning",  "#d97706", "🟡"
        else:
            return "Normal",   "#16a34a", "🟢"

    def compute_full_anomaly_report(eng_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        For every sensor, compute:
        - historical mean / std (global baseline)
        - latest value & its global z-score
        - rolling z-score at the latest point
        - trend direction (rising / falling / stable over last 10 cycles)
        - anomaly status (Critical / Warning / Normal)
        """
        sensor_cols = [c for c in eng_df.columns if c.startswith("sensor_")]
        rows = []
        latest_row = eng_df.iloc[-1]
        recent_10  = eng_df.tail(10)
        for col in sensor_cols:
            series  = eng_df[col]
            mean_v  = series.mean()
            std_v   = series.std()
            current = float(latest_row[col])
            # Global z-score
            g_z     = (current - mean_v) / max(std_v, 1e-9)
            # Rolling z-score (more sensitive to sudden spikes)
            roll_z_series = rolling_anomaly(series, window)
            r_z     = float(roll_z_series.iloc[-1]) if not roll_z_series.empty else g_z
            # Trend: slope of last 10 cycles normalised to % per cycle
            if len(recent_10) >= 3:
                x = np.arange(len(recent_10))
                y = recent_10[col].values
                slope = np.polyfit(x, y, 1)[0]
                trend_pct = slope / max(abs(mean_v), 1e-6) * 100
                if trend_pct > 0.05:
                    trend = "↑ Rising"
                elif trend_pct < -0.05:
                    trend = "↓ Falling"
                else:
                    trend = "→ Stable"
            else:
                trend, trend_pct = "→ Stable", 0.0
            # Use the larger of global z and rolling z for status classification
            worst_z = g_z if abs(g_z) >= abs(r_z) else r_z
            status, color, icon = classify_sensor_anomaly(worst_z)
            dev_pct = (current - mean_v) / max(abs(mean_v), 1e-9) * 100
            rows.append({
                "sensor":      col,
                "label":       SENSOR_LABELS.get(col, col),
                "category":    next((c for c, sl in CATEGORY_MAP.items() if col in sl), "Other"),
                "current":     round(current, 3),
                "mean":        round(mean_v, 3),
                "std":         round(std_v, 3),
                "global_z":    round(g_z, 2),
                "rolling_z":   round(r_z, 2),
                "worst_z":     round(worst_z, 2),
                "deviation_pct": round(dev_pct, 2),
                "trend":       trend,
                "trend_pct":   round(trend_pct, 4),
                "status":      status,
                "color":       color,
                "icon":        icon,
            })
        return pd.DataFrame(rows)

    def engine_condition_verdict(report_df: pd.DataFrame, rul_v: float) -> tuple:
        """
        Combine sensor anomaly counts with RUL to give an overall engine condition.
        Returns (verdict_str, color, icon, description).
        """
        n_crit = (report_df["status"] == "Critical").sum()
        n_warn = (report_df["status"] == "Warning").sum()
        if rul_v < 30 or n_crit >= 3:
            return ("CRITICAL ENGINE CONDITION", "#dc2626", "🔴",
                    f"{n_crit} critical sensor(s) + RUL ≈ {rul_v:.0f} cycles — immediate attention required.")
        elif rul_v < 80 or n_crit >= 1 or n_warn >= 4:
            return ("DEGRADED ENGINE CONDITION", "#d97706", "🟡",
                    f"{n_crit} critical / {n_warn} warning sensor(s) + RUL ≈ {rul_v:.0f} cycles — plan maintenance.")
        else:
            return ("HEALTHY ENGINE CONDITION", "#16a34a", "🟢",
                    f"Sensors reading normally · RUL ≈ {rul_v:.0f} cycles — continue standard operations.")

    # -----------------------------------------------------------------------
    # COMPUTE ANOMALY REPORT
    # -----------------------------------------------------------------------
    with st.spinner("Running anomaly detection…"):
        anom_report = compute_full_anomaly_report(engine_df)

    verdict_str, verdict_color, verdict_icon, verdict_desc = engine_condition_verdict(anom_report, rul_pred)

    n_crit_a = int((anom_report["status"] == "Critical").sum())
    n_warn_a = int((anom_report["status"] == "Warning").sum())
    n_norm_a = int((anom_report["status"] == "Normal").sum())

    # -----------------------------------------------------------------------
    # SECTION 1 — ENGINE CONDITION VERDICT BANNER
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="
            background:{'#fef2f2' if verdict_color=='#dc2626' else '#fffbeb' if verdict_color=='#d97706' else '#f0fdf4'};
            border: 2.5px solid {verdict_color};
            border-radius: 18px; padding: 26px 36px; margin-bottom: 22px;
            display: flex; align-items: center; gap: 28px;
        ">
            <div style="font-size:60px;line-height:1;">{verdict_icon}</div>
            <div style="flex:1;">
                <div style="font-size:12px;font-weight:600;color:#64748b;letter-spacing:.08em;text-transform:uppercase;">
                    Engine #{int(engine_id):02d} &nbsp;·&nbsp; {fd_choice} &nbsp;·&nbsp; Cycle {cycle_now}
                </div>
                <div style="font-size:32px;font-weight:900;color:{verdict_color};margin:4px 0 6px;">
                    {verdict_str}
                </div>
                <div style="font-size:14px;color:#374151;">{verdict_desc}</div>
            </div>
            <div style="text-align:right;min-width:130px;">
                <div style="font-size:11px;color:#64748b;font-weight:600;">ANOMALOUS SENSORS</div>
                <div style="font-size:44px;font-weight:900;color:{verdict_color};line-height:1;">
                    {n_crit_a + n_warn_a}
                    <span style="font-size:20px;">/ {len(anom_report)}</span>
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;">total sensors checked</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # SECTION 2 — KPI ROW
    # -----------------------------------------------------------------------
    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🔴 Critical Sensors</div>
            <div class="kpi-value" style="color:#dc2626;">{n_crit_a}</div>
            <div class="kpi-sub">|Z| &gt; 2.5 — abnormal</div>
            </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🟡 Warning Sensors</div>
            <div class="kpi-value" style="color:#d97706;">{n_warn_a}</div>
            <div class="kpi-sub">|Z| 1.5 – 2.5 — elevated</div>
            </div>""", unsafe_allow_html=True)
    with a3:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🟢 Normal Sensors</div>
            <div class="kpi-value" style="color:#16a34a;">{n_norm_a}</div>
            <div class="kpi-sub">Within expected range</div>
            </div>""", unsafe_allow_html=True)
    with a4:
        rising_count = int((anom_report["trend"] == "↑ Rising").sum())
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">↑ Rising Trends</div>
            <div class="kpi-value" style="color:#7c3aed;">{rising_count}</div>
            <div class="kpi-sub">Sensors trending upward</div>
            </div>""", unsafe_allow_html=True)
    with a5:
        worst_sensor_row = anom_report.loc[anom_report["worst_z"].abs().idxmax()]
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⚡ Worst Anomaly</div>
            <div class="kpi-value" style="font-size:18px;color:{worst_sensor_row['color']};">
                {worst_sensor_row['icon']} {worst_sensor_row['label'][:18]}
            </div>
            <div class="kpi-sub">Z = {worst_sensor_row['worst_z']:+.2f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SECTION 3 — ANOMALY STATUS GRID  (sensor cards)
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="panel-card"><div class="panel-title">🔬 Sensor-by-Sensor Anomaly Status</div>'
        '<div class="panel-sub">Current reading vs. historical baseline. Colour-coded by anomaly severity.</div></div>',
        unsafe_allow_html=True,
    )

    # Sort: critical first, then warning, then normal; within group by |z|
    sorted_report = anom_report.sort_values(
        ["status", "worst_z"],
        key=lambda col: col.map({"Critical": 0, "Warning": 1, "Normal": 2}) if col.name == "status" else col.abs(),
        ascending=[True, False]
    ).reset_index(drop=True)

    COLS_PER_ROW = 4
    rows_list = [sorted_report.iloc[i:i+COLS_PER_ROW] for i in range(0, len(sorted_report), COLS_PER_ROW)]
    for row_chunk in rows_list:
        grid_cols = st.columns(COLS_PER_ROW)
        for col_idx, (_, sensor_row) in enumerate(row_chunk.iterrows()):
            bg_map  = {"Critical": "#fff1f2", "Warning": "#fffbeb", "Normal": "#f0fdf4"}
            brd_map = {"Critical": "#fca5a5", "Warning": "#fcd34d", "Normal": "#86efac"}
            bg  = bg_map[sensor_row["status"]]
            brd = brd_map[sensor_row["status"]]
            with grid_cols[col_idx]:
                st.markdown(
                    f"""<div style="background:{bg};border:1.5px solid {brd};border-radius:12px;
                        padding:14px 16px;margin-bottom:12px;">
                        <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">
                            {sensor_row['category']}
                        </div>
                        <div style="font-size:14px;font-weight:800;color:#0f172a;margin:4px 0 2px;line-height:1.2;">
                            {sensor_row['label']}
                        </div>
                        <div style="font-size:11px;color:#64748b;margin-bottom:8px;">{sensor_row['sensor']}</div>
                        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
                            <span style="background:{sensor_row['color']};color:white;border-radius:8px;
                                padding:2px 8px;font-size:11px;font-weight:700;">
                                {sensor_row['icon']} {sensor_row['status']}
                            </span>
                            <span style="font-size:11px;color:#64748b;">{sensor_row['trend']}</span>
                        </div>
                        <div style="font-size:13px;color:#0f172a;">
                            <b>Current:</b> {sensor_row['current']:,.3g}
                            &nbsp;·&nbsp; <b>Mean:</b> {sensor_row['mean']:,.3g}
                        </div>
                        <div style="font-size:12px;color:{sensor_row['color']};font-weight:700;margin-top:4px;">
                            Z-score: {sensor_row['worst_z']:+.2f}
                            &nbsp;·&nbsp; Dev: {sensor_row['deviation_pct']:+.1f}%
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # SECTION 4 — ROLLING Z-SCORE CHART  +  CATEGORY HEALTH RADAR
    # -----------------------------------------------------------------------
    rc1, rc2 = st.columns([2, 1])

    with rc1:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">📈 Rolling Anomaly Score Over Time</div>'
            '<div class="panel-sub">Rolling 20-cycle window z-score for the most anomalous sensors. '
            'Values outside ±1.5 (dashed lines) indicate elevated deviation.</div></div>',
            unsafe_allow_html=True,
        )
        # Pick the top 5 most anomalous sensors
        top_sensors = anom_report.nlargest(5, "worst_z", keep="all") if len(anom_report) >= 5 else anom_report
        # Also include sensors with worst (most negative) z
        bottom_sensors = anom_report.nsmallest(3, "worst_z", keep="all")
        highlight_sensors = pd.concat([top_sensors, bottom_sensors]).drop_duplicates("sensor").head(6)

        PALETTE = ["#2563eb", "#dc2626", "#d97706", "#7c3aed", "#16a34a", "#0d9488"]
        roll_fig = go.Figure()
        for i, (_, sr) in enumerate(highlight_sensors.iterrows()):
            col_key = sr["sensor"]
            rz = rolling_anomaly(engine_df[col_key], window=20)
            roll_fig.add_trace(go.Scatter(
                x=engine_df["cycle"], y=rz,
                name=sr["label"],
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                hovertemplate=f"{sr['label']}: %{{y:.2f}} z<br>Cycle: %{{x}}<extra></extra>",
            ))

        # Warning / Critical threshold lines
        for threshold, dash_, color_, label_ in [
            (1.5,  "dash",  "#d97706", "Warning (+1.5)"),
            (-1.5, "dash",  "#d97706", "Warning (−1.5)"),
            (2.5,  "dot",   "#dc2626", "Critical (+2.5)"),
            (-2.5, "dot",   "#dc2626", "Critical (−2.5)"),
        ]:
            roll_fig.add_hline(
                y=threshold, line_dash=dash_, line_color=color_,
                annotation_text=label_, annotation_font_color=color_,
                annotation_font_size=10,
            )

        roll_fig.update_layout(**styled_layout(
            height=360, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Engine Cycle",
            yaxis_title="Rolling Z-Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(roll_fig, use_container_width=True, key="anom_roll_chart")

    with rc2:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">🎯 Category Health Radar</div>'
            '<div class="panel-sub">Average |Z-score| per sensor category. Higher = more anomalous.</div></div>',
            unsafe_allow_html=True,
        )
        cat_z = anom_report.groupby("category")["worst_z"].apply(lambda s: s.abs().mean()).reset_index()
        cat_z.columns = ["category", "mean_abs_z"]
        categories = cat_z["category"].tolist()
        values     = cat_z["mean_abs_z"].tolist()

        radar2 = go.Figure()
        radar2.add_trace(go.Scatterpolar(
            r=[1.5] * len(categories) + [1.5],
            theta=categories + [categories[0]],
            name="Warning Threshold",
            line=dict(color="#d97706", dash="dash", width=1.5),
            fill=None,
        ))
        radar2.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name="Anomaly Score",
            line=dict(color="#2563eb", width=2.5),
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
        ))
        _dm_r = st.session_state.get("dark_mode", False)
        _r_txt = "#e2e8f0" if _dm_r else TEXT_DARK
        _r_bg  = "#1e293b" if _dm_r else "white"
        radar2.update_layout(
            height=340, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=_r_bg, font=dict(color=_r_txt),
            polar=dict(
                bgcolor=_r_bg,
                radialaxis=dict(visible=True, range=[0, max(3.0, max(values) * 1.1)],
                                color=_r_txt, gridcolor="#334155" if _dm_r else "#e5e9f2"),
                angularaxis=dict(color=_r_txt),
            ),
            showlegend=True,
            legend=dict(font=dict(color=_r_txt), orientation="h", y=-0.15),
        )
        st.plotly_chart(radar2, use_container_width=True, key="anom_radar_chart")

    # -----------------------------------------------------------------------
    # SECTION 5 — DETAILED ANOMALY TABLE
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="panel-card"><div class="panel-title">📋 Full Anomaly Analysis Table</div>'
        '<div class="panel-sub">All 21 sensors — sorted by anomaly severity. '
        'Rolling Z uses a 20-cycle sliding window; Global Z uses the entire engine history.</div></div>',
        unsafe_allow_html=True,
    )

    table_df = anom_report.copy()
    table_df["Status"] = table_df.apply(lambda r: f"{r['icon']} {r['status']}", axis=1)
    table_df["Global Z"] = table_df["global_z"].apply(lambda x: f"{x:+.2f}")
    table_df["Rolling Z"] = table_df["rolling_z"].apply(lambda x: f"{x:+.2f}")
    table_df["Deviation"] = table_df["deviation_pct"].apply(lambda x: f"{x:+.1f}%")
    table_df["Trend"]     = table_df["trend"]

    show_tbl = table_df.sort_values("worst_z", key=lambda s: s.abs(), ascending=False)[[
        "label", "category", "current", "mean", "std",
        "Global Z", "Rolling Z", "Deviation", "Trend", "Status"
    ]].rename(columns={
        "label": "Sensor", "category": "Category",
        "current": "Current", "mean": "Baseline Mean", "std": "Std Dev",
    })

    st.dataframe(show_tbl, use_container_width=True, height=380, hide_index=True)

    # -----------------------------------------------------------------------
    # SECTION 6 — SENSOR DEEP DIVE
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Sensor Deep Dive")
    st.markdown("Select any sensor to inspect its full history, rolling z-score, and anomaly timeline.")

    sensor_opts = [c for c in engine_df.columns if c.startswith("sensor_")]
    # Default to the most anomalous sensor
    default_idx = int(anom_report["worst_z"].abs().idxmax())
    default_sensor = anom_report.loc[default_idx, "sensor"]
    dd_sensor = st.selectbox(
        "Choose a sensor",
        sensor_opts,
        index=sensor_opts.index(default_sensor) if default_sensor in sensor_opts else 0,
        format_func=lambda c: f"{c} — {SENSOR_LABELS.get(c, c)}",
        key="anom_dd_sensor",
    )

    dd_row    = anom_report[anom_report["sensor"] == dd_sensor].iloc[0]
    dd_series = engine_df[dd_sensor]
    dd_rz     = rolling_anomaly(dd_series, window=20)

    dd_col1, dd_col2, dd_col3, dd_col4 = st.columns(4)
    for col_, lbl_, val_, sfx_ in [
        (dd_col1, "Current Value",    dd_row["current"],      ""),
        (dd_col2, "Baseline Mean",    dd_row["mean"],         ""),
        (dd_col3, "Global Z-Score",   dd_row["global_z"],     ""),
        (dd_col4, "Rolling Z-Score",  dd_row["rolling_z"],    ""),
    ]:
        with col_:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-label">{lbl_}</div>
                <div class="kpi-value" style="font-size:22px;">{val_:+,.3g}{sfx_}</div>
                <div class="kpi-sub">{dd_row['trend']} &nbsp;·&nbsp;
                    <b style="color:{dd_row['color']};">{dd_row['icon']} {dd_row['status']}</b>
                </div>
                </div>""", unsafe_allow_html=True)

    # Two sub-charts side by side
    dd_c1, dd_c2 = st.columns(2)

    with dd_c1:
        st.markdown(
            f'<div class="panel-card"><div class="panel-title">📈 {SENSOR_LABELS.get(dd_sensor, dd_sensor)} — Raw Reading</div></div>',
            unsafe_allow_html=True)
        raw_fig = go.Figure()
        raw_fig.add_trace(go.Scatter(
            x=engine_df["cycle"], y=dd_series, mode="lines",
            line=dict(color="#2563eb", width=2), fill="tozeroy",
            fillcolor="rgba(37,99,235,0.07)",
            name="Sensor Reading",
        ))
        # Shade the last 20 cycles used for rolling window
        raw_fig.add_vrect(
            x0=engine_df["cycle"].iloc[-20] if len(engine_df) >= 20 else engine_df["cycle"].iloc[0],
            x1=engine_df["cycle"].iloc[-1],
            fillcolor="rgba(220,38,38,0.07)", line_width=0,
            annotation_text="Analysis window", annotation_position="top left",
            annotation_font_size=10,
        )
        # Mark latest point
        raw_fig.add_trace(go.Scatter(
            x=[engine_df["cycle"].iloc[-1]], y=[dd_series.iloc[-1]],
            mode="markers", marker=dict(color=dd_row["color"], size=10),
            name="Latest reading",
        ))
        raw_fig.update_layout(**styled_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Engine Cycle",
            yaxis_title=SENSOR_LABELS.get(dd_sensor, dd_sensor),
        ))
        st.plotly_chart(raw_fig, use_container_width=True, key="anom_raw_chart")

    with dd_c2:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">📉 Rolling Z-Score Timeline</div>'
            '<div class="panel-sub">Shaded zones: yellow = Warning (±1.5), red = Critical (±2.5).</div></div>',
            unsafe_allow_html=True)
        rz_fig = go.Figure()
        # Shaded bands
        rz_fig.add_hrect(y0=-1.5, y1=1.5,  fillcolor="rgba(22,163,74,0.08)",  line_width=0)
        rz_fig.add_hrect(y0=1.5,  y1=2.5,  fillcolor="rgba(217,119,6,0.12)",  line_width=0)
        rz_fig.add_hrect(y0=-2.5, y1=-1.5, fillcolor="rgba(217,119,6,0.12)",  line_width=0)
        rz_fig.add_hrect(y0=2.5,  y1=4.0,  fillcolor="rgba(220,38,38,0.12)",  line_width=0)
        rz_fig.add_hrect(y0=-4.0, y1=-2.5, fillcolor="rgba(220,38,38,0.12)",  line_width=0)

        rz_fig.add_trace(go.Scatter(
            x=engine_df["cycle"], y=dd_rz, mode="lines",
            line=dict(color=dd_row["color"], width=2),
            name="Rolling Z-Score",
            hovertemplate="Cycle: %{x}<br>Z-Score: %{y:.2f}<extra></extra>",
        ))
        rz_fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
        rz_fig.add_hline(y=1.5,  line_dash="dash", line_color="#d97706", line_width=1)
        rz_fig.add_hline(y=-1.5, line_dash="dash", line_color="#d97706", line_width=1)
        rz_fig.add_hline(y=2.5,  line_dash="dot",  line_color="#dc2626", line_width=1)
        rz_fig.add_hline(y=-2.5, line_dash="dot",  line_color="#dc2626", line_width=1)
        rz_fig.update_layout(**styled_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Engine Cycle",
            yaxis_title="Z-Score",
            yaxis=dict(range=[-4.2, 4.2]),
        ))
        st.plotly_chart(rz_fig, use_container_width=True, key="anom_rz_chart")

    # -----------------------------------------------------------------------
    # SECTION 7 — ANOMALY TIMELINE HEATMAP (all sensors across all cycles)
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="panel-card"><div class="panel-title">🌡️ Full Anomaly Timeline Heatmap</div>'
        '<div class="panel-sub">Rolling z-score for every sensor across all cycles. '
        'Red = anomalously high, Green = anomalously low. Last 50 cycles shown.</div></div>',
        unsafe_allow_html=True,
    )
    all_sensor_cols = [c for c in engine_df.columns if c.startswith("sensor_")]
    timeline_df = engine_df.tail(50)
    heat_z_matrix = []
    heat_y_labels = []
    for col_ in all_sensor_cols:
        rz_full = rolling_anomaly(engine_df[col_], window=20)
        rz_win  = rz_full.iloc[-50:]
        heat_z_matrix.append(rz_win.values)
        heat_y_labels.append(SENSOR_LABELS.get(col_, col_))
    heat_z_matrix = np.array(heat_z_matrix)

    _dm_h2 = st.session_state.get("dark_mode", False)
    _h2_txt = "#e2e8f0" if _dm_h2 else TEXT_DARK
    _h2_bg  = "#1e293b" if _dm_h2 else "white"

    tl_fig = go.Figure(go.Heatmap(
        z=heat_z_matrix,
        x=timeline_df["cycle"].tolist(),
        y=heat_y_labels,
        colorscale=[
            [0.0,  "#16a34a"],
            [0.3,  "#86efac"],
            [0.5,  "#fef9c3"],
            [0.7,  "#fbbf24"],
            [1.0,  "#dc2626"],
        ],
        zmid=0, zmin=-4, zmax=4,
        colorbar=dict(
            title=dict(text="Z-score", font=dict(color=_h2_txt)),
            tickfont=dict(color=_h2_txt),
        ),
        hovertemplate="Sensor: %{y}<br>Cycle: %{x}<br>Rolling Z: %{z:.2f}<extra></extra>",
    ))
    tl_fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=_h2_bg, plot_bgcolor=_h2_bg,
        font=dict(color=_h2_txt),
        xaxis=dict(title="Engine Cycle", color=_h2_txt,
                   gridcolor="#334155" if _dm_h2 else "#e5e9f2"),
        yaxis=dict(title="Sensor", color=_h2_txt, autorange="reversed",
                   gridcolor="#334155" if _dm_h2 else "#e5e9f2"),
    )
    st.plotly_chart(tl_fig, use_container_width=True, key="anom_timeline_heatmap")

    # -----------------------------------------------------------------------
    # SECTION 8 — MAINTENANCE RECOMMENDATION
    # -----------------------------------------------------------------------
    st.markdown("### 🛠️ Recommended Actions")
    crit_sensors = anom_report[anom_report["status"] == "Critical"]["label"].tolist()
    warn_sensors = anom_report[anom_report["status"] == "Warning"]["label"].tolist()

    if verdict_color == "#dc2626":
        action_bg, action_border, action_text = "#fef2f2", "#dc2626", "#7f1d1d"
        action_icon  = "🔴 CRITICAL — Immediate Maintenance Required"
        action_items = [
            f"<li>Ground Engine #{int(engine_id):02d} immediately if safety margins cannot be confirmed.</li>",
            f"<li>Critical anomalies detected in: <b>{', '.join(crit_sensors) if crit_sensors else 'multiple sensors'}</b>.</li>",
            "<li>Schedule a full borescope inspection before the next flight cycle.</li>",
            "<li>Activate spare-parts logistics and downtime planning NOW.</li>",
            f"<li>Cross-reference these sensor deviations with known failure signatures for {fd_choice} engines.</li>",
        ]
    elif verdict_color == "#d97706":
        action_bg, action_border, action_text = "#fffbeb", "#d97706", "#78350f"
        action_icon  = "🟡 WARNING — Schedule Maintenance Soon"
        action_items = [
            f"<li>Elevated anomalies in: <b>{', '.join((crit_sensors + warn_sensors)[:5]) if (crit_sensors + warn_sensors) else 'several sensors'}</b>.</li>",
            f"<li>Slot Engine #{int(engine_id):02d} into the next planned maintenance window.</li>",
            "<li>Increase sensor monitoring frequency — check readings every 5 cycles.</li>",
            "<li>No immediate grounding needed — continue standard operations with caution.</li>",
            "<li>Re-run anomaly detection after the next 10 operating cycles.</li>",
        ]
    else:
        action_bg, action_border, action_text = "#f0fdf4", "#16a34a", "#14532d"
        action_icon  = "🟢 NORMAL — No Immediate Action Required"
        action_items = [
            "<li>All sensor readings are within expected operating bounds.</li>",
            f"<li>Continue standard monitoring cadence for Engine #{int(engine_id):02d}.</li>",
            "<li>Re-run anomaly detection after the next scheduled cycle batch.</li>",
            "<li>No maintenance action required at this time.</li>",
        ]

    st.markdown(
        f"""<div style="background:{action_bg};border:2px solid {action_border};
            border-radius:14px;padding:22px 26px;margin-bottom:20px;">
            <div style="font-size:20px;font-weight:900;color:{action_border};">{action_icon}</div>
            <ul style="margin-top:12px;font-size:13.5px;color:{action_text};">
                {''.join(action_items)}
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )

st.caption(
    f"AeroGuard · Safer Skies, Smarter Engines · Dataset: {fd_choice} · "
    f"Model: {'loaded' if model is not None else 'not found — showing heuristic RUL'}"
)