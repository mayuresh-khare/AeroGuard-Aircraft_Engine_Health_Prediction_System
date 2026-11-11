import os
import glob
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


TEXT_DARK = "#10182c"


def styled_layout(**overrides):
    """Mission Control telemetry Plotly layout using a crisp soft whitish-grey canvas with deep-merged overrides."""
    _text = "#0f172a"
    _axis_text = "#1e293b"
    _pbg = "#eef2f6"
    _paper_bg = "#e5ebf3"
    _grid = "#cbd5e1"
    _axis_base = dict(
        color=_axis_text,
        gridcolor=_grid,
        gridwidth=1,
        griddash="dot",
        zerolinecolor="#94a3b8",
        tickfont=dict(color=_axis_text),
        title_font=dict(color=_text, size=11.5),
    )
    layout = {
        "plot_bgcolor": _pbg,
        "paper_bgcolor": _paper_bg,
        "font": dict(family="'Consolas', 'JetBrains Mono', 'Fira Code', monospace", color=_text, size=11),
        "xaxis": dict(_axis_base),
        "yaxis": dict(_axis_base),
        "legend": dict(font=dict(color=_text, size=11)),
        "hoverlabel": dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font=dict(color=_text)),
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            merged = dict(layout[k])
            merged.update(v)
            layout[k] = merged
        elif k in ("yaxis2", "yaxis3") and isinstance(v, dict):
            merged = dict(_axis_base)
            merged.update(v)
            layout[k] = merged
        else:
            layout[k] = v
    return layout


def render_turbofan_schematic_svg(hpc_temp: float, lpc_temp: float, pressure: float, rpm: float, cycle: int, health_label: str, health_color: str) -> str:
    """Generates an authentic aerospace-grade SVG cross-section of a high-bypass turbofan engine with real-time station telemetry."""
    _dm = st.session_state.get("dark_mode", True)
    bg_c = "#070d18" if _dm else "#f8fafc"
    stroke_c = "#38bdf8" if _dm else "#0284c7"
    txt_c = "#f1f5f9" if _dm else "#0f172a"
    sub_txt = "#94a3b8" if _dm else "#64748b"
    card_bg = "rgba(11, 20, 38, 0.9)" if _dm else "rgba(255, 255, 255, 0.95)"
    grid_c = "rgba(56, 189, 248, 0.08)" if _dm else "rgba(2, 132, 199, 0.08)"

    svg = f"""<div style="background:{bg_c};border:1px solid {'#1e3a5f' if _dm else '#cbd5e1'};border-radius:10px;padding:12px;margin:8px 0;box-shadow:inset 0 0 20px rgba(0,0,0,0.3);">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-family:'Consolas','JetBrains Mono',monospace;font-size:10px;">
    <span style="color:#38bdf8;font-weight:700;letter-spacing:0.5px;">// TURBOFAN CROSS-SECTION (STATIONS 1.0 - 8.0)</span>
    <span style="background:{health_color}22;border:1px solid {health_color};color:{health_color};padding:2px 7px;border-radius:4px;font-weight:800;">{health_label.upper()}</span>
</div>
<svg viewBox="0 0 680 230" width="100%" height="210" xmlns="http://www.w3.org/2000/svg" style="display:block;">
  <defs>
    <pattern id="reticleGrid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="{grid_c}" stroke-width="1"/>
    </pattern>
    <linearGradient id="nacelleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e3a5f"/>
      <stop offset="100%" stop-color="#091426"/>
    </linearGradient>
    <linearGradient id="flameGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#f97316"/>
      <stop offset="50%" stop-color="#ef4444"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
    <linearGradient id="coreGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="50%" stop-color="#334155"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
  </defs>

  <rect width="680" height="230" fill="url(#reticleGrid)" rx="6"/>

  <line x1="30" y1="115" x2="650" y2="115" stroke="{sub_txt}" stroke-width="1" stroke-dasharray="8,4" opacity="0.4"/>
  <text x="655" y="118" fill="{sub_txt}" font-size="8" font-family="monospace">CL</text>

  <path d="M 120,48 Q 280,38 520,62" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>
  <path d="M 120,182 Q 280,192 520,168" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>
  <text x="320" y="44" fill="#38bdf8" font-size="8.5" font-family="monospace" font-weight="bold">BYPASS DUCT STREAM (COLD AIRFLOW)</text>

  <path d="M 45,28 Q 150,18 360,28 L 570,55 L 550,70 L 360,42 Q 150,34 55,42 Z" fill="url(#nacelleGrad)" stroke="{stroke_c}" stroke-width="1.5"/>
  <path d="M 45,202 Q 150,212 360,202 L 570,175 L 550,160 L 360,188 Q 150,196 55,188 Z" fill="url(#nacelleGrad)" stroke="{stroke_c}" stroke-width="1.5"/>

  <path d="M 170,72 L 490,85 L 530,115 L 490,145 L 170,158 Z" fill="url(#coreGrad)" stroke="#475569" stroke-width="1.5"/>

  <path d="M 35,115 Q 65,100 105,100 L 105,130 Q 65,130 35,115 Z" fill="#0284c7" stroke="#38bdf8" stroke-width="1.5"/>

  <rect x="105" y="42" width="12" height="146" rx="2" fill="#0284c7" stroke="#38bdf8" stroke-width="1.5"/>
  <line x1="111" y1="42" x2="111" y2="188" stroke="#bae6fd" stroke-width="1"/>

  <polygon points="180,76 210,81 210,149 180,154" fill="#0369a1" stroke="#38bdf8" stroke-width="1.2"/>
  <line x1="195" y1="78" x2="195" y2="152" stroke="#7dd3fc" stroke-width="1" stroke-dasharray="2,2"/>

  <polygon points="225,83 270,90 270,140 225,147" fill="#0284c7" stroke="#38bdf8" stroke-width="1.2"/>
  <line x1="240" y1="85" x2="240" y2="145" stroke="#bae6fd" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="255" y1="88" x2="255" y2="142" stroke="#bae6fd" stroke-width="1" stroke-dasharray="2,2"/>

  <rect x="285" y="93" width="46" height="44" rx="4" fill="url(#flameGrad)" stroke="#fbbf24" stroke-width="1.5"/>
  <circle cx="308" cy="115" r="8" fill="#fef08a" opacity="0.8"/>

  <polygon points="345,91 370,86 370,144 345,139" fill="#dc2626" stroke="#f87171" stroke-width="1.2"/>

  <polygon points="385,85 425,78 425,152 385,145" fill="#991b1b" stroke="#f87171" stroke-width="1.2"/>
  <line x1="405" y1="82" x2="405" y2="148" stroke="#fca5a5" stroke-width="1" stroke-dasharray="2,2"/>

  <polygon points="440,95 550,115 440,135" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>

  <rect x="245" y="112" width="115" height="6" fill="#38bdf8" stroke="#0284c7" stroke-width="1"/>
  <rect x="105" y="113.5" width="310" height="3" fill="#f1f5f9" stroke="#64748b" stroke-width="0.8"/>

  <!-- Station 2.0 LPC -->
  <g transform="translate(155, 170)">
    <rect width="68" height="30" rx="4" fill="{card_bg}" stroke="#0284c7" stroke-width="1"/>
    <text x="5" y="12" fill="{sub_txt}" font-size="7" font-family="monospace">ST 2.0 LPC</text>
    <text x="5" y="24" fill="{txt_c}" font-size="10" font-family="monospace" font-weight="bold">{lpc_temp:,.1f}°C</text>
  </g>

  <!-- Station 3.0 HPC -->
  <g transform="translate(230, 170)">
    <rect width="70" height="30" rx="4" fill="{card_bg}" stroke="#38bdf8" stroke-width="1"/>
    <text x="5" y="12" fill="{sub_txt}" font-size="7" font-family="monospace">ST 3.0 HPC</text>
    <text x="5" y="24" fill="#38bdf8" font-size="10" font-family="monospace" font-weight="bold">{hpc_temp:,.1f}°C</text>
  </g>

  <!-- Station 4.0 Combustor -->
  <g transform="translate(308, 170)">
    <rect width="78" height="30" rx="4" fill="{card_bg}" stroke="#f59e0b" stroke-width="1"/>
    <text x="5" y="12" fill="{sub_txt}" font-size="7" font-family="monospace">ST 4.0 PRESS</text>
    <text x="5" y="24" fill="#fbbf24" font-size="10" font-family="monospace" font-weight="bold">{pressure:,.2f} psia</text>
  </g>

  <!-- Core Spool RPM -->
  <g transform="translate(395, 170)">
    <rect width="74" height="30" rx="4" fill="{card_bg}" stroke="#818cf8" stroke-width="1"/>
    <text x="5" y="12" fill="{sub_txt}" font-size="7" font-family="monospace">CORE SPEED</text>
    <text x="5" y="24" fill="#a5b4fc" font-size="10" font-family="monospace" font-weight="bold">{rpm:,.0f} RPM</text>
  </g>

  <!-- Fan / N1 Speed Indicator -->
  <g transform="translate(45, 170)">
    <rect width="60" height="30" rx="4" fill="{card_bg}" stroke="#38bdf8" stroke-width="1"/>
    <text x="5" y="12" fill="{sub_txt}" font-size="7" font-family="monospace">N1 FAN</text>
    <text x="5" y="24" fill="{txt_c}" font-size="10" font-family="monospace" font-weight="bold">ACTIVE</text>
  </g>
</svg>
</div>"""
    return svg


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
    st.session_state.dark_mode = True
_DM = st.session_state.dark_mode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gradient_boosting_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
IMPROVED_MODEL_PATH = os.path.join(BASE_DIR, "improved_rul_model.pkl")
IMPROVED_SCALER_PATH = os.path.join(BASE_DIR, "improved_rul_scaler.pkl")
IMPROVED_FEATS_PATH = os.path.join(BASE_DIR, "improved_model_features.json")
IMPROVED_Q10_PATH = os.path.join(BASE_DIR, "improved_rul_q10.pkl")
IMPROVED_Q90_PATH = os.path.join(BASE_DIR, "improved_rul_q90.pkl")
IMPROVED_QMETA_PATH = os.path.join(BASE_DIR, "improved_rul_quantiles_meta.json")
COMPARISON_METRICS_PATH = os.path.join(BASE_DIR, "model_comparison_metrics.csv")
FEATURE_IMPORTANCES_PATH = os.path.join(BASE_DIR, "feature_importances.csv")
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

    /* ===== Sidebar Expand Button (Modern Tactical Drawer Handle) ===== */
    [data-testid="stExpandSidebarButton"] {{
        position:   fixed        !important;
        top:        16px         !important;
        left:       0            !important;
        z-index:    999999       !important;
        transform:  none         !important;
        margin:     0            !important;
        padding:    0            !important;
        background: transparent  !important;
        box-shadow: none         !important;
        border:     none         !important;
    }}
    [data-testid="stExpandSidebarButton"] button {{
        display:         flex               !important;
        align-items:     center             !important;
        justify-content: center             !important;
        background:      #0b1730            !important;
        border:          1.5px solid #1e3a5f!important;
        border-left:     none               !important;
        border-radius:   0 8px 8px 0        !important;
        padding:         8px 12px 8px 10px  !important;
        box-shadow:      2px 2px 14px rgba(0,0,0,0.45) !important;
        cursor:          pointer            !important;
        transition:      all 0.2s ease      !important;
    }}
    [data-testid="stExpandSidebarButton"] button:hover {{
        background:   #172a52                             !important;
        border-color: #38bdf8                             !important;
        box-shadow:   0 0 16px rgba(56,189,248,0.45)     !important;
        transform:    translateX(3px)                     !important;
    }}
    [data-testid="stExpandSidebarButton"] button svg,
    [data-testid="stExpandSidebarButton"] button svg * {{
        fill:   #38bdf8 !important;
        stroke: #38bdf8 !important;
        color:  #38bdf8 !important;
        width:  18px    !important;
        height: 18px    !important;
        filter: drop-shadow(0 0 4px rgba(56,189,248,0.6)) !important;
    }}

    /* ===== Sidebar Collapse Button (Inside Sidebar Header) ===== */
    [data-testid="stSidebarCollapseButton"] button {{
        background:    #0f1c38 !important;
        border:        1px solid #1e3a5f !important;
        border-radius: 6px !important;
        padding:       5px 8px !important;
        transition:    all 0.2s ease !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{
        background:   #172a52 !important;
        border-color: #38bdf8 !important;
        box-shadow:   0 0 10px rgba(56,189,248,0.3) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapseButton"] button svg * {{
        fill:   #38bdf8 !important;
        stroke: #38bdf8 !important;
        color:  #38bdf8 !important;
    }}


    /* ===== App background ===== */
    .stApp {{ background-color: {_bg}; }}
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    section.main [data-testid="stWidgetLabel"],
    section.main [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stSliderTickBarMin"],
    [data-testid="stMain"] [data-testid="stSliderTickBarMax"],
    [data-testid="stMain"] [data-testid="stThumbValue"],
    section.main [data-testid="stSliderTickBarMin"],
    section.main [data-testid="stSliderTickBarMax"],
    section.main [data-testid="stThumbValue"] {{
        color: {'#9ca3af' if _DM else '#6b7280'} !important;
    }}

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


    /* ===== Radar Beacon Animation ===== */
    @keyframes pulseGlow {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
    .pulse-beacon {{
        width: 10px; height: 10px; background: #10b981; border-radius: 50%;
        display: inline-block; margin-right: 6px;
        animation: pulseGlow 2s infinite;
    }}

    /* ===== Tactical Sidebar Radio Buttons (Avionics Selector) ===== */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        background: #0f1c38 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin-bottom: 6px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        background: #172a52 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
        transform: translateX(3px) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.2) 0%, rgba(30, 58, 95, 0.45) 100%) !important;
        border: 1px solid #38bdf8 !important;
        border-left: 4px solid #38bdf8 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.25) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {{
        color: #38bdf8 !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px !important;
    }}

    /* Tactical Number Shortcut Badges [1]..[9] on right */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label::after {{
        font-family: 'Consolas', 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 800 !important;
        color: #64748b !important;
        background: #081226 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        margin-left: auto !important;
        letter-spacing: 0.5px !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.4) !important;
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(1)::after {{ content: "[1]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(2)::after {{ content: "[2]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(3)::after {{ content: "[3]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(4)::after {{ content: "[4]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(5)::after {{ content: "[5]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(6)::after {{ content: "[6]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(7)::after {{ content: "[7]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(8)::after {{ content: "[8]"; }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(9)::after {{ content: "[9]"; }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover::after {{
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.4) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked)::after {{
        color: #070d18 !important;
        background: #38bdf8 !important;
        border-color: #38bdf8 !important;
        font-weight: 900 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
    }}

    /* Keymap quick-access pill hover */
    #aeroguard-keymap-trigger:hover,
    .aeroguard-keymap-trigger:hover {{
        background: rgba(56, 189, 248, 0.28) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important;
        transform: scale(1.05) !important;
    }}
    #aeroguard-keymap-trigger:active,
    .aeroguard-keymap-trigger:active {{
        transform: scale(0.95) !important;
    }}

    /* Seamless hidden helper for keyboard listener — keeping iframe active */
    iframe[data-testid="stCustomComponentV1"] {{
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
        border: none !important;
    }}

    /* ===== Smooth colour transitions on the whole app ===== */
    .stApp, section[data-testid="stSidebar"],
    .kpi-card, .panel-card, .hero-banner {{
        transition: background-color 0.35s ease, color 0.35s ease,
                    border-color 0.35s ease !important;
    }}

    /* ===== Sidebar Horizontal Toggle Switch (Avionics Mode Slider) ===== */
    section[data-testid="stSidebar"] div[data-testid="stToggle"] {{
        background: #0f1c38 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        padding: 7px 12px !important;
        margin-bottom: 10px !important;
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stToggle"]:hover {{
        border-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stToggle"] label {{
        display: flex !important;
        flex-direction: row-reverse !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 100% !important;
        cursor: pointer !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stToggle"] label p {{
        font-family: 'Consolas', 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        color: #e2e8f0 !important;
        margin: 0 !important;
    }}

    /* ===== Mission Control HUD Header Console ===== */
    .hud-header {{
        background: linear-gradient(135deg, #091326 0%, #0d1e3d 50%, #132b56 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 20px 26px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }}
    .hero-banner {{
        background: linear-gradient(135deg, #091326 0%, #0d1e3d 50%, #132b56 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 20px 26px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }}
    .hero-title {{ font-size: 24px; font-weight: 900; margin: 0; font-family: 'Consolas', 'JetBrains Mono', monospace; letter-spacing: -0.5px; }}
    .hero-sub {{ opacity: 0.85; margin-top: 3px; font-size: 12.5px; font-family: 'Consolas', 'JetBrains Mono', monospace; }}

    /* ===== Cockpit Instrumentation Cards ===== */
    .kpi-card, .panel-card {{
        background: {_card} !important; color: {_txt} !important;
        border-radius: 10px; padding: 16px 18px;
        box-shadow: 0 2px 8px {_shadow}; border: 1px solid {_border};
        border-top: 3px solid #38bdf8 !important;
        margin-bottom: 16px;
    }}
    .kpi-card table, .panel-card table {{ color: {_txt} !important; width: 100%; }}
    .kpi-card td, .panel-card td, .kpi-card th, .panel-card th {{ color: {_txt} !important; }}
    .kpi-label {{
        color: {_lbl} !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Consolas', 'JetBrains Mono', monospace;
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 900;
        color: {_txt} !important;
        margin: 3px 0;
        font-family: 'Consolas', 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.5px;
    }}
    .kpi-sub {{
        color: {_sub_c} !important;
        font-size: 11.5px;
        font-family: 'Consolas', 'JetBrains Mono', monospace;
    }}
    .kpi-up {{ color: #16a34a !important; font-weight: 700; font-size: 12px; }}
    .kpi-down {{ color: #dc2626 !important; font-weight: 700; font-size: 12px; }}
    .panel-title {{
        font-size: 15px;
        font-weight: 800;
        color: {_txt} !important;
        font-family: 'Consolas', 'JetBrains Mono', monospace;
        letter-spacing: 0.2px;
    }}
    .panel-sub {{
        color: {_sub_c} !important;
        font-size: 11.5px;
        margin-bottom: 8px;
        font-family: 'Consolas', 'JetBrains Mono', monospace;
    }}
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
    data["RUL"] = np.maximum(5.0, n_cycles - cycles + 10)
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
            rul_series = None
            if "RUL" in df.columns:
                rul_series = df["RUL"].values
            elif df.shape[1] > len(COLS):
                rul_series = df.iloc[:, len(COLS)].values

            if df.shape[1] >= len(COLS):
                df = df.iloc[:, : len(COLS)]
                df.columns = COLS
            else:
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                rename = {c: n for c, n in zip(df.columns, COLS)}
                df = df.rename(columns=rename)
            if rul_series is not None:
                df["RUL"] = rul_series
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


@st.cache_resource(show_spinner=False)
def load_improved_model_and_scaler():
    imp_model, imp_scaler, imp_feats = None, None, None
    if joblib is not None:
        try:
            if os.path.exists(IMPROVED_MODEL_PATH):
                imp_model = joblib.load(IMPROVED_MODEL_PATH)
        except Exception:
            imp_model = None
        try:
            if os.path.exists(IMPROVED_SCALER_PATH):
                imp_scaler = joblib.load(IMPROVED_SCALER_PATH)
        except Exception:
            imp_scaler = None
        try:
            if os.path.exists(IMPROVED_FEATS_PATH):
                with open(IMPROVED_FEATS_PATH, "r") as f:
                    imp_feats = json.load(f)
        except Exception:
            imp_feats = None
    return imp_model, imp_scaler, imp_feats


@st.cache_resource(show_spinner=False)
def load_quantile_models():
    """Loads calibrated Q10/Q90 quantile regressors and conformal margin metadata."""
    q10_model, q90_model, q_meta = None, None, None
    if joblib is not None:
        try:
            if os.path.exists(IMPROVED_Q10_PATH):
                q10_model = joblib.load(IMPROVED_Q10_PATH)
        except Exception:
            q10_model = None
        try:
            if os.path.exists(IMPROVED_Q90_PATH):
                q90_model = joblib.load(IMPROVED_Q90_PATH)
        except Exception:
            q90_model = None
    try:
        if os.path.exists(IMPROVED_QMETA_PATH):
            with open(IMPROVED_QMETA_PATH, "r", encoding="utf-8") as f:
                q_meta = json.load(f)
    except Exception:
        q_meta = None
    return q10_model, q90_model, q_meta


def engineer_engine_features(eng: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Computes all 73 engineered degradation features for a given engine."""
    df = eng.copy().sort_values("cycle").reset_index(drop=True)
    max_c = df["cycle"].max()
    df["cycle_ratio"] = df["cycle"] / max(max_c, 1)

    key_sensors = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_17", "sensor_20", "sensor_21"
    ]
    for s in key_sensors:
        if s in df.columns:
            df[f"{s}_roll5_mean"] = df[s].rolling(5, min_periods=1).mean()
            df[f"{s}_roll5_std"] = df[s].rolling(5, min_periods=1).std().fillna(0)
            df[f"{s}_delta"] = df[s].diff().fillna(0)

    if "sensor_3" in df.columns and "sensor_2" in df.columns:
        df["temp_spread"] = df["sensor_3"] - df["sensor_2"]
    if "sensor_4" in df.columns and "sensor_3" in df.columns:
        df["turbine_temp_drop"] = df["sensor_4"] - df["sensor_3"]
    if "sensor_7" in df.columns and "sensor_11" in df.columns:
        df["pressure_ratio"] = df["sensor_7"] / (df["sensor_11"] + 1e-6)
    if "sensor_14" in df.columns and "sensor_9" in df.columns:
        df["core_speed_drift"] = df["sensor_14"] - df["sensor_9"]
    if "sensor_20" in df.columns and "sensor_21" in df.columns:
        df["coolant_bleed_sum"] = df["sensor_20"] + df["sensor_21"]

    for fn in feature_names:
        if fn not in df.columns:
            df[fn] = 0.0

    return df[feature_names].values.astype(float)


def predict_improved_rul_batch(imp_model, imp_scaler, imp_feats, frame: pd.DataFrame):
    """Vectorized prediction using the improved feature-engineered model."""
    if imp_model is not None and imp_scaler is not None and imp_feats is not None and not frame.empty:
        try:
            X = engineer_engine_features(frame, imp_feats)
            X_sc = imp_scaler.transform(X)
            preds = imp_model.predict(X_sc)
            preds = np.maximum(1.0, preds)
            return np.ravel(preds).astype(float), True
        except Exception:
            pass

    cycles = frame["cycle"].values if "cycle" in frame else np.arange(1, len(frame) + 1)
    heuristic = np.maximum(5.0, 220 - cycles * 0.9)
    return heuristic, False


def predict_baseline_rul_batch(model, scaler, frame: pd.DataFrame):
    """Vectorized baseline RUL prediction using the original Gradient Boosting model."""
    if model is not None and scaler is not None and not frame.empty:
        try:
            cycles = frame["cycle"].values.astype(float) if "cycle" in frame.columns else np.arange(1, len(frame) + 1, dtype=float)
            s1 = frame["sensor_1"].values.astype(float) if "sensor_1" in frame.columns else np.full(len(frame), 518.67)
            s2 = frame["sensor_2"].values.astype(float) if "sensor_2" in frame.columns else np.full(len(frame), 642.0)
            s3 = frame["sensor_3"].values.astype(float) if "sensor_3" in frame.columns else np.full(len(frame), 1585.0)
            temp_diff = s3 - s2
            s3_series = pd.Series(s3)
            hpc_roll = s3_series.rolling(5, min_periods=1).mean().values

            X_base = np.column_stack([cycles, s1, s2, s3, temp_diff, hpc_roll])
            X_sc = scaler.transform(X_base)
            preds = model.predict(X_sc)
            preds = np.maximum(1.0, preds)
            return np.ravel(preds).astype(float), True
        except Exception:
            pass

    cycles = frame["cycle"].values if "cycle" in frame else np.arange(1, len(frame) + 1)
    heuristic = np.maximum(5.0, 220 - cycles * 0.9)
    return heuristic, False


def predict_rul_batch(model, scaler, frame: pd.DataFrame):
    """Default RUL prediction across AeroGuard: uses improved model if loaded, else baseline."""
    imp_model, imp_scaler, imp_feats = load_improved_model_and_scaler()
    if imp_model is not None and imp_scaler is not None and imp_feats is not None:
        preds, used = predict_improved_rul_batch(imp_model, imp_scaler, imp_feats, frame)
        if used:
            return preds, True
    return predict_baseline_rul_batch(model, scaler, frame)


def predict_rul_row(model, scaler, row: pd.Series):
    preds, used = predict_rul_batch(model, scaler, pd.DataFrame([row]))
    return float(preds[0]), used


def predict_improved_rul_intervals_batch(
    imp_model,
    imp_scaler,
    imp_feats,
    q10_model,
    q90_model,
    q_meta,
    frame: pd.DataFrame,
):
    """Returns (point_preds, lower_q10, upper_q90, imp_used, quantiles_used) with monotonic non-crossing CQR bounds."""
    pt_preds, imp_used = predict_improved_rul_batch(imp_model, imp_scaler, imp_feats, frame)
    if (
        imp_used
        and q10_model is not None
        and q90_model is not None
        and imp_scaler is not None
        and imp_feats is not None
        and not frame.empty
    ):
        try:
            X = engineer_engine_features(frame, imp_feats)
            X_sc = imp_scaler.transform(X)
            cqr_pad = max(0.0, float(q_meta.get("cqr_margin", 0.0))) if isinstance(q_meta, dict) else 0.0
            raw_lo = np.ravel(q10_model.predict(X_sc)).astype(float) - cqr_pad
            raw_hi = np.ravel(q90_model.predict(X_sc)).astype(float) + cqr_pad
            lo = np.maximum(1.0, np.minimum(raw_lo, pt_preds))
            hi = np.maximum(pt_preds, np.maximum(raw_hi, lo + 2.0))
            return pt_preds, lo, hi, imp_used, True
        except Exception:
            pass

    half_w = np.maximum(6.0, 0.18 * pt_preds + 8.0)
    lo = np.maximum(1.0, pt_preds - half_w)
    hi = pt_preds + half_w
    return pt_preds, lo, hi, imp_used, False


def _format_xai_metric_val(val: float) -> str:
    """Formats sensor/engineered feature values cleanly across magnitudes."""
    if not np.isfinite(val):
        return "N/A"
    av = abs(val)
    if 0 < av < 1.0:
        return f"{val:.4f}"
    if av >= 100.0:
        return f"{val:,.1f}"
    return f"{val:.2f}"


def format_xai_feature_label(feat_name: str) -> str:
    """Maps an engineered model feature name to a concise aerospace label."""
    special_map = {
        "cycle": "Operating Cycle Count",
        "cycle_ratio": "Lifespan Ratio (cycle/max)",
        "op1": "Op Setting 1 (Altitude)",
        "op2": "Op Setting 2 (Mach No.)",
        "op3": "Op Setting 3 (Throttle TRA)",
        "temp_spread": "Temp Spread (HPC − LPC)",
        "turbine_temp_drop": "Turbine Temp Drop (LPT − HPC)",
        "pressure_ratio": "HPC Pressure Ratio (P30 / Ps30)",
        "core_speed_drift": "Core Speed Drift (Nc Corr − Phys)",
        "coolant_bleed_sum": "Total Coolant Bleed (HPT + LPT)",
    }
    if feat_name in special_map:
        return special_map[feat_name]
    if feat_name in SENSOR_LABELS:
        return SENSOR_LABELS[feat_name]
    for suffix, tag in (
        ("_roll5_mean", "5-Cyc Mean"),
        ("_roll5_std", "5-Cyc Volatility"),
        ("_delta", "Step Δ"),
    ):
        if feat_name.endswith(suffix):
            base = feat_name[: -len(suffix)]
            base_lbl = SENSOR_LABELS.get(base, base)
            return f"{base_lbl} ({tag})"
    return feat_name.replace("_", " ").title()


def compute_local_rul_attribution(imp_model, imp_scaler, imp_feats, frame: pd.DataFrame, row_idx: int = -1):
    """Computes exact additive local feature attributions (Saabas tree-path decomposition)
    for the improved GradientBoostingRegressor, decomposing prediction = base_value + sum(contribs).
    """
    if imp_model is None or imp_scaler is None or not imp_feats or frame is None or frame.empty:
        return None

    try:
        df_sorted = frame.copy().sort_values("cycle").reset_index(drop=True) if "cycle" in frame.columns else frame.reset_index(drop=True)
        n_rows = len(df_sorted)
        idx = n_rows - 1 if row_idx < 0 else min(max(int(row_idx), 0), n_rows - 1)

        X = engineer_engine_features(df_sorted, imp_feats)
        X_sc = imp_scaler.transform(X)
        x_sc = X_sc[idx : idx + 1]
        x_raw = X[idx]

        raw_pred = float(imp_model.predict(x_sc)[0])
        pred_rul = float(max(1.0, raw_pred))
        cycle_val = int(df_sorted.loc[idx, "cycle"]) if "cycle" in df_sorted.columns else idx + 1

        n_feats = len(imp_feats)
        contribs = np.zeros(n_feats, dtype=float)

        if hasattr(imp_model, "estimators_") and hasattr(imp_model, "learning_rate"):
            lr = float(imp_model.learning_rate)
            base_value = float(imp_model.init_.predict(x_sc)[0]) if hasattr(imp_model, "init_") else 0.0
            for est in np.asarray(imp_model.estimators_).ravel():
                tree = est.tree_
                base_value += lr * float(tree.value[0, 0, 0])
                node = 0
                while tree.children_left[node] != tree.children_right[node]:
                    feat_idx = int(tree.feature[node])
                    thresh = float(tree.threshold[node])
                    next_node = int(tree.children_left[node]) if x_sc[0, feat_idx] <= thresh else int(tree.children_right[node])
                    contribs[feat_idx] += lr * (float(tree.value[next_node, 0, 0]) - float(tree.value[node, 0, 0]))
                    node = next_node
        else:
            # Fallback single-feature scaled-mean perturbation
            base_x = np.zeros_like(x_sc)
            base_value = float(imp_model.predict(base_x)[0])
            for j in range(n_feats):
                pert = base_x.copy()
                pert[0, j] = x_sc[0, j]
                contribs[j] = float(imp_model.predict(pert)[0]) - base_value

        recon_pred = base_value + float(contribs.sum())
        recon_err = abs(recon_pred - raw_pred)

        rows = []
        for j, fn in enumerate(imp_feats):
            c_val = float(contribs[j])
            rows.append({
                "feature": fn,
                "label": format_xai_feature_label(fn),
                "raw_value": float(x_raw[j]),
                "raw_display": _format_xai_metric_val(float(x_raw[j])),
                "contribution": c_val,
                "abs_impact": abs(c_val),
                "direction": "Degrading (−RUL)" if c_val < 0 else ("Stabilizing (+RUL)" if c_val > 0 else "Neutral"),
            })

        attr_df = pd.DataFrame(rows).sort_values("abs_impact", ascending=False).reset_index(drop=True)
        top_degrading = attr_df[attr_df["contribution"] < -1e-6].sort_values("contribution", ascending=True).reset_index(drop=True)
        top_stabilizing = attr_df[attr_df["contribution"] > 1e-6].sort_values("contribution", ascending=False).reset_index(drop=True)

        return {
            "engine_cycle": cycle_val,
            "row_index": idx,
            "n_cycles": n_rows,
            "base_value": base_value,
            "raw_prediction": raw_pred,
            "predicted_rul": pred_rul,
            "net_shift": float(contribs.sum()),
            "recon_error": recon_err,
            "df": attr_df,
            "top_degrading": top_degrading,
            "top_stabilizing": top_stabilizing,
        }
    except Exception:
        return None


def build_xai_waterfall_figure(xai_res: dict, top_n: int = 10, dark_mode: bool = True) -> go.Figure:
    """Builds an interactive horizontal Plotly Waterfall decomposing fleet baseline -> predicted RUL."""
    attr_df = xai_res["df"]
    base_val = xai_res["base_value"]
    raw_pred = xai_res["raw_prediction"]
    cyc = xai_res["engine_cycle"]

    shown = attr_df.head(max(3, int(top_n))).copy()
    rest = attr_df.iloc[len(shown) :]
    other_sum = float(rest["contribution"].sum()) if not rest.empty else 0.0

    # Order active drivers cleanly by contribution ascending inside shown block so negative drivers pull left and positive pull right
    shown = shown.sort_values("contribution", ascending=True).reset_index(drop=True)

    y_labels = ["Fleet Baseline E[RUL]"]
    x_vals = [base_val]
    measures = ["absolute"]
    text_vals = [f"{base_val:.1f} cyc"]
    hover_texts = [f"<b>Fleet Baseline Prediction</b><br>Expected RUL: {base_val:.2f} cycles<extra></extra>"]

    for _, row in shown.iterrows():
        c = float(row["contribution"])
        y_labels.append(row["label"])
        x_vals.append(c)
        measures.append("relative")
        text_vals.append(f"{c:+.1f}")
        hover_texts.append(
            f"<b>{row['label']}</b><br>"
            f"Observed Value: {row['raw_display']}<br>"
            f"RUL Impact: <b>{c:+.2f} cycles</b><extra></extra>"
        )

    if not rest.empty and abs(other_sum) > 1e-4:
        y_labels.append(f"Other {len(rest)} Features")
        x_vals.append(other_sum)
        measures.append("relative")
        text_vals.append(f"{other_sum:+.1f}")
        hover_texts.append(
            f"<b>Remaining {len(rest)} Features</b><br>"
            f"Aggregated Net Impact: <b>{other_sum:+.2f} cycles</b><extra></extra>"
        )

    final_tag = f"Predicted RUL (Cyc {cyc})"
    y_labels.append(final_tag)
    x_vals.append(0.0)
    measures.append("total")
    text_vals.append(f"{raw_pred:.1f} cyc")
    hover_texts.append(f"<b>{final_tag}</b><br>Model Output: <b>{raw_pred:.2f} cycles</b><extra></extra>")

    pos_col = "#10b981"
    neg_col = "#ef4444"
    tot_col = "#2563eb"
    conn_col = "rgba(100,116,139,0.5)"
    txt_col = "#0f172a"

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=y_labels,
            x=x_vals,
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11, color=txt_col, family="JetBrains Mono, Consolas, monospace"),
            hovertemplate=hover_texts,
            connector=dict(line=dict(color=conn_col, width=1.2, dash="dot")),
            increasing=dict(marker=dict(color=pos_col, line=dict(color="#34d399", width=1))),
            decreasing=dict(marker=dict(color=neg_col, line=dict(color="#f87171", width=1))),
            totals=dict(marker=dict(color=tot_col, line=dict(color="#60a5fa", width=1.4))),
        )
    )

    chart_h = max(390, 29 * len(y_labels) + 95)
    fig.update_layout(
        **styled_layout(
            height=chart_h,
            margin=dict(l=12, r=58, t=18, b=36),
            xaxis_title="Remaining Useful Life Contribution (Cycles)",
            yaxis_title="",
            showlegend=False,
        )
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11.5))
    return fig


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
# ENGINE HEALTH SCORE HELPERS (TASK 4)
# --------------------------------------------------------------------------
def compute_engine_health_score(eng_df: pd.DataFrame, rul_prediction: float):
    """
    Computes a comprehensive physical & prognostic Engine Health Score (0–100) combining:
      1. RUL Prognostic Factor (40% weight): Remaining useful life
      2. Sensor Stability Factor (35% weight): Multi-sensor z-score deviations & critical penalties
      3. Operating Cycle Consumption (25% weight): Cumulative fatigue life

    4-Tier Classification:
      - Healthy (80 – 100): Optimal condition
      - Monitor (60 – 79): Mild drift, observational monitoring
      - Warning (40 – 59): Noticeable degradation, plan maintenance
      - Critical (0 – 39): Severe wear / urgent attention required
    """
    latest_row = eng_df.iloc[-1]
    cycle_now = int(latest_row["cycle"]) if "cycle" in latest_row else len(eng_df)

    # 1. RUL Score (0 to 100, 40% weight)
    rul_score = min(max(rul_prediction / 125.0 * 100.0, 0.0), 100.0)

    # 2. Sensor Stability Score (0 to 100, 35% weight)
    sensor_cols = [c for c in eng_df.columns if c.startswith("sensor_")]
    anom_rows = []
    z_scores = []
    crit_count = 0
    warn_count = 0

    for c in sensor_cols:
        s = eng_df[c]
        mu = s.mean()
        sig = s.std()
        curr = float(latest_row[c])
        z = (curr - mu) / max(sig, 1e-9)
        abs_z = abs(z)
        z_scores.append(abs_z)

        if abs_z > 2.5:
            st_sensor = "Critical"
            crit_count += 1
        elif abs_z > 1.5:
            st_sensor = "Warning"
            warn_count += 1
        else:
            st_sensor = "Normal"

        anom_rows.append({
            "sensor": c,
            "label": SENSOR_LABELS.get(c, c),
            "current": curr,
            "mean": mu,
            "std": sig,
            "z": z,
            "abs_z": abs_z,
            "status": st_sensor
        })

    mean_abs_z = float(np.mean(z_scores)) if z_scores else 0.0
    raw_sensor_score = max(0.0, 100.0 - (mean_abs_z / 3.0) * 100.0)
    sensor_score = min(max(raw_sensor_score - (crit_count * 4.0 + warn_count * 1.5), 0.0), 100.0)

    # 3. Operating Cycle Fatigue Score (0 to 100, 25% weight)
    total_life_est = cycle_now + max(rul_prediction, 1.0)
    cycle_score = min(max((rul_prediction / total_life_est) * 100.0, 0.0), 100.0)

    # Composite Health Score
    total_score = round(0.40 * rul_score + 0.35 * sensor_score + 0.25 * cycle_score, 1)

    # 4-Tier Classification Standard
    if total_score >= 80.0:
        tier = "Healthy"
        color = "#16a34a"
        bg_color = "#f0fdf4"
        icon = "🟢"
        desc = "Engine is operating in optimal health. All sensors and operating metrics indicate strong mechanical integrity within normal envelopes."
    elif total_score >= 60.0:
        tier = "Monitor"
        color = "#2563eb"
        bg_color = "#eff6ff"
        icon = "🔵"
        desc = "Engine exhibits early operational drift or mild sensor fluctuations. Continue flight operations with increased telemetry logging."
    elif total_score >= 40.0:
        tier = "Warning"
        color = "#d97706"
        bg_color = "#fffbeb"
        icon = "🟡"
        desc = "Noticeable sensor degradation and reduced remaining life. Recommend scheduling shop inspection within the next 15–20 operating cycles."
    else:
        tier = "Critical"
        color = "#dc2626"
        bg_color = "#fef2f2"
        icon = "🔴"
        desc = "Urgent attention required! Severe sensor anomalies or imminent end-of-life condition detected. Recommend immediate borescope inspection or grounding."

    def _subsystem_score(cols):
        zs = [r["abs_z"] for r in anom_rows if r["sensor"] in cols]
        if not zs:
            return 88.0
        return round(min(max(100.0 - float(np.mean(zs)) * 26.0, 0.0), 100.0), 1)

    subsystems = {
        "Compressor Subsystem": _subsystem_score(["sensor_2", "sensor_3", "sensor_7", "sensor_11"]),
        "Turbine & Hot Section": _subsystem_score(["sensor_4", "sensor_20", "sensor_21"]),
        "Rotor & Spool Dynamics": _subsystem_score(["sensor_8", "sensor_9", "sensor_13", "sensor_14"]),
        "Fuel & Combustion": _subsystem_score(["sensor_12", "sensor_16"]),
        "Bypass & Aerodynamics": _subsystem_score(["sensor_1", "sensor_5", "sensor_6", "sensor_15"]),
    }

    return {
        "total_score": total_score,
        "tier": tier,
        "color": color,
        "bg_color": bg_color,
        "icon": icon,
        "rul_score": round(rul_score, 1),
        "sensor_score": round(sensor_score, 1),
        "cycle_score": round(cycle_score, 1),
        "rul_contrib": round(0.40 * rul_score, 1),
        "sensor_contrib": round(0.35 * sensor_score, 1),
        "cycle_contrib": round(0.25 * cycle_score, 1),
        "crit_count": crit_count,
        "warn_count": warn_count,
        "description": desc,
        "anomaly_df": pd.DataFrame(anom_rows),
        "subsystems": subsystems,
    }


def compute_health_trajectory(eng_df: pd.DataFrame, rul_series: np.ndarray) -> np.ndarray:
    """Computes the cycle-by-cycle Engine Health Score trajectory across all operating cycles."""
    scores = []
    sensor_cols = [c for c in eng_df.columns if c.startswith("sensor_")]
    n = len(eng_df)
    for i in range(n):
        cyc = int(eng_df["cycle"].iloc[i]) if "cycle" in eng_df.columns else i + 1
        rul_val = float(rul_series[i])
        rul_s = min(max(rul_val / 125.0 * 100.0, 0.0), 100.0)

        sub_df = eng_df.iloc[: i + 1]
        z_vals = []
        for c in sensor_cols:
            s_all = eng_df[c]
            curr_v = float(sub_df[c].iloc[-1])
            z_vals.append(abs((curr_v - s_all.mean()) / max(s_all.std(), 1e-9)))
        m_z = float(np.mean(z_vals)) if z_vals else 0.0
        sens_s = max(0.0, 100.0 - (m_z / 3.0) * 100.0)

        cyc_s = min(max((rul_val / (cyc + max(rul_val, 1.0))) * 100.0, 0.0), 100.0)
        tot_s = round(0.40 * rul_s + 0.35 * sens_s + 0.25 * cyc_s, 1)
        scores.append(tot_s)
    return np.array(scores)


@st.cache_data(show_spinner=False)
def get_fleet_health_summary(_df: pd.DataFrame, fd_key: str) -> pd.DataFrame:
    """Compute overall health score and classification tier for all engines in the dataset."""
    rows = []
    imp_m, imp_sc, imp_f = load_improved_model_and_scaler()
    base_m, base_sc = load_model_and_scaler()

    for uid in sorted(_df["unit"].unique()):
        eng = _df[_df["unit"] == uid].reset_index(drop=True)
        if eng.empty:
            continue
        p_imp, used_imp = predict_improved_rul_batch(imp_m, imp_sc, imp_f, eng)
        if used_imp:
            r_val = float(p_imp[-1])
        else:
            p_base, _ = predict_baseline_rul_batch(base_m, base_sc, eng)
            r_val = float(p_base[-1])

        h_info = compute_engine_health_score(eng, r_val)
        c_now = int(eng["cycle"].iloc[-1]) if "cycle" in eng.columns else len(eng)
        rows.append({
            "unit": uid,
            "engine": f"Engine #{int(uid):02d}",
            "health_score": h_info["total_score"],
            "tier": h_info["tier"],
            "color": h_info["color"],
            "icon": h_info["icon"],
            "cycle": c_now,
            "rul": round(r_val, 1),
            "sensor_score": h_info["sensor_score"],
            "rul_score": h_info["rul_score"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# PREDICTIVE MAINTENANCE INTELLIGENCE (TASK 6)
# --------------------------------------------------------------------------
def evaluate_predictive_maintenance(rul_val: float, health_score: float, crit_count: int, warn_count: int, subsystems: dict = None):
    """
    Evaluates multi-factor decision-making logic across:
      1. RUL prognostic exhaustion (<=30 cyc immediate, <=75 cyc scheduled)
      2. Health score physical degradation (<40 critical, <60 warning)
      3. Abnormal sensor behavior (critical Z-scores >2.5, warning Z-scores >1.5)
      4. Subsystem mechanical integrity

    Returns comprehensive decision matrix:
      - recommendation: clear text directive
      - action_code: IMMEDIATE | SCHEDULED | MONITORING | NOMINAL
      - clearance: flight clearance verdict
      - urgency_index: 0 to 100% composite score
      - triggers: list of triggered decision rules
      - target_subsystem: primary affected subsystem
      - downtime: estimated maintenance turnaround hours
      - color & badge
    """
    triggers = []
    if rul_val <= 30:
        triggers.append(f"⏱️ RUL Critical Threshold: Only {rul_val:.0f} operating cycles remaining (<= 30 cyc).")
    elif rul_val <= 75:
        triggers.append(f"⏱️ RUL Degradation Warning: {rul_val:.0f} operating cycles remaining (<= 75 cyc).")
    elif rul_val <= 120:
        triggers.append(f"⏱️ RUL Observational Range: {rul_val:.0f} operating cycles remaining (<= 120 cyc).")

    if health_score < 40:
        triggers.append(f"🩺 Engine Health Critical: Index is {health_score:.1f}/100 (< 40 critical threshold).")
    elif health_score < 60:
        triggers.append(f"🩺 Engine Health Warning: Index is {health_score:.1f}/100 (< 60 warning threshold).")
    elif health_score < 80:
        triggers.append(f"🩺 Engine Health Observational: Index is {health_score:.1f}/100 (< 80 monitor threshold).")

    if crit_count > 0:
        triggers.append(f"🔬 Critical Sensor Anomalies: {crit_count} channel(s) exceed |Z| > 2.5σ extreme outlier limit.")
    if warn_count > 0:
        triggers.append(f"🔬 Warning Sensor Fluctuations: {warn_count} channel(s) exceed |Z| > 1.5σ elevated variance limit.")

    # Calculate Maintenance Urgency Index (0 to 100%)
    rul_urgency = min(100.0, max(0.0, (150.0 - rul_val) / 1.5))
    health_urgency = max(0.0, min(100.0, 100.0 - health_score))
    anomaly_urgency = min(100.0, crit_count * 30.0 + warn_count * 10.0)
    urgency_index = round(0.35 * rul_urgency + 0.35 * health_urgency + 0.30 * anomaly_urgency, 1)

    # Primary target subsystem
    target_sub = "Turbine & Hot Section"
    if subsystems:
        sorted_subs = sorted(subsystems.items(), key=lambda kv: kv[1])
        if sorted_subs:
            target_sub = sorted_subs[0][0]

    # Decision classification
    if rul_val <= 30 or crit_count >= 2 or health_score < 40 or urgency_index >= 75:
        action_code = "IMMEDIATE"
        badge = "🔴 IMMEDIATE ACTION REQUIRED"
        recommendation = "GROUND AIRCRAFT: Engine must be removed from active flight schedule. Immediate teardown and depot overhaul required."
        clearance = "🚫 DISPATCH PROHIBITED (AOG — Aircraft On Ground)"
        color = "#dc2626"
        bg_color = "#fef2f2"
        downtime = "36 – 72 Hours"
        work_scope = [
            f"Perform full borescope inspection of {target_sub} and high-pressure turbine blades.",
            "Remove engine from wing pylon and transport to overhaul test-cell.",
            "Inspect magnetic chip detectors (MCD) for metallic spalling debris.",
            "Replace thermal barrier coatings and turbine nozzle guide vanes.",
        ]
    elif rul_val <= 75 or crit_count >= 1 or warn_count >= 3 or health_score < 60 or urgency_index >= 50:
        action_code = "SCHEDULED"
        badge = "🟡 SCHEDULED MAINTENANCE RECOMMENDED"
        recommendation = f"SCHEDULE INSPECTION: Pre-emptive maintenance required within 15–20 cycles. Accelerated wear detected in {target_sub}."
        clearance = "⚠️ CONDITIONAL DISPATCH (Max 20 flight cycles before mandatory shop visit)"
        color = "#d97706"
        bg_color = "#fffbeb"
        downtime = "12 – 24 Hours"
        work_scope = [
            f"Schedule focused borescope inspection of {target_sub}.",
            "Inspect compressor bleed valves and fuel manifold nozzles.",
            "Perform engine ground run vibration analysis across spool RPM ranges.",
            "Verify sensor harness continuity and recalibrate thermistor probes.",
        ]
    elif rul_val <= 120 or warn_count >= 1 or health_score < 80 or urgency_index >= 25:
        action_code = "MONITORING"
        badge = "🔵 ENHANCED MONITORING RECOMMENDED"
        recommendation = "WATCHLIST MONITORING: Turbofan is safe for dispatch with enhanced sensor telemetry sampling. No immediate component replacement needed."
        clearance = "ℹ️ CLEARED FOR DISPATCH (Enhanced Telemetry Logging Active)"
        color = "#2563eb"
        bg_color = "#eff6ff"
        downtime = "2 – 4 Hours (Turnaround Check)"
        work_scope = [
            "Conduct walkaround visual inspection of fan blades and exhaust cone.",
            "Sample lubrication oil for spectrographic analysis at next hub turn.",
            "Increase high-frequency sensor logging rate to 5 Hz on next flight leg.",
        ]
    else:
        action_code = "NOMINAL"
        badge = "🟢 NO ACTION REQUIRED"
        recommendation = "ALL SYSTEMS NOMINAL: Engine is operating well within standard operational envelopes. Normal service schedule applies."
        clearance = "✅ UNRESTRICTED FLIGHT CLEARANCE"
        color = "#16a34a"
        bg_color = "#f0fdf4"
        downtime = "0 Hours (Standard Transit Check)"
        work_scope = [
            "Perform standard pre-flight visual walkaround.",
            "Continue standard line maintenance cycle intervals.",
        ]

    return {
        "action_code": action_code,
        "badge": badge,
        "recommendation": recommendation,
        "clearance": clearance,
        "urgency_index": urgency_index,
        "color": color,
        "bg_color": bg_color,
        "downtime": downtime,
        "triggers": triggers,
        "target_subsystem": target_sub,
        "work_scope": work_scope,
    }



# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    def on_theme_change():
        st.session_state.dark_mode = st.session_state.theme_switch

    st.markdown(
        """<div style="display:flex;align-items:center;justify-content:space-between;padding:0 2px 4px 2px;">
            <span style="font-size:9px;font-weight:800;letter-spacing:1px;color:#94a3b8;font-family:'Consolas','JetBrains Mono',monospace;">// COCKPIT HUD MODE</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.toggle(
        "🌙 Night HUD" if _DM else "☀️ Day Mode",
        value=_DM,
        key="theme_switch",
        on_change=on_theme_change,
        help="Toggle between Cockpit Night Vision HUD and Daylight Mode",
    )
    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """<div style="padding:4px 0 14px 0;border-bottom:1px solid #1e3a5f;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:24px;filter:drop-shadow(0 0 8px #38bdf8);">🛰️</span>
            <div>
                <div style="font-size:17px;font-weight:900;letter-spacing:1px;color:#f8fafc;font-family:'Consolas','JetBrains Mono',monospace;">AEROGUARD</div>
                <div style="font-size:10px;font-weight:700;color:#38bdf8;letter-spacing:1px;font-family:monospace;">FLIGHT INTELLIGENCE // MIL-STD</div>
            </div>
        </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:10px;font-weight:800;color:#64748b;letter-spacing:1px;font-family:monospace;">// CONSOLE NAVIGATION</span>
            <span id="aeroguard-keymap-trigger" class="aeroguard-keymap-trigger" style="font-size:9px;font-weight:700;color:#38bdf8;background:rgba(56,189,248,0.12);padding:2px 7px;border-radius:4px;border:1px solid rgba(56,189,248,0.3);font-family:monospace;cursor:pointer;user-select:none;transition:all 0.2s ease;" title="Click or press '?' / 'H' for avionics flight keymap">[?] KEYMAP</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📈 Engine Monitoring", "🗄️ Sensor Data", "🛠️ Predictive Maintenance", "⚠️ Risk Detection", "🔍 Anomaly Detection", "🎯 RUL Improvement", "🩺 Engine Health Score", "📡 Real-Time Monitoring"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("<div style='font-size:10px;font-weight:800;color:#64748b;letter-spacing:1px;margin:16px 0 8px 0;font-family:monospace;'>// TELEMETRY ACQUISITION</div>", unsafe_allow_html=True)
    st.markdown("**Dataset / Fleet Stream**")
    fd_choice = st.selectbox(
        "Dataset", ["FD001", "FD002", "FD003", "FD004"], label_visibility="collapsed", key="fd_choice"
    )

    df = load_dataset(fd_choice)
    unit_ids = sorted(df["unit"].unique().tolist()) if "unit" in df else [1]
    st.markdown("**Engine Unit Designation**")
    engine_id = st.selectbox(
        "Engine #", unit_ids, format_func=lambda u: f"Turbofan Unit #{int(u):02d}", key="engine_id", label_visibility="collapsed"
    )

    st.markdown("<div style='border-top:1px solid #1e3a5f;margin:16px 0 10px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="font-family:'Consolas','JetBrains Mono',monospace;font-size:11px;color:#cbd5e1;">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <span class="pulse-beacon"></span><b style="color:#10b981;">TELEMETRY LINK: ONLINE</b>
        </div>
        <div style="color:#64748b;font-size:10px;">SAMPLING: 10 Hz · UDP SYNC</div>
        <div style="color:#64748b;font-size:10px;">SYS TIME: {dt.datetime.now().strftime('%H:%M:%S')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# THEME STATE SYNC
# --------------------------------------------------------------------------
_is_dark = st.session_state.get("dark_mode", True)

# --------------------------------------------------------------------------
# AVIONICS FLIGHT KEYMAP & HOTKEY CONTROLLER (JS Bridge)
# --------------------------------------------------------------------------
HOTKEY_CONTROLLER_HTML = """
<script>
(function() {
    try {
        const win = window.parent || window;
        const doc = win.document || document;

        function createModal() {
            if (doc.getElementById('aeroguard-keymap-modal')) return;
            const modal = doc.createElement('div');
            modal.id = 'aeroguard-keymap-modal';
            modal.style.cssText = 'display:none;position:fixed;top:0;left:0;right:0;bottom:0;z-index:2147483647;background:rgba(3,7,18,0.88);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:linear-gradient(145deg, #091326 0%, #0d1e3d 100%);border:1.5px solid rgba(56,189,248,0.4);box-shadow:0 0 40px rgba(56,189,248,0.3), 0 25px 50px rgba(0,0,0,0.85);border-radius:12px;width:92%;max-width:600px;padding:24px 28px;color:#f8fafc;font-family:'Consolas','JetBrains Mono',monospace;">
                    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(56,189,248,0.25);padding-bottom:12px;margin-bottom:16px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="font-size:22px;">🛰️</span>
                            <div>
                                <div style="font-size:15px;font-weight:900;letter-spacing:1px;color:#38bdf8;">AEROGUARD // FLIGHT AVIONICS KEYMAP</div>
                                <div style="font-size:10px;color:#94a3b8;">MIL-STD TELEMETRY SHORTCUT MATRIX</div>
                            </div>
                        </div>
                        <button id="close-aeroguard-keymap-btn" style="background:#0f1c38;border:1px solid #1e3a5f;color:#e2e8f0;padding:5px 12px;border-radius:6px;cursor:pointer;font-family:monospace;font-size:11px;font-weight:700;">[ESC / ×]</button>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;font-size:11px;">
                        <div>
                            <div style="color:#38bdf8;font-weight:800;font-size:10px;letter-spacing:1px;margin-bottom:8px;border-bottom:1px solid #1e293b;padding-bottom:4px;">DIRECT MODULE JUMP</div>
                            <div style="display:flex;flex-direction:column;gap:6px;">
                                <div style="display:flex;justify-content:space-between;"><span>🏠 Dashboard</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">1</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>📈 Engine Monitoring</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">2</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>🗄️ Sensor Data</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">3</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>🛠️ Predictive Maint.</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">4</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>⚠️ Risk Detection</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">5</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>🔍 Anomaly Detection</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">6</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>🎯 RUL Improvement</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">7</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>🩺 Health Score</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">8</kbd></div>
                                <div style="display:flex;justify-content:space-between;"><span>📡 Real-Time Monitor</span><kbd style="background:#081226;border:1px solid #38bdf8;color:#38bdf8;padding:1px 6px;border-radius:3px;">9</kbd></div>
                            </div>
                        </div>
                        <div>
                            <div style="color:#10b981;font-weight:800;font-size:10px;letter-spacing:1px;margin-bottom:8px;border-bottom:1px solid #1e293b;padding-bottom:4px;">SYSTEM & DISPLAY CONTROLS</div>
                            <div style="display:flex;flex-direction:column;gap:8px;">
                                <div>
                                    <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Next Screen</span><div><kbd style="background:#081226;border:1px solid #10b981;color:#10b981;padding:1px 5px;border-radius:3px;">→</kbd> or <kbd style="background:#081226;border:1px solid #10b981;color:#10b981;padding:1px 5px;border-radius:3px;">]</kbd></div></div>
                                    <div style="font-size:9px;color:#94a3b8;">Cycle forward through modules</div>
                                </div>
                                <div>
                                    <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Previous Screen</span><div><kbd style="background:#081226;border:1px solid #10b981;color:#10b981;padding:1px 5px;border-radius:3px;">←</kbd> or <kbd style="background:#081226;border:1px solid #10b981;color:#10b981;padding:1px 5px;border-radius:3px;">[</kbd></div></div>
                                    <div style="font-size:9px;color:#94a3b8;">Cycle backward through modules</div>
                                </div>
                                <div>
                                    <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Toggle Sidebar</span><div><kbd style="background:#081226;border:1px solid #f59e0b;color:#f59e0b;padding:1px 6px;border-radius:3px;font-weight:800;">SPACE</kbd> or <kbd style="background:#081226;border:1px solid #f59e0b;color:#f59e0b;padding:1px 5px;border-radius:3px;">\\\\</kbd></div></div>
                                    <div style="font-size:9px;color:#94a3b8;">Wide-screen focus mode (open/close)</div>
                                </div>
                                <div>
                                    <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Toggle Keymap</span><div><kbd style="background:#081226;border:1px solid #a855f7;color:#a855f7;padding:1px 5px;border-radius:3px;">?</kbd> or <kbd style="background:#081226;border:1px solid #a855f7;color:#a855f7;padding:1px 5px;border-radius:3px;">H</kbd></div></div>
                                    <div style="font-size:9px;color:#94a3b8;">Show / hide this keymap</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top:16px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#64748b;">
                        <span>🛡️ SMART INPUT GUARD ACTIVE (DISABLED WHILE TYPING)</span>
                        <span>PRESS [ESC] TO DISMISS</span>
                    </div>
                </div>
            `;
            doc.body.appendChild(modal);
            modal.addEventListener('click', function(ev) {
                if (ev.target === modal) modal.style.display = 'none';
            });
            const closeBtn = doc.getElementById('close-aeroguard-keymap-btn');
            if (closeBtn) closeBtn.addEventListener('click', function() { modal.style.display = 'none'; });
        }

        win.toggleAeroguardKeymap = function() {
            let modal = doc.getElementById('aeroguard-keymap-modal');
            if (!modal) {
                createModal();
                modal = doc.getElementById('aeroguard-keymap-modal');
            }
            if (modal) {
                const isHidden = (modal.style.display === 'none' || modal.style.display === '');
                modal.style.display = isHidden ? 'flex' : 'none';
            }
        };

        function isInputActive() {
            const el = doc.activeElement;
            if (!el) return false;
            const tag = el.tagName ? el.tagName.toLowerCase() : '';
            if (tag === 'textarea') return true;
            if (el.isContentEditable) return true;
            if (tag === 'input') {
                const type = (el.type || 'text').toLowerCase();
                // Do NOT block radio buttons, checkboxes, range sliders or buttons!
                if (['text', 'number', 'email', 'password', 'search', 'tel', 'url'].includes(type)) {
                    return true;
                }
                return false;
            }
            if (el.getAttribute('role') === 'textbox') return true;
            return false;
        }

        function toggleSidebar() {
            // 1. Look for expand button (when sidebar is closed)
            const expandBtn = doc.querySelector('[data-testid="stExpandSidebarButton"] button')
                           || doc.querySelector('[data-testid="stExpandSidebarButton"]')
                           || doc.querySelector('[data-testid="stSidebarCollapsedControl"] button')
                           || doc.querySelector('[data-testid="stSidebarCollapsedControl"]');

            if (expandBtn) {
                expandBtn.click();
                return;
            }

            // 2. Look for collapse button (when sidebar is open)
            const collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                             || doc.querySelector('[data-testid="stSidebarCollapseButton"]')
                             || doc.querySelector('section[data-testid="stSidebar"] button[kind="header"]')
                             || doc.querySelector('section[data-testid="stSidebar"] button');

            if (collapseBtn) {
                collapseBtn.click();
            }
        }

        function selectModule(idx) {
            const radioGroup = doc.querySelector('section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"]');
            if (!radioGroup) return;
            const labels = Array.from(radioGroup.querySelectorAll('label'));
            if (!labels || !labels[idx]) return;
            const targetLabel = labels[idx];
            const input = targetLabel.querySelector('input');
            if (input) {
                try {
                    const nativeSetter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'checked');
                    if (nativeSetter && nativeSetter.set) {
                        nativeSetter.set.call(input, true);
                    } else {
                        input.checked = true;
                    }
                } catch(e) {
                    input.checked = true;
                }
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
                input.click();
            }
            targetLabel.click();
        }

        function cycleNav(dir) {
            const radioGroup = doc.querySelector('section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"]');
            if (!radioGroup) return;
            const labels = Array.from(radioGroup.querySelectorAll('label'));
            if (!labels.length) return;
            let activeIdx = labels.findIndex(lbl => {
                const inp = lbl.querySelector('input');
                return inp && inp.checked;
            });
            if (activeIdx === -1) activeIdx = 0;
            let nextIdx = activeIdx + dir;
            if (nextIdx < 0) nextIdx = labels.length - 1;
            if (nextIdx >= labels.length) nextIdx = 0;
            selectModule(nextIdx);
        }

        function handleKeyDown(e) {
            if (e.ctrlKey || e.altKey || e.metaKey) return;
            if (isInputActive()) return;

            const key = e.key;
            const code = e.code || '';

            // 1. SPACEBAR: Toggle Sidebar (Expand / Collapse)
            if (key === ' ' || code === 'Space') {
                e.preventDefault();
                e.stopPropagation();
                if (doc.activeElement && doc.activeElement.tagName.toLowerCase() === 'button') {
                    doc.activeElement.blur();
                }
                toggleSidebar();
                return;
            }

            // 2. Direct Number Jump 1-9 (Top row or numpad)
            let num = null;
            if (key >= '1' && key <= '9') {
                num = parseInt(key, 10);
            } else if (code.startsWith('Digit') && code.length === 6) {
                const d = parseInt(code.replace('Digit', ''), 10);
                if (d >= 1 && d <= 9) num = d;
            } else if (code.startsWith('Numpad') && code.length === 7) {
                const d = parseInt(code.replace('Numpad', ''), 10);
                if (d >= 1 && d <= 9) num = d;
            }

            if (num !== null) {
                e.preventDefault();
                e.stopPropagation();
                selectModule(num - 1);
                return;
            }

            // 3. Sequential Cycling
            if (key === 'ArrowRight' || key === ']' || code === 'BracketRight') {
                e.preventDefault();
                e.stopPropagation();
                cycleNav(1);
                return;
            }
            if (key === 'ArrowLeft' || key === '[' || code === 'BracketLeft') {
                e.preventDefault();
                e.stopPropagation();
                cycleNav(-1);
                return;
            }

            // 4. Toggle Sidebar: '\' or 'b' / 'B'
            if (key === '\\\\' || code === 'Backslash' || key === 'b' || key === 'B') {
                e.preventDefault();
                e.stopPropagation();
                toggleSidebar();
                return;
            }

            // 5. Keymap Modal: '?' or '/' or 'h' / 'H'
            if (key === '?' || key === '/' || code === 'Slash' || code === 'KeyH' || key.toLowerCase() === 'h') {
                e.preventDefault();
                e.stopPropagation();
                win.toggleAeroguardKeymap();
                return;
            }

            // 6. Close Modal on Escape
            if (key === 'Escape') {
                const modal = doc.getElementById('aeroguard-keymap-modal');
                if (modal && modal.style.display !== 'none') {
                    e.preventDefault();
                    modal.style.display = 'none';
                }
            }
        }

        function handleDocClick(e) {
            const trigger = e.target.closest('#aeroguard-keymap-trigger, .aeroguard-keymap-trigger');
            if (trigger) {
                e.preventDefault();
                e.stopPropagation();
                win.toggleAeroguardKeymap();
                return;
            }
        }

        // Always re-bind handler on parent doc/window
        if (win.__aeroguard_handler) {
            doc.removeEventListener('keydown', win.__aeroguard_handler, true);
            win.removeEventListener('keydown', win.__aeroguard_handler, true);
            document.removeEventListener('keydown', win.__aeroguard_handler, true);
            window.removeEventListener('keydown', win.__aeroguard_handler, true);
        }
        if (win.__aeroguard_click_handler) {
            doc.removeEventListener('click', win.__aeroguard_click_handler, true);
            win.removeEventListener('click', win.__aeroguard_click_handler, true);
            document.removeEventListener('click', win.__aeroguard_click_handler, true);
            window.removeEventListener('click', win.__aeroguard_click_handler, true);
        }
        win.__aeroguard_handler = handleKeyDown;
        win.__aeroguard_click_handler = handleDocClick;

        doc.addEventListener('keydown', handleKeyDown, true);
        win.addEventListener('keydown', handleKeyDown, true);
        document.addEventListener('keydown', handleKeyDown, true);
        window.addEventListener('keydown', handleKeyDown, true);

        doc.addEventListener('click', handleDocClick, true);
        win.addEventListener('click', handleDocClick, true);
        document.addEventListener('click', handleDocClick, true);
        window.addEventListener('click', handleDocClick, true);

        createModal();
    } catch (err) {
        console.warn("AeroGuard Hotkey Controller:", err);
    }
})();
</script>
"""
components.html(HOTKEY_CONTROLLER_HTML, height=0, width=0)

# --------------------------------------------------------------------------
# SHARED COMPUTATIONS (based on selected engine/dataset, used across all pages)
# --------------------------------------------------------------------------
engine_df = df[df["unit"] == engine_id].reset_index(drop=True) if "unit" in df else df
if engine_df.empty:
    engine_df = df
latest = engine_df.iloc[-1]
prev = engine_df.iloc[-2] if len(engine_df) > 1 else latest

model, scaler = load_model_and_scaler()
imp_model, imp_scaler, imp_feats = load_improved_model_and_scaler()
q10_model, q90_model, q_meta = load_quantile_models()

rul_base_arr, base_used = predict_baseline_rul_batch(model, scaler, engine_df)
rul_imp_arr, rul_imp_lo_arr, rul_imp_hi_arr, imp_used, q_bounds_used = predict_improved_rul_intervals_batch(
    imp_model, imp_scaler, imp_feats, q10_model, q90_model, q_meta, engine_df
)

rul_pred = float(rul_imp_arr[-1]) if imp_used else float(rul_base_arr[-1])
rul_lo_latest = float(rul_imp_lo_arr[-1])
rul_hi_latest = float(rul_imp_hi_arr[-1])
rul_half_width = max(1.0, 0.5 * (rul_hi_latest - rul_lo_latest))
rul_base_latest = float(rul_base_arr[-1])
rul_true_latest = float(latest["RUL"]) if "RUL" in latest else None
model_used = imp_used or base_used

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
    "🛠️ Predictive Maintenance": ("Predictive Maintenance Intelligence System", "Multi-factor decision-making engine · Sensor anomalies, RUL, health score & operational action protocols"),
    "🔧 Maintenance": ("Predictive Maintenance Intelligence System", "Multi-factor decision-making engine · Sensor anomalies, RUL, health score & operational action protocols"),
    "⚠️ Risk Detection": ("Engine Failure Risk Detection", "ML-powered failure risk classification · Normal · Warning · Critical"),
    "🔍 Anomaly Detection": ("Sensor Anomaly Detection", "Identify abnormal sensor readings · ML-based z-score & trend analysis"),
    "🎯 RUL Improvement": ("RUL Prediction Improvement", "Feature-engineered machine learning model · 45.7% lower error · Model comparison"),
    "🩺 Engine Health Score": ("Engine Health Score System", "Multi-factor prognostic health index · Healthy · Monitor · Warning · Critical"),
    "📡 Real-Time Monitoring": ("Real-Time Aircraft Engine Monitoring", "Continuous cycle-by-cycle sensor stream · Live RUL, health status & automated alerts"),
}
title_text, sub_text = page_titles[page]
st.markdown(
    f"""
    <div class="hud-header">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
            <div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                    <span class="pulse-beacon"></span>
                    <span style="font-family:'Consolas','JetBrains Mono',monospace;font-size:10.5px;font-weight:800;letter-spacing:1.2px;color:#38bdf8;">AEROGUARD // FLIGHT TELEMETRY COMMAND CONSOLE</span>
                    <span style="background:rgba(56,189,248,0.15);border:1px solid #38bdf8;color:#38bdf8;font-size:10px;font-weight:800;padding:2px 8px;border-radius:4px;font-family:monospace;">MIL-SPEC MONITORING</span>
                </div>
                <div style="font-size:24px;font-weight:900;margin:0;letter-spacing:-0.5px;color:#f8fafc;font-family:'Consolas','JetBrains Mono',monospace;">{title_text}</div>
                <div style="font-size:12px;color:#94a3b8;font-family:'Consolas','JetBrains Mono',monospace;margin-top:2px;">{sub_text}</div>
            </div>
            <div style="text-align:right;background:rgba(15,28,56,0.7);border:1px solid #1e3a5f;padding:10px 18px;border-radius:8px;font-family:'Consolas','JetBrains Mono',monospace;font-size:11.5px;color:#cbd5e1;">
                <div><b style="color:#38bdf8;">UNIT:</b> ENG #{int(engine_id):02d} &nbsp;|&nbsp; <b style="color:#38bdf8;">FLEET:</b> {fd_choice} &nbsp;|&nbsp; <b style="color:#38bdf8;">RATE:</b> 10 Hz</div>
                <div style="margin-top:3px;font-size:11px;color:#64748b;">UTC {dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}Z &nbsp;·&nbsp; {dt.datetime.now().strftime('%b %d, %Y')} &nbsp;|&nbsp; <span style="color:#10b981;font-weight:bold;">🟢 AIRWORTHY</span></div>
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
        src = "CQR 80% CI" if q_bounds_used else ("Est. CI" if model_used else "Estimated")
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📊 RUL Prediction</div>
            <div class="kpi-value">{rul_pred:,.0f} cycles</div>
            <div class="kpi-sub">{src}: <b>{rul_lo_latest:.0f}–{rul_hi_latest:.0f} cyc</b> (±{rul_half_width:.0f})</div>
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
            height=380, margin=dict(l=14, r=14, t=14, b=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
            <div class="panel-title">🛰️ Turbofan Digital Twin & Station Telemetry</div>
            <div class="panel-sub">Engine #{int(engine_id):02d} · Twin-Spool High-Bypass Architecture</div>
            {render_turbofan_schematic_svg(hpc_temp, lpc_temp, pressure, rpm, cycle_now, health_label, health_color)}
            <table style="width:100%;font-size:12px;font-family:'Consolas','JetBrains Mono',monospace;margin-top:10px;">
                <tr><td style="color:{_sub_c};">⚙️ OPERATING CYCLE</td><td style="text-align:right;font-weight:800;color:#38bdf8;">{cycle_now} cyc</td></tr>
                <tr><td style="color:{_sub_c};">🔄 CORE SPEED (RPM)</td><td style="text-align:right;font-weight:800;color:#818cf8;">{rpm:,.0f} RPM</td></tr>
                <tr><td style="color:{_sub_c};">📟 COMBUSTOR STATIC PRESS</td><td style="text-align:right;font-weight:800;color:#fbbf24;">{pressure:,.2f} psia</td></tr>
                <tr><td style="color:{_sub_c};">🌡️ HPC OUTLET TEMP (ST 3)</td><td style="text-align:right;font-weight:800;color:#ef4444;">{hpc_temp:,.1f} °C</td></tr>
                <tr><td style="color:{_sub_c};">🌡️ LPC OUTLET TEMP (ST 2)</td><td style="text-align:right;font-weight:800;color:#10b981;">{lpc_temp:,.1f} °C</td></tr>
            </table>
            </div>""",
            unsafe_allow_html=True,
        )

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
                             paper_bgcolor="#e5ebf3", font=dict(color=TEXT_DARK))
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
                             paper_bgcolor="#e5ebf3", font=dict(color=TEXT_DARK),
                             polar=dict(bgcolor="#eef2f6",
                                        radialaxis=dict(visible=True, range=[0, 100], color=TEXT_DARK,
                                                         gridcolor="#cbd5e1"),
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
    _mon_lbl_col = "#9ca3af" if _DM else "#6b7280"
    st.markdown(
        f'<div style="color:{_mon_lbl_col};font-size:14px;font-weight:700;margin:2px 0 6px;">Choose a sensor to inspect</div>',
        unsafe_allow_html=True,
    )
    chosen_sensor = st.selectbox(
        "Choose a sensor to inspect",
        sensor_options,
        label_visibility="collapsed",
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
    st.markdown(
        f'<div style="color:{_mon_lbl_col};font-size:14px;font-weight:700;margin:4px 0 6px;">Sensors to compare</div>',
        unsafe_allow_html=True,
    )
    multi_sensors = st.multiselect(
        "Sensors to compare", sensor_options,
        default=[s for s in default_multi if s in sensor_options],
        label_visibility="collapsed",
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)))
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

    _sd_lbl_col = "#9ca3af" if _DM else "#6b7280"
    st.markdown(
        f'<div style="color:{_sd_lbl_col};font-size:14px;font-weight:700;margin:2px 0 6px;">Columns to show</div>',
        unsafe_allow_html=True,
    )
    show_cols = st.multiselect(
        "Columns to show", list(engine_df.columns), default=list(engine_df.columns)[:10],
        label_visibility="collapsed",
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
# PAGE 4 — PREDICTIVE MAINTENANCE SYSTEM (TASK 6)
# ==========================================================================
elif page in ["🛠️ Predictive Maintenance", "🔧 Maintenance"]:

    _dm = st.session_state.get("dark_mode", True)
    banner_txt = "#f8fafc" if _dm else "#0f172a"
    sub_box_bg = "#334155" if _dm else "#ffffff"
    sub_box_txt = "#cbd5e1" if _dm else "#475569"

    # 1. Compute multi-factor inputs for currently selected engine
    hlth_info = compute_engine_health_score(engine_df, rul_pred)
    maint_eval = evaluate_predictive_maintenance(
        rul_val=rul_pred,
        health_score=hlth_info["total_score"],
        crit_count=hlth_info["crit_count"],
        warn_count=hlth_info["warn_count"],
        subsystems=hlth_info["subsystems"],
    )

    m_code = maint_eval["action_code"]
    m_badge = maint_eval["badge"]
    m_rec = maint_eval["recommendation"]
    m_clr = maint_eval["clearance"]
    m_urgency = maint_eval["urgency_index"]
    m_col = maint_eval["color"]
    m_bg = maint_eval["bg_color"] if not _dm else "#1e293b"
    m_downtime = maint_eval["downtime"]
    m_target = maint_eval["target_subsystem"]

    # -----------------------------------------------------------------------
    # 1. MASTER DECISION & DISPATCH RECOMMENDATION BANNER
    # -----------------------------------------------------------------------
    st.markdown(
f"""<div style="background:{m_bg};border:2px solid {m_col};border-radius:16px;padding:22px 26px;margin-bottom:20px;box-shadow:0 4px 18px rgba(0,0,0,0.08);">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:gap:16px;">
<div>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
<span style="font-size:26px;font-weight:900;color:{m_col};letter-spacing:-0.5px;">{m_badge}</span>
</div>
<div style="font-size:14.5px;color:{banner_txt};max-width:700px;line-height:1.5;margin-bottom:8px;">{m_rec}</div>
<div style="font-size:13px;font-weight:800;color:{m_col};background:{m_col}22;padding:4px 12px;border-radius:6px;display:inline-block;">{m_clr}</div>
</div>
<div style="text-align:right;background:{sub_box_bg};padding:14px 22px;border-radius:14px;border:1px solid {m_col}44;min-width:180px;">
<div style="font-size:11px;font-weight:700;color:{sub_box_txt};text-transform:uppercase;letter-spacing:0.5px;">Maintenance Urgency Index</div>
<div style="font-size:38px;font-weight:900;color:{m_col};line-height:1.1;">{m_urgency}<span style="font-size:18px;color:{sub_box_txt};font-weight:600;">%</span></div>
<div style="font-size:12px;color:{m_col};font-weight:700;">Est. Turnaround: {m_downtime}</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 2. DECISION KPI GRID
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🛠️ Maintenance Action</div>
            <div class="kpi-value" style="font-size:22px;color:{m_col};">{m_code}</div>
            <div class="kpi-sub">{m_badge.split()[0]} Decision Outcome</div>
            </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📊 Urgency Index (MUI)</div>
            <div class="kpi-value" style="color:{m_col};">{m_urgency}%</div>
            <div class="kpi-sub">Composite maintenance priority</div>
            </div>""", unsafe_allow_html=True)
    with k3:
        _pm_ci_tag = "80% CQR" if q_bounds_used else "Est. Band"
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⏱️ Forecasted RUL</div>
            <div class="kpi-value" style="color:#2563eb;">{rul_pred:,.0f} cyc</div>
            <div class="kpi-sub">{_pm_ci_tag}: <b>{rul_lo_latest:.0f}–{rul_hi_latest:.0f} cyc</b></div>
            </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🎯 Target Subsystem</div>
            <div class="kpi-value" style="font-size:19px;color:#7c3aed;">{m_target.split()[0]} Section</div>
            <div class="kpi-sub">{m_target}</div>
            </div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. MULTI-FACTOR DECISION-MAKING LOGIC MATRIX
    # -----------------------------------------------------------------------
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">🧠 Multi-Factor Decision-Making Engine Logic</div>
        <div class="panel-sub">Integrated evaluation of prognostic RUL, health score physical wear, sensor anomalies, and failure risk.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {'#dc2626' if rul_pred <= 30 else ('#d97706' if rul_pred <= 75 else '#16a34a')};border-radius:10px;padding:14px;">
            <div style="font-size:11px;font-weight:700;color:{sub_box_txt};">1. PROGNOSTIC RUL</div>
            <div style="font-size:22px;font-weight:800;color:{'#dc2626' if rul_pred <= 30 else ('#d97706' if rul_pred <= 75 else '#16a34a')};">{rul_pred:,.0f} cyc</div>
            <div style="font-size:12px;color:{sub_box_txt};line-height:1.4;margin-top:4px;">
                {'🔴 Critical (<= 30 cyc)' if rul_pred <= 30 else ('🟡 Scheduled (<= 75 cyc)' if rul_pred <= 75 else '🟢 Normal Clearance')}
            </div>
            </div>""", unsafe_allow_html=True)
    with f2:
        h_val = hlth_info["total_score"]
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {hlth_info['color']};border-radius:10px;padding:14px;">
            <div style="font-size:11px;font-weight:700;color:{sub_box_txt};">2. ENGINE HEALTH INDEX</div>
            <div style="font-size:22px;font-weight:800;color:{hlth_info['color']};">{h_val:.1f} / 100</div>
            <div style="font-size:12px;color:{sub_box_txt};line-height:1.4;margin-top:4px;">
                {hlth_info['icon']} Tier: {hlth_info['tier']} Condition
            </div>
            </div>""", unsafe_allow_html=True)
    with f3:
        n_crit = hlth_info["crit_count"]
        n_warn = hlth_info["warn_count"]
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {'#dc2626' if n_crit > 0 else ('#d97706' if n_warn > 0 else '#16a34a')};border-radius:10px;padding:14px;">
            <div style="font-size:11px;font-weight:700;color:{sub_box_txt};">3. SENSOR ANOMALIES</div>
            <div style="font-size:22px;font-weight:800;color:{'#dc2626' if n_crit > 0 else ('#d97706' if n_warn > 0 else '#16a34a')};">{n_crit} Crit · {n_warn} Warn</div>
            <div style="font-size:12px;color:{sub_box_txt};line-height:1.4;margin-top:4px;">
                {'🔴 Severe multi-channel drift' if n_crit > 0 else ('🟡 Elevated variance' if n_warn > 0 else '🟢 Nominal telemetry')}
            </div>
            </div>""", unsafe_allow_html=True)
    with f4:
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {health_color};border-radius:10px;padding:14px;">
            <div style="font-size:11px;font-weight:700;color:{sub_box_txt};">4. ML RISK LEVEL</div>
            <div style="font-size:22px;font-weight:800;color:{health_color};">{risk_label}</div>
            <div style="font-size:12px;color:{sub_box_txt};line-height:1.4;margin-top:4px;">
                AeroGuard risk model assessment
            </div>
            </div>""", unsafe_allow_html=True)

    # Active Trigger Rules Ledger
    st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
    if maint_eval["triggers"]:
        triggers_html = "".join([f"<li style='margin-bottom:6px;'>{t}</li>" for t in maint_eval["triggers"]])
        st.markdown(
f"""<div style="background:{m_col}15;border-left:4px solid {m_col};border-radius:8px;padding:12px 18px;margin-bottom:15px;">
<div style="font-weight:800;font-size:13.5px;color:{m_col};margin-bottom:6px;">⚠️ Triggered Decision-Making Rules (Engine #{int(engine_id):02d}):</div>
<ul style="margin:0;padding-left:18px;font-size:13px;color:{banner_txt};">
{triggers_html}
</ul>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
f"""<div style="background:#f0fdf4;border-left:4px solid #16a34a;border-radius:8px;padding:12px 18px;margin-bottom:15px;">
<div style="font-weight:800;font-size:13.5px;color:#16a34a;">✅ All Rules Satisfied: Engine #{int(engine_id):02d} operates within safe flight envelopes. No maintenance triggers active.</div>
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Compact XAI Root-Cause Attribution Snapshot (Why this RUL?)
    maint_xai = compute_local_rul_attribution(imp_model, imp_scaler, imp_feats, engine_df, row_idx=-1)
    if maint_xai is not None:
        m_deg = maint_xai["top_degrading"].head(3)
        m_stab = maint_xai["top_stabilizing"].head(2)
        deg_chips = "".join(
            f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:4px 9px;border-radius:6px;'
            f'background:rgba(239,68,68,0.13);border:1px solid rgba(239,68,68,0.35);font-size:12px;color:{banner_txt};">'
            f'<b>{r["label"]}</b> <span style="color:#ef4444;font-family:monospace;font-weight:800;">{r["contribution"]:+.1f} cyc</span></span>'
            for _, r in m_deg.iterrows()
        ) or f'<span style="font-size:12px;color:{sub_box_txt};">None active</span>'
        stab_chips = "".join(
            f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:4px 9px;border-radius:6px;'
            f'background:rgba(16,185,129,0.13);border:1px solid rgba(16,185,129,0.35);font-size:12px;color:{banner_txt};">'
            f'<b>{r["label"]}</b> <span style="color:#10b981;font-family:monospace;font-weight:800;">{r["contribution"]:+.1f} cyc</span></span>'
            for _, r in m_stab.iterrows()
        ) or f'<span style="font-size:12px;color:{sub_box_txt};">None active</span>'

        _snap_brd = "#334155" if _dm else "#cbd5e1"
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {_snap_brd};border-left:4px solid #38bdf8;border-radius:10px;padding:13px 18px;margin-bottom:18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
                <div style="font-weight:800;font-size:13.5px;color:#38bdf8;">🧠 XAI RUL Root-Cause Snapshot · Cycle {maint_xai['engine_cycle']} (Baseline {maint_xai['base_value']:.0f} → {maint_xai['predicted_rul']:.0f} cyc)</div>
                <div style="font-size:11.5px;color:{sub_box_txt};">Open <b>[7] 🎯 RUL Improvement</b> for full interactive Waterfall chart</div>
            </div>
            <div style="font-size:12px;color:{sub_box_txt};margin-bottom:4px;"><b style="color:#ef4444;">🔻 Degrading Penalties:</b> {deg_chips}</div>
            <div style="font-size:12px;color:{sub_box_txt};"><b style="color:#10b981;">🔺 Stabilizing Buffers:</b> {stab_chips}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 4. INTERACTIVE "WHAT-IF" SCENARIO SIMULATOR & CONDITION STRESS-TESTER
    # -----------------------------------------------------------------------
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">🧪 Interactive Engine Condition Stress-Tester ("What-If" Simulator)</div>
        <div class="panel-sub">Test and validate the predictive maintenance decision logic across diverse operational flight conditions and degradation states.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Pre-set scenario buttons
    _pm_lbl_col = "#9ca3af" if _dm else "#6b7280"
    st.markdown(
        f'<div style="color:{_pm_lbl_col};font-size:14px;font-weight:700;margin:4px 0 8px;">Choose a Benchmark Operational Condition Scenario:</div>',
        unsafe_allow_html=True,
    )
    sc1, sc2, sc3, sc4 = st.columns(4)

    # Session state for simulator values
    if "sim_rul" not in st.session_state:
        st.session_state.sim_rul = float(rul_pred)
    if "sim_health" not in st.session_state:
        st.session_state.sim_health = float(hlth_info["total_score"])
    if "sim_crit" not in st.session_state:
        st.session_state.sim_crit = int(hlth_info["crit_count"])
    if "sim_warn" not in st.session_state:
        st.session_state.sim_warn = int(hlth_info["warn_count"])

    with sc1:
        if st.button("🟢 Scenario 1: Factory Fresh", use_container_width=True, key="sc_fresh"):
            st.session_state.sim_rul = 195.0
            st.session_state.sim_health = 96.0
            st.session_state.sim_crit = 0
            st.session_state.sim_warn = 0
            st.rerun()
    with sc2:
        if st.button("🔵 Scenario 2: Mid-Life Flutter", use_container_width=True, key="sc_mid"):
            st.session_state.sim_rul = 92.0
            st.session_state.sim_health = 74.0
            st.session_state.sim_crit = 0
            st.session_state.sim_warn = 2
            st.rerun()
    with sc3:
        if st.button("🟡 Scenario 3: Turbine Thermal Wear", use_container_width=True, key="sc_warn"):
            st.session_state.sim_rul = 48.0
            st.session_state.sim_health = 52.0
            st.session_state.sim_crit = 0
            st.session_state.sim_warn = 4
            st.rerun()
    with sc4:
        if st.button("🔴 Scenario 4: Imminent Failure", use_container_width=True, key="sc_crit"):
            st.session_state.sim_rul = 16.0
            st.session_state.sim_health = 26.0
            st.session_state.sim_crit = 2
            st.session_state.sim_warn = 5
            st.rerun()

    # Manual interactive sliders
    st.markdown(
        f'<div style="color:{_pm_lbl_col};font-size:14px;font-weight:700;margin:10px 0 6px;">Or fine-tune custom stress-test parameters manually:</div>',
        unsafe_allow_html=True,
    )
    sl_col1, sl_col2, sl_col3, sl_col4 = st.columns(4)
    with sl_col1:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:13px;font-weight:600;margin-bottom:2px;">Simulated RUL (Cycles)</div>', unsafe_allow_html=True)
        sim_rul_in = st.slider("Simulated RUL (Cycles)", min_value=0, max_value=250, value=int(st.session_state.sim_rul), key="sl_rul", label_visibility="collapsed")
    with sl_col2:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:13px;font-weight:600;margin-bottom:2px;">Simulated Health Score</div>', unsafe_allow_html=True)
        sim_health_in = st.slider("Simulated Health Score", min_value=0, max_value=100, value=int(st.session_state.sim_health), key="sl_health", label_visibility="collapsed")
    with sl_col3:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:13px;font-weight:600;margin-bottom:2px;">Critical Anomalies (|Z|&gt;2.5σ)</div>', unsafe_allow_html=True)
        sim_crit_in = st.slider("Critical Anomalies (|Z|>2.5σ)", min_value=0, max_value=6, value=int(st.session_state.sim_crit), key="sl_crit", label_visibility="collapsed")
    with sl_col4:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:13px;font-weight:600;margin-bottom:2px;">Warning Anomalies (|Z|&gt;1.5σ)</div>', unsafe_allow_html=True)
        sim_warn_in = st.slider("Warning Anomalies (|Z|>1.5σ)", min_value=0, max_value=12, value=int(st.session_state.sim_warn), key="sl_warn", label_visibility="collapsed")

    # Evaluate simulated scenario
    sim_res = evaluate_predictive_maintenance(sim_rul_in, sim_health_in, sim_crit_in, sim_warn_in)
    sim_col = sim_res["color"]
    sim_bg = sim_res["bg_color"] if not _dm else "#1e293b"

    st.markdown(
f"""<div style="background:{sim_bg};border:2px dashed {sim_col};border-radius:12px;padding:18px 22px;margin:15px 0;">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
<div>
<div style="font-size:12px;font-weight:700;color:{sub_box_txt};text-transform:uppercase;">🧪 Simulator Stress-Test Outcome:</div>
<div style="font-size:20px;font-weight:900;color:{sim_col};margin:2px 0;">{sim_res['badge']}</div>
<div style="font-size:13.5px;color:{banner_txt};max-width:650px;">{sim_res['recommendation']}</div>
</div>
<div style="text-align:right;">
<div style="font-size:12px;color:{sub_box_txt};font-weight:700;">Simulated Urgency Index:</div>
<div style="font-size:32px;font-weight:900;color:{sim_col};">{sim_res['urgency_index']}%</div>
<div style="font-size:12px;font-weight:800;color:{sim_col};">{sim_res['clearance']}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 5. COMPONENT-LEVEL MAINTENANCE SCOPE & WORK ORDER
    # -----------------------------------------------------------------------
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">📋 Prescriptive Work Order & Shop Action Scope</div>
        <div class="panel-sub">Targeted inspection directives, tooling requirements, and component staging guidelines.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    wo_col1, wo_col2 = st.columns([1.3, 1.0])
    with wo_col1:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:14px;font-weight:700;margin-bottom:6px;">FAA / EASA Standard Work Order Directives:</div>', unsafe_allow_html=True)
        wo_items = "".join([f"<li style='margin-bottom:8px;line-height:1.5;'>{item}</li>" for item in maint_eval["work_scope"]])
        st.markdown(
            f"""<div style="background:{sub_box_bg};border:1px solid {_dm and '#334155' or '#e2e8f0'};border-radius:10px;padding:16px 20px;">
            <ul style="margin:0;padding-left:18px;font-size:13.5px;color:{banner_txt};">
            {wo_items}
            </ul>
            </div>""", unsafe_allow_html=True)

    with wo_col2:
        st.markdown(f'<div style="color:{_pm_lbl_col};font-size:14px;font-weight:700;margin-bottom:6px;">Component Degradation Radar:</div>', unsafe_allow_html=True)
        subsys_dict = hlth_info["subsystems"]
        for s_n, s_v in subsys_dict.items():
            if s_v < 60:
                bar_c = "#dc2626"
            elif s_v < 80:
                bar_c = "#d97706"
            else:
                bar_c = "#16a34a"
            st.markdown(
f"""<div style="margin-bottom:8px;font-size:12.5px;">
<div style="display:flex;justify-content:space-between;margin-bottom:2px;">
<span style="font-weight:600;color:{banner_txt};">{s_n}</span>
<span style="font-weight:800;color:{bar_c};">{s_v:.1f}/100</span>
</div>
<div style="width:100%;background:{_dm and '#334155' or '#e2e8f0'};height:6px;border-radius:3px;">
<div style="width:{min(max(s_v,0),100)}%;background:{bar_c};height:6px;border-radius:3px;"></div>
</div>
</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 6. FLEET-WIDE MAINTENANCE PRIORITY QUEUE
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🌐 Fleet Maintenance Dispatch & Priority Queue ({fd_choice})</div>
        <div class="panel-sub">Rank-ordered maintenance scheduling queue across all engines in the dataset, prioritized by Maintenance Urgency Index (MUI).</div>
        </div>""",
        unsafe_allow_html=True,
    )

    fleet_maint_summary = get_fleet_health_summary(df, fd_choice)
    if not fleet_maint_summary.empty:
        fleet_queue_rows = []
        for _, r in fleet_maint_summary.iterrows():
            m_res = evaluate_predictive_maintenance(
                rul_val=float(r["rul"]),
                health_score=float(r["health_score"]),
                crit_count=2 if r["tier"] == "Critical" else (1 if r["tier"] == "Warning" else 0),
                warn_count=4 if r["tier"] == "Critical" else (2 if r["tier"] == "Warning" else (1 if r["tier"] == "Monitor" else 0)),
            )
            fleet_queue_rows.append({
                "Engine": r["engine"],
                "Urgency Index (MUI)": f"{m_res['urgency_index']:.1f}%",
                "Action Verdict": f"{m_res['badge'].split()[0]} {m_res['action_code']}",
                "Dispatch Clearance": m_res["clearance"].split()[0] + " " + m_res["clearance"].split()[1],
                "Forecasted RUL": f"{r['rul']:.0f} cyc",
                "Health Score": f"{r['health_score']:.1f} / 100",
                "Est. Turnaround": m_res["downtime"].split()[0] + " " + m_res["downtime"].split()[1] + " " + m_res["downtime"].split()[2],
                "_raw_urgency": m_res["urgency_index"],
            })

        fleet_queue_df = pd.DataFrame(fleet_queue_rows).sort_values("_raw_urgency", ascending=False).drop(columns=["_raw_urgency"]).reset_index(drop=True)
        st.dataframe(fleet_queue_df, use_container_width=True, hide_index=True, height=300)


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
            paper_bgcolor="#e5ebf3",
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
        _heat_text = "#0f172a"
        _heat_bg = "#e5ebf3"
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
            plot_bgcolor="#eef2f6",
            font=dict(color=_heat_text),
            xaxis=dict(title="Engine Cycle", color=_heat_text, gridcolor="#cbd5e1"),
            yaxis=dict(title="Sensor", color=_heat_text, autorange="reversed",
                       gridcolor="#cbd5e1"),
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
    _rk_lbl_col = "#9ca3af" if _DM else "#6b7280"
    st.markdown(f'<h3 style="color:{_rk_lbl_col};margin:8px 0 10px;">🚨 Maintenance Action</h3>', unsafe_allow_html=True)
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
            yaxis=dict(range=[0, 115]),
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
        _r_txt = "#0f172a"
        _r_bg = "#e5ebf3"
        radar2.update_layout(
            height=340, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=_r_bg, font=dict(color=_r_txt),
            polar=dict(
                bgcolor="#eef2f6",
                radialaxis=dict(visible=True, range=[0, max(3.0, max(values) * 1.1)],
                                color=_r_txt, gridcolor="#cbd5e1"),
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
    _an_lbl_col = "#9ca3af" if _DM else "#6b7280"
    st.markdown(f'<h3 style="color:{_an_lbl_col};margin:6px 0 4px;">🔍 Sensor Deep Dive</h3>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{_an_lbl_col};font-size:14px;margin-bottom:10px;">Select any sensor to inspect its full history, rolling z-score, and anomaly timeline.</div>', unsafe_allow_html=True)

    sensor_opts = [c for c in engine_df.columns if c.startswith("sensor_")]
    # Default to the most anomalous sensor
    default_idx = int(anom_report["worst_z"].abs().idxmax())
    default_sensor = anom_report.loc[default_idx, "sensor"]
    st.markdown(f'<div style="color:{_an_lbl_col};font-size:13.5px;font-weight:700;margin:2px 0 6px;">Choose a sensor</div>', unsafe_allow_html=True)
    dd_sensor = st.selectbox(
        "Choose a sensor",
        sensor_opts,
        index=sensor_opts.index(default_sensor) if default_sensor in sensor_opts else 0,
        label_visibility="collapsed",
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

    _h2_txt = "#0f172a"
    _h2_bg = "#e5ebf3"

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
        paper_bgcolor=_h2_bg, plot_bgcolor="#eef2f6",
        font=dict(color=_h2_txt),
        xaxis=dict(title="Engine Cycle", color=_h2_txt, gridcolor="#cbd5e1"),
        yaxis=dict(title="Sensor", color=_h2_txt, autorange="reversed",
                   gridcolor="#cbd5e1"),
    )
    st.plotly_chart(tl_fig, use_container_width=True, key="anom_timeline_heatmap")

    # -----------------------------------------------------------------------
    # SECTION 8 — MAINTENANCE RECOMMENDATION
    # -----------------------------------------------------------------------
    st.markdown(f'<h3 style="color:{_an_lbl_col};margin:12px 0 10px;">🛠️ Recommended Actions</h3>', unsafe_allow_html=True)
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

# ==========================================================================
# PAGE 7 — RUL PREDICTION IMPROVEMENT (TASK 3)
# ==========================================================================
elif page == "🎯 RUL Improvement":

    # Compute cycle-by-cycle predictions for this engine under both models
    eng_cycles = engine_df["cycle"].values.astype(int)
    has_ground_truth = "RUL" in engine_df.columns
    true_rul_arr = engine_df["RUL"].values.astype(float) if has_ground_truth else None

    # Baseline Model prediction across all cycles of this engine
    base_preds_engine, _ = predict_baseline_rul_batch(model, scaler, engine_df)
    # Improved Model prediction + 80% Conformal Quantile Bounds across all cycles
    imp_preds_engine, imp_lo_engine, imp_hi_engine, _, imp_q_used = predict_improved_rul_intervals_batch(
        imp_model, imp_scaler, imp_feats, q10_model, q90_model, q_meta, engine_df
    )

    latest_imp_pred = float(imp_preds_engine[-1])
    latest_imp_lo = float(imp_lo_engine[-1])
    latest_imp_hi = float(imp_hi_engine[-1])
    latest_base_pred = float(base_preds_engine[-1])
    latest_true = float(true_rul_arr[-1]) if has_ground_truth else None

    # Error computations on latest reading
    if latest_true is not None:
        imp_err = abs(latest_imp_pred - latest_true)
        base_err = abs(latest_base_pred - latest_true)
        err_reduction_latest = max(0.0, base_err - imp_err)
    else:
        imp_err, base_err, err_reduction_latest = None, None, None

    # -----------------------------------------------------------------------
    # 1. ENGINE PREDICTION VERDICT BANNER
    # -----------------------------------------------------------------------
    _b_bg = "#1e293b" if _DM else "#f0fdf4"
    _b_brd = "#3b82f6" if _DM else "#16a34a"
    _st_col = "#94a3b8" if _DM else "#64748b"
    _mt_col = "#60a5fa" if _DM else "#0f172a"
    _ds_col = "#cbd5e1" if _DM else "#334155"

    _c_bg = "#0f172a" if _DM else "#ffffff"
    _c_brd = "#334155" if _DM else "#e2e8f0"
    _c_lbl = "#94a3b8" if _DM else "#64748b"
    _imp_sub = f"Error: {imp_err:.1f} cyc" if imp_err is not None else "73 Features"
    _base_sub = f"Error: {base_err:.1f} cyc" if base_err is not None else "6 Features"

    improved_card_html = (
        f'<div style="background:{_c_bg};padding:12px 18px;border-radius:12px;border:1px solid {_c_brd};text-align:center;min-width:130px;">'
        f'<div style="font-size:11px;color:{_c_lbl};font-weight:700;">IMPROVED MODEL</div>'
        f'<div style="font-size:24px;font-weight:900;color:#2563eb;">{latest_imp_pred:,.0f} <span style="font-size:12px;">cyc</span></div>'
        f'<div style="font-size:11px;color:#38bdf8;font-weight:700;">80% CI: {latest_imp_lo:.0f}–{latest_imp_hi:.0f}</div>'
        f'<div style="font-size:10.5px;color:#16a34a;font-weight:700;">{_imp_sub}</div>'
        f'</div>'
    )

    baseline_card_html = (
        f'<div style="background:{_c_bg};padding:12px 18px;border-radius:12px;border:1px solid {_c_brd};text-align:center;min-width:110px;">'
        f'<div style="font-size:11px;color:{_c_lbl};font-weight:700;">BASELINE MODEL</div>'
        f'<div style="font-size:24px;font-weight:900;color:#dc2626;">{latest_base_pred:,.0f} <span style="font-size:12px;">cyc</span></div>'
        f'<div style="font-size:11px;color:#dc2626;font-weight:700;">{_base_sub}</div>'
        f'</div>'
    )

    if latest_true is not None:
        true_card_html = (
            f'<div style="background:{_c_bg};padding:12px 18px;border-radius:12px;border:1px solid {_c_brd};text-align:center;min-width:110px;">'
            f'<div style="font-size:11px;color:{_c_lbl};font-weight:700;">TRUE RUL</div>'
            f'<div style="font-size:24px;font-weight:900;color:#16a34a;">{latest_true:,.0f} <span style="font-size:12px;">cyc</span></div>'
            f'<div style="font-size:11px;color:#16a34a;font-weight:700;">Ground Truth</div>'
            f'</div>'
        )
    else:
        true_card_html = ""

    st.markdown(
        f"""<div style="background:{_b_bg};border:2px solid {_b_brd};border-radius:18px;padding:24px 32px;margin-bottom:22px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px;">
<div style="min-width:260px;">
<div style="font-size:12px;font-weight:700;color:{_st_col};text-transform:uppercase;letter-spacing:.08em;">Engine #{int(engine_id):02d} &nbsp;·&nbsp; {fd_choice} &nbsp;·&nbsp; Cycle {cycle_now}</div>
<div style="font-size:28px;font-weight:900;color:{_mt_col};margin:4px 0 2px;">🎯 Improved RUL: {latest_imp_pred:,.0f} Cycles</div>
<div style="font-size:13.5px;color:{_ds_col};">Feature-engineered Gradient Boosting · <b>45.7% lower prediction error</b> across full fleet</div>
</div>
<div style="display:flex;gap:16px;align-items:center;">
{improved_card_html}
{baseline_card_html}
{true_card_html}
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 1B. EXPLAINABLE AI (XAI) — PER-ENGINE ROOT-CAUSE WATERFALL
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🧠 Explainable AI (XAI) · Per-Engine Root-Cause Attribution</div>
        <div class="panel-sub">Exact tree-path decomposition showing which specific sensors, rolling volatilities, and thermodynamic deltas increased or reduced predicted RUL for Engine #{int(engine_id):02d}.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if imp_model is not None and imp_scaler is not None and imp_feats is not None:
        xc1, xc2 = st.columns([2.2, 1.0])
        with xc1:
            if len(eng_cycles) > 1:
                sel_cyc = st.select_slider(
                    "Inspect Operating Cycle (Scrub Timeline to See How Failure Drivers Evolve)",
                    options=[int(c) for c in eng_cycles],
                    value=int(eng_cycles[-1]),
                    key=f"xai_cycle_slider_{fd_choice}_{int(engine_id)}",
                )
                sel_idx = int(np.searchsorted(eng_cycles, sel_cyc))
            else:
                sel_idx = 0
                sel_cyc = int(eng_cycles[0])
        with xc2:
            xai_top_n = st.selectbox(
                "Top Drivers Displayed",
                options=[8, 10, 14, 18],
                index=1,
                key=f"xai_top_n_{fd_choice}_{int(engine_id)}",
            )

        xai_res = compute_local_rul_attribution(imp_model, imp_scaler, imp_feats, engine_df, row_idx=sel_idx)
        if xai_res is not None:
            col_wf, col_diag = st.columns([1.65, 1.0])
            with col_wf:
                fig_wf = build_xai_waterfall_figure(xai_res, top_n=xai_top_n, dark_mode=_DM)
                st.plotly_chart(fig_wf, use_container_width=True, key=f"xai_wf_{fd_choice}_{int(engine_id)}_{sel_cyc}")
                _clamp_note = " (clamped to ≥ 1.0 cyc floor in UI)" if xai_res["raw_prediction"] < 1.0 else ""
                st.caption(
                    f"Exact Additive Audit · Baseline `{xai_res['base_value']:.2f}` + Net Feature Shift "
                    f"`{xai_res['net_shift']:+.2f}` = `{xai_res['raw_prediction']:.2f}` cyc{_clamp_note} "
                    f"(reconstruction Δ = `{xai_res['recon_error']:.1e}`)"
                )

            with col_diag:
                _card_bg = "#0f172a" if _DM else "#ffffff"
                _card_brd = "#334155" if _DM else "#e2e8f0"
                _txt_main = "#e2e8f0" if _DM else "#0f172a"
                _txt_mut = "#94a3b8" if _DM else "#64748b"

                # Top 3 Degrading Drivers
                deg_df = xai_res["top_degrading"].head(3)
                deg_rows_html = ""
                if not deg_df.empty:
                    for _, r in deg_df.iterrows():
                        deg_rows_html += (
                            f'<div style="display:flex;justify-content:space-between;align-items:center;'
                            f'padding:7px 10px;margin-top:6px;border-radius:8px;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);">'
                            f'<div><div style="font-size:12.5px;font-weight:700;color:{_txt_main};">{r["label"]}</div>'
                            f'<div style="font-size:11px;color:{_txt_mut};">Observed: {r["raw_display"]}</div></div>'
                            f'<div style="font-family:monospace;font-size:13px;font-weight:800;color:#ef4444;">{r["contribution"]:+.1f} cyc</div>'
                            f'</div>'
                        )
                else:
                    deg_rows_html = f'<div style="font-size:12px;color:{_txt_mut};margin-top:6px;">No negative degradation penalties active at Cycle {sel_cyc}.</div>'

                # Top 3 Stabilizing Factors
                stab_df = xai_res["top_stabilizing"].head(3)
                stab_rows_html = ""
                if not stab_df.empty:
                    for _, r in stab_df.iterrows():
                        stab_rows_html += (
                            f'<div style="display:flex;justify-content:space-between;align-items:center;'
                            f'padding:7px 10px;margin-top:6px;border-radius:8px;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);">'
                            f'<div><div style="font-size:12.5px;font-weight:700;color:{_txt_main};">{r["label"]}</div>'
                            f'<div style="font-size:11px;color:{_txt_mut};">Observed: {r["raw_display"]}</div></div>'
                            f'<div style="font-family:monospace;font-size:13px;font-weight:800;color:#10b981;">{r["contribution"]:+.1f} cyc</div>'
                            f'</div>'
                        )
                else:
                    stab_rows_html = f'<div style="font-size:12px;color:{_txt_mut};margin-top:6px;">No positive lifespan buffers active at Cycle {sel_cyc}.</div>'

                # Plain-English Diagnostic Verdict
                net_s = xai_res["net_shift"]
                lead_deg = deg_df.iloc[0]["label"] if not deg_df.empty else "None"
                lead_deg_val = f"{deg_df.iloc[0]['contribution']:+.1f} cyc" if not deg_df.empty else "0.0 cyc"
                lead_stab = stab_df.iloc[0]["label"] if not stab_df.empty else "None"
                lead_stab_val = f"{stab_df.iloc[0]['contribution']:+.1f} cyc" if not stab_df.empty else "0.0 cyc"
                verdict_col = "#10b981" if net_s >= 0 else "#ef4444"
                verdict_word = "above" if net_s >= 0 else "below"

                st.markdown(
                    f"""<div style="background:{_card_bg};border:1px solid {_card_brd};border-radius:12px;padding:14px 16px;">
                    <div style="font-size:11.5px;font-weight:800;color:#38bdf8;text-transform:uppercase;letter-spacing:.06em;">Engine #{int(engine_id):02d} · Cycle {sel_cyc} Attribution</div>
                    <div style="font-size:13px;color:{_txt_main};margin:6px 0 12px;line-height:1.45;">
                        Prediction is <b style="color:{verdict_col};">{abs(net_s):.1f} cycles {verdict_word}</b> the fleet baseline ({xai_res['base_value']:.1f} cyc).
                        Largest RUL penalty is <b>{lead_deg}</b> ({lead_deg_val}); strongest buffer is <b>{lead_stab}</b> ({lead_stab_val}).
                    </div>
                    <div style="font-size:11px;font-weight:800;color:#ef4444;text-transform:uppercase;letter-spacing:.05em;">🔻 Top Degrading Drivers (−RUL)</div>
                    {deg_rows_html}
                    <div style="font-size:11px;font-weight:800;color:#10b981;text-transform:uppercase;letter-spacing:.05em;margin-top:12px;">🔺 Top Stabilizing Signals (+RUL)</div>
                    {stab_rows_html}
                    </div>""",
                    unsafe_allow_html=True,
                )

            with st.expander("🔍 Inspect Full Per-Feature Attribution Ledger (Top 25 Signals)", expanded=False):
                ledger_df = xai_res["df"].head(25)[["label", "feature", "raw_display", "contribution", "direction"]].copy()
                ledger_df.columns = ["Aerospace Signal", "Model Feature", "Observed Value", "RUL Impact (Cycles)", "Effect"]
                ledger_df["RUL Impact (Cycles)"] = ledger_df["RUL Impact (Cycles)"].map(lambda v: f"{v:+.2f}")
                st.dataframe(ledger_df, use_container_width=True, hide_index=True)
    else:
        st.info("Improved Gradient Boosting model artifacts are not loaded; XAI tree-path attribution requires `improved_rul_model.pkl`.")

    # -----------------------------------------------------------------------
    # 2. BENCHMARK COMPARISON KPIS (Baseline vs Improved)
    # -----------------------------------------------------------------------
    st.markdown('<div class="panel-card"><div class="panel-title">📊 Quantitative Model Performance Comparison</div><div class="panel-sub">Evaluated across all 104,897 NASA C-MAPSS operating cycles (FD001–FD004).</div></div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📉 Root Mean Sq Error (RMSE)</div>
            <div class="kpi-value" style="color:#16a34a;">34.66 cyc</div>
            <div class="kpi-sub">Baseline: <span style="color:#dc2626;font-weight:700;">63.83 cyc</span> &nbsp;(<b style="color:#16a34a;">-45.7%</b>)</div>
            </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🎯 Mean Absolute Error (MAE)</div>
            <div class="kpi-value" style="color:#16a34a;">27.10 cyc</div>
            <div class="kpi-sub">Baseline: <span style="color:#dc2626;font-weight:700;">48.15 cyc</span> &nbsp;(<b style="color:#16a34a;">-43.7%</b>)</div>
            </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📈 Explained Variance (R²)</div>
            <div class="kpi-value" style="color:#2563eb;">0.8155</div>
            <div class="kpi-sub">Baseline: <span style="color:#dc2626;font-weight:700;">0.3743</span> &nbsp;(<b style="color:#2563eb;">+0.4412</b>)</div>
            </div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🛡️ NASA Risk Penalty Score</div>
            <div class="kpi-value" style="color:#16a34a;">106.4 K</div>
            <div class="kpi-sub">Baseline: <span style="color:#dc2626;font-weight:700;">19.3 M</span> &nbsp;(<b style="color:#16a34a;">-99.4% safer</b>)</div>
            </div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. INTERACTIVE RUL DEGRADATION CURVES & ERROR RESIDUAL TIMELINE
    # -----------------------------------------------------------------------
    ch1, ch2 = st.columns([1.6, 1.2])

    with ch1:
        st.markdown(
            f"""<div class="panel-card">
            <div class="panel-title">📉 RUL Degradation Trajectory (Engine #{int(engine_id):02d})</div>
            <div class="panel-sub">Ground Truth vs Improved Model vs Baseline Model across all recorded cycles.</div>
            </div>""", unsafe_allow_html=True)

        fig_rul = go.Figure()

        # 80% Conformal Quantile Uncertainty Cone [Q10 .. Q90]
        fig_rul.add_trace(go.Scatter(
            x=eng_cycles, y=imp_hi_engine, mode="lines",
            name="Q90 Upper Bound",
            line=dict(width=0),
            hovertemplate="Cycle: %{x}<br>Upper 90th Pct: %{y:,.1f} cyc<extra></extra>",
            showlegend=False,
        ))
        fig_rul.add_trace(go.Scatter(
            x=eng_cycles, y=imp_lo_engine, mode="lines",
            name="80% Uncertainty Band (Q10–Q90)",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.18)",
            hovertemplate="Cycle: %{x}<br>Lower 10th Pct: %{y:,.1f} cyc<extra></extra>",
        ))

        # Ground Truth line (if available)
        if has_ground_truth:
            fig_rul.add_trace(go.Scatter(
                x=eng_cycles, y=true_rul_arr, mode="lines",
                name="Ground Truth (Actual RUL)",
                line=dict(color="#16a34a", width=3),
                hovertemplate="Cycle: %{x}<br>Ground Truth: %{y:,.1f} cyc<extra></extra>",
            ))

        # Improved Model prediction line
        fig_rul.add_trace(go.Scatter(
            x=eng_cycles, y=imp_preds_engine, mode="lines+markers",
            name="Improved Model (73 Features)",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=4, color="#2563eb"),
            hovertemplate="Cycle: %{x}<br>Improved Prediction: %{y:,.1f} cyc<extra></extra>",
        ))

        # Baseline Model prediction line
        fig_rul.add_trace(go.Scatter(
            x=eng_cycles, y=base_preds_engine, mode="lines",
            name="Baseline Model (6 Features)",
            line=dict(color="#dc2626", width=2, dash="dash"),
            hovertemplate="Cycle: %{x}<br>Baseline Prediction: %{y:,.1f} cyc<extra></extra>",
        ))

        # Thresholds
        fig_rul.add_hline(y=80, line_dash="dot", line_color="#d97706",
                          annotation_text="Warning (80 cyc)", annotation_font_color="#d97706", annotation_font_size=10)
        fig_rul.add_hline(y=30, line_dash="dash", line_color="#dc2626",
                          annotation_text="Critical (30 cyc)", annotation_font_color="#dc2626", annotation_font_size=10)

        fig_rul.update_layout(**styled_layout(
            height=370, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Engine Operating Cycle",
            yaxis_title="Remaining Useful Life (Cycles)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig_rul, use_container_width=True, key="rul_degradation_chart")
        if imp_q_used and isinstance(q_meta, dict):
            st.caption(
                f"Shaded cone = Conformalized 80% Quantile Interval [Q10–Q90] · CQR pad `+{q_meta.get('cqr_margin', 0.0):.2f} cyc` · "
                f"Holdout coverage: `{q_meta.get('empirical_test_coverage', 0.8)*100:.1f}%` (pinball losses: Q10={q_meta.get('pinball_loss_q10', 0):.2f}, Q90={q_meta.get('pinball_loss_q90', 0):.2f})."
            )

    with ch2:
        st.markdown(
            f"""<div class="panel-card">
            <div class="panel-title">🎯 Prediction Residual / Absolute Error</div>
            <div class="panel-sub">|Predicted RUL - Ground Truth| cycle-by-cycle error for Engine #{int(engine_id):02d}.</div>
            </div>""", unsafe_allow_html=True)

        if has_ground_truth:
            err_imp_series = np.abs(imp_preds_engine - true_rul_arr)
            err_base_series = np.abs(base_preds_engine - true_rul_arr)

            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=eng_cycles, y=err_base_series, mode="lines",
                name="Baseline Error",
                line=dict(color="#dc2626", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
                hovertemplate="Cycle: %{x}<br>Baseline Error: %{y:.1f} cyc<extra></extra>",
            ))
            fig_err.add_trace(go.Scatter(
                x=eng_cycles, y=err_imp_series, mode="lines",
                name="Improved Error",
                line=dict(color="#2563eb", width=2.5),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.15)",
                hovertemplate="Cycle: %{x}<br>Improved Error: %{y:.1f} cyc<extra></extra>",
            ))
            fig_err.update_layout(**styled_layout(
                height=370, margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Engine Cycle",
                yaxis_title="Absolute Error (Cycles)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            ))
            st.plotly_chart(fig_err, use_container_width=True, key="rul_error_chart")
        else:
            st.info("Ground Truth RUL is not available for synthetic preview data.")

    # -----------------------------------------------------------------------
    # 4. FEATURE ENGINEERING IMPACT & PERMUTATION IMPORTANCE
    # -----------------------------------------------------------------------
    fe1, fe2 = st.columns([1.5, 1.2])

    with fe1:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">🔬 Top Degradation Feature Importances</div>
            <div class="panel-sub">Normalized relative importance of engineered features in predicting RUL accurately.</div>
            </div>""", unsafe_allow_html=True)

        # Load feature importances if present
        if os.path.exists(FEATURE_IMPORTANCES_PATH):
            imp_df = pd.read_csv(FEATURE_IMPORTANCES_PATH).head(10).sort_values("importance_pct", ascending=True)
            top_feats = imp_df["feature"].tolist()
            top_pcts = imp_df["importance_pct"].tolist()
            top_labels = [SENSOR_LABELS.get(f, f.replace("_roll5_mean", " (5-cyc Mean)").replace("_delta", " (Δ change)")) for f in top_feats]
        else:
            top_labels = ["HPC Coolant Bleed", "Pressure Ratio", "Temp Spread (HPC-LPC)", "Bypass Duct Pressure", "Core Speed Drift", "Rolling HPC Temp", "Corrected Fan Speed", "Bypass Ratio", "Operating Cycle", "Cycle Ratio (Lifespan %)"]
            top_pcts = [1.5, 2.3, 3.1, 4.2, 5.0, 7.8, 9.4, 11.2, 24.5, 31.0]

        fig_imp = go.Figure(go.Bar(
            x=top_pcts,
            y=top_labels,
            orientation="h",
            marker=dict(
                color=top_pcts,
                colorscale="Blues",
                line=dict(color="#1e3a8a", width=1),
            ),
            text=[f"{p:.1f}%" for p in top_pcts],
            textposition="outside",
            hovertemplate="Feature: %{y}<br>Importance: %{x:.2f}%<extra></extra>",
        ))
        fig_imp.update_layout(**styled_layout(
            height=360, margin=dict(l=10, r=20, t=10, b=10),
            xaxis_title="Relative Importance (%)",
            yaxis_title="",
        ))
        st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance_chart")

    with fe2:
        st.markdown(
            """<div class="panel-card">
            <div class="panel-title">💡 Why Feature Engineering Works Here</div>
            <div class="panel-sub">Key physical principles that enabled a 45.7% error reduction:</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size:13.5px;line-height:1.6;">
                <div style="background:#eaf3ff;border:1px solid #cfe4ff;border-radius:10px;padding:12px 14px;margin-bottom:10px;color:#0b3d8a;">
                    <b>⏱️ Operational Cycle Ratio (<code>cycle_ratio</code>)</b><br/>
                    Normalizes cycle count against maximum lifespan, allowing the model to distinguish early wear-in from late-stage thermal exhaustion.
                </div>
                <div style="background:#eef9f2;border:1px solid #c7f0d6;border-radius:10px;padding:12px 14px;margin-bottom:10px;color:#0d592a;">
                    <b>🌀 5-Cycle Rolling Statistics (<code>roll5_mean</code>, <code>std</code>)</b><br/>
                    Averages out random sensor measurement jitter while preserving physical degradation slope and flutter.
                </div>
                <div style="background:#fff8ec;border:1px solid #fde2b4;border-radius:10px;padding:12px 14px;margin-bottom:10px;color:#7a4a00;">
                    <b>🌡️ Thermodynamic Domain Interactions</b><br/>
                    Combines High-Pressure Compressor (HPC) and Low-Pressure Turbine (LPT) thermal spreads & pressure ratios (<code>temp_spread</code>, <code>pressure_ratio</code>).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 5. DETAILED ARCHITECTURE & METRIC BENCHMARK TABLE
    # -----------------------------------------------------------------------
    st.markdown('<div class="panel-card"><div class="panel-title">📋 Comprehensive Model Benchmark Matrix</div><div class="panel-sub">Direct head-to-head comparison between original AeroGuard baseline and improved pipeline.</div></div>', unsafe_allow_html=True)

    if os.path.exists(COMPARISON_METRICS_PATH):
        comp_df = pd.read_csv(COMPARISON_METRICS_PATH)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        comp_table = pd.DataFrame([
            {"Metric": "RMSE (Root Mean Sq Error)", "Baseline Model": "63.83 cycles", "Improved Model": "34.66 cycles", "Improvement": "45.7% lower error", "Verdict": "Improved Model"},
            {"Metric": "MAE (Mean Absolute Error)", "Baseline Model": "48.15 cycles", "Improved Model": "27.10 cycles", "Improvement": "43.7% lower error", "Verdict": "Improved Model"},
            {"Metric": "R² Explained Variance", "Baseline Model": "0.3743", "Improved Model": "0.8155", "Improvement": "+0.4412 higher accuracy", "Verdict": "Improved Model"},
            {"Metric": "NASA Risk Penalty Score", "Baseline Model": "19,321,064", "Improved Model": "106,423", "Improvement": "99.4% safer predictions", "Verdict": "Improved Model"},
            {"Metric": "Feature Dimensionality", "Baseline Model": "6 raw features", "Improved Model": "73 engineered features", "Improvement": "+67 degradation signals", "Verdict": "Improved Model"},
        ])
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # 6. FLEET-WIDE RUL FORECAST & HEALTH RANKING
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🌐 Fleet-Wide RUL Forecast & Maintenance Priority ({fd_choice})</div>
        <div class="panel-sub">RUL predictions across all engines in the selected fleet using the improved model.</div>
        </div>""", unsafe_allow_html=True)

    fleet_summary_rows = []
    for u in sorted(df["unit"].unique()[:20]):
        u_df = df[df["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if u_df.empty:
            continue
        u_last_c = int(u_df["cycle"].iloc[-1])
        u_imp_pred, _ = predict_improved_rul_batch(imp_model, imp_scaler, imp_feats, u_df)
        u_base_pred, _ = predict_baseline_rul_batch(model, scaler, u_df)
        u_imp_val = float(u_imp_pred[-1])
        u_base_val = float(u_base_pred[-1])
        u_true = float(u_df["RUL"].iloc[-1]) if "RUL" in u_df.columns else None

        if u_imp_val < 30:
            prio = "🔴 Immediate Action"
        elif u_imp_val < 80:
            prio = "🟡 Scheduled Soon"
        else:
            prio = "🟢 Normal Cadence"

        fleet_summary_rows.append({
            "Engine": f"Engine #{int(u):02d}",
            "Operating Cycles": u_last_c,
            "Improved Predicted RUL": f"{u_imp_val:.0f} cyc",
            "Baseline Predicted RUL": f"{u_base_val:.0f} cyc",
            "Ground Truth RUL": f"{u_true:.0f} cyc" if u_true is not None else "N/A",
            "Maintenance Priority": prio,
        })

    fleet_sum_df = pd.DataFrame(fleet_summary_rows)
    st.dataframe(fleet_sum_df, use_container_width=True, hide_index=True, height=280)


# ==========================================================================
# PAGE 8 — ENGINE HEALTH SCORE (TASK 4)
# ==========================================================================
elif page == "🩺 Engine Health Score":

    # 1. Compute Health Score for current engine
    health_info = compute_engine_health_score(engine_df, rul_pred)
    total_score = health_info["total_score"]
    tier = health_info["tier"]
    tier_color = health_info["color"]
    tier_bg = health_info["bg_color"]
    tier_icon = health_info["icon"]
    desc = health_info["description"]

    # Theme-adaptive variables
    _dm = st.session_state.get("dark_mode", True)
    banner_bg = "#1e293b" if _dm else tier_bg
    banner_border = tier_color
    banner_txt = "#f8fafc" if _dm else "#0f172a"
    sub_box_bg = "#334155" if _dm else "#ffffff"
    sub_box_txt = "#cbd5e1" if _dm else "#475569"

    # Master Health Score Banner (Zero indentation to avoid markdown code-block bug)
    st.markdown(
f"""<div style="background:{banner_bg};border:2px solid {banner_border};border-radius:16px;padding:22px 26px;margin-bottom:20px;box-shadow:0 4px 18px rgba(0,0,0,0.08);">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
<div>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
<span style="font-size:32px;">{tier_icon}</span>
<span style="font-size:26px;font-weight:900;color:{tier_color};letter-spacing:-0.5px;">{tier.upper()} CONDITION</span>
<span style="background:{tier_color};color:#ffffff;font-size:12px;font-weight:800;padding:4px 14px;border-radius:20px;letter-spacing:0.5px;">TIER: {tier}</span>
</div>
<div style="font-size:14px;color:{banner_txt};max-width:680px;line-height:1.5;">{desc}</div>
</div>
<div style="text-align:right;background:{sub_box_bg};padding:14px 22px;border-radius:14px;border:1px solid {tier_color}44;min-width:170px;">
<div style="font-size:11px;font-weight:700;color:{sub_box_txt};text-transform:uppercase;letter-spacing:0.5px;">Overall Health Score</div>
<div style="font-size:40px;font-weight:900;color:{tier_color};line-height:1.1;">{total_score}<span style="font-size:18px;color:{sub_box_txt};font-weight:600;"> / 100</span></div>
<div style="font-size:12px;color:{tier_color};font-weight:700;">Engine #{int(engine_id):02d} · Cycle {cycle_now}</div>
</div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;margin-top:18px;border-top:1px solid {tier_color}33;padding-top:14px;">
<div style="background:{sub_box_bg};padding:12px 14px;border-radius:10px;border-left:4px solid #2563eb;">
<div style="font-size:11px;color:{sub_box_txt};font-weight:700;">PROGNOSTIC RUL (40% WEIGHT)</div>
<div style="font-size:18px;font-weight:800;color:#2563eb;">{health_info['rul_score']:.1f}<span style="font-size:12px;font-weight:500;">/100</span> <span style="font-size:12px;color:{sub_box_txt};">({health_info['rul_contrib']:.1f} pts)</span></div>
<div style="font-size:11px;color:{sub_box_txt};">Est. {rul_pred:.0f} operating cycles left</div>
</div>
<div style="background:{sub_box_bg};padding:12px 14px;border-radius:10px;border-left:4px solid #7c3aed;">
<div style="font-size:11px;color:{sub_box_txt};font-weight:700;">SENSOR STABILITY (35% WEIGHT)</div>
<div style="font-size:18px;font-weight:800;color:#7c3aed;">{health_info['sensor_score']:.1f}<span style="font-size:12px;font-weight:500;">/100</span> <span style="font-size:12px;color:{sub_box_txt};">({health_info['sensor_contrib']:.1f} pts)</span></div>
<div style="font-size:11px;color:{sub_box_txt};">{health_info['crit_count']} Critical · {health_info['warn_count']} Warning alerts</div>
</div>
<div style="background:{sub_box_bg};padding:12px 14px;border-radius:10px;border-left:4px solid #059669;">
<div style="font-size:11px;color:{sub_box_txt};font-weight:700;">CYCLE FATIGUE LIFE (25% WEIGHT)</div>
<div style="font-size:18px;font-weight:800;color:#059669;">{health_info['cycle_score']:.1f}<span style="font-size:12px;font-weight:500;">/100</span> <span style="font-size:12px;color:{sub_box_txt};">({health_info['cycle_contrib']:.1f} pts)</span></div>
<div style="font-size:11px;color:{sub_box_txt};">Cycle {cycle_now} of ~{cycle_now + max(int(rul_pred), 1)} total lifespan</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # 2. KPI Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🩺 Health Index</div>
            <div class="kpi-value" style="color:{tier_color};">{total_score} <span style="font-size:16px;">/ 100</span></div>
            <div class="kpi-sub">Overall mechanical condition</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🏷️ Health Tier</div>
            <div class="kpi-value" style="color:{tier_color};">{tier_icon} {tier}</div>
            <div class="kpi-sub">Operating status classification</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⏱️ Forecasted RUL</div>
            <div class="kpi-value" style="color:#2563eb;">{rul_pred:,.0f} cyc</div>
            <div class="kpi-sub">Feature-engineered prediction</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⚠️ Active Sensor Alerts</div>
            <div class="kpi-value" style="color:{'#dc2626' if health_info['crit_count'] > 0 else ('#d97706' if health_info['warn_count'] > 0 else '#16a34a')};">
                {health_info['crit_count']} Crit · {health_info['warn_count']} Warn
            </div>
            <div class="kpi-sub">Out of {len(health_info['anomaly_df'])} tracked telemetry channels</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 3. Health Score Trajectory Degradation Curve
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">📉 Engine Health Score Degradation Trajectory</div>
        <div class="panel-sub">Continuous cycle-by-cycle evolution of engine health with shaded 4-tier operational thresholds.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    rul_series_for_traj = rul_imp_arr if imp_used else rul_base_arr
    traj_scores = compute_health_trajectory(engine_df, rul_series_for_traj)
    cycles_arr = engine_df["cycle"].values.astype(int) if "cycle" in engine_df.columns else np.arange(1, len(engine_df) + 1)

    fig_traj = go.Figure()

    # Shaded threshold bands
    max_c = float(cycles_arr[-1])
    fig_traj.add_hrect(y0=80, y1=100, fillcolor="#16a34a", opacity=0.10, line_width=0, annotation_text="Healthy (80–100)", annotation_position="top right")
    fig_traj.add_hrect(y0=60, y1=80, fillcolor="#2563eb", opacity=0.10, line_width=0, annotation_text="Monitor (60–79)", annotation_position="top right")
    fig_traj.add_hrect(y0=40, y1=60, fillcolor="#d97706", opacity=0.12, line_width=0, annotation_text="Warning (40–59)", annotation_position="top right")
    fig_traj.add_hrect(y0=0, y1=40, fillcolor="#dc2626", opacity=0.12, line_width=0, annotation_text="Critical (0–39)", annotation_position="top right")

    # Trajectory curve
    fig_traj.add_trace(go.Scatter(
        x=cycles_arr,
        y=traj_scores,
        mode="lines",
        name="Engine Health Score",
        line=dict(color=tier_color, width=3.5),
        hovertemplate="Cycle: %{x}<br>Health Score: %{y:.1f} / 100<extra></extra>",
    ))

    # Mark current cycle point
    fig_traj.add_trace(go.Scatter(
        x=[cycles_arr[-1]],
        y=[traj_scores[-1]],
        mode="markers+text",
        name="Current Cycle",
        marker=dict(size=14, color=tier_color, line=dict(color="#ffffff", width=2.5)),
        text=[f"  Cycle {cycles_arr[-1]}: {traj_scores[-1]:.1f}"],
        textposition="top right",
        hoverinfo="skip",
    ))

    fig_traj.update_layout(**styled_layout(
        height=380,
        margin=dict(l=10, r=20, t=20, b=10),
        xaxis_title="Operating Cycle",
        yaxis_title="Health Score (0–100)",
        yaxis=dict(range=[0, 105]),
        showlegend=False,
    ))
    st.plotly_chart(fig_traj, use_container_width=True, key="health_trajectory_chart")

    # 4. Subsystem Breakdown & Diagnostics Radar
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">⚙️ Subsystem Health Diagnostics & Polar Assessment</div>
        <div class="panel-sub">Decomposed mechanical condition across primary turbofan component groups.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    sub_col1, sub_col2 = st.columns([1.1, 1.0])
    subsystems_dict = health_info["subsystems"]
    sub_names = list(subsystems_dict.keys())
    sub_scores = list(subsystems_dict.values())

    with sub_col1:
        # Radar Polar Chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=sub_scores + [sub_scores[0]],
            theta=sub_names + [sub_names[0]],
            fill="toself",
            fillcolor=f"{tier_color}33",
            line=dict(color=tier_color, width=2.5),
            marker=dict(size=7, color=tier_color),
            hovertemplate="Subsystem: %{theta}<br>Health Score: %{r:.1f} / 100<extra></extra>",
        ))
        _polar_bg = "#e5ebf3"
        _polar_line = "#cbd5e1"
        _polar_txt = "#0f172a"

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#eef2f6",
                radialaxis=dict(visible=True, range=[0, 100], color=_polar_txt, gridcolor=_polar_line),
                angularaxis=dict(color=_polar_txt, gridcolor=_polar_line),
            ),
            paper_bgcolor=_polar_bg,
            font=dict(color=_polar_txt),
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="subsystem_radar_chart")

    with sub_col2:
        st.markdown(f"<div style='padding-top:10px;'>", unsafe_allow_html=True)
        for s_name, s_val in subsystems_dict.items():
            if s_val >= 80:
                s_col = "#16a34a"
                s_lbl = "Optimal"
            elif s_val >= 60:
                s_col = "#2563eb"
                s_lbl = "Monitoring"
            elif s_val >= 40:
                s_col = "#d97706"
                s_lbl = "Wear Observed"
            else:
                s_col = "#dc2626"
                s_lbl = "High Degradation"

            st.markdown(
f"""<div style="background:{sub_box_bg};border:1px solid {_polar_line};border-radius:10px;padding:12px 16px;margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
<span style="font-weight:700;font-size:13.5px;color:{banner_txt};">{s_name}</span>
<span style="font-weight:800;font-size:13.5px;color:{s_col};">{s_val:.1f} / 100 &nbsp;<span style="font-size:11px;padding:2px 8px;border-radius:12px;background:{s_col}22;">{s_lbl}</span></span>
</div>
<div style="width:100%;background:{_polar_line};height:8px;border-radius:4px;overflow:hidden;">
<div style="width:{min(max(s_val, 0), 100)}%;background:{s_col};height:8px;border-radius:4px;"></div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Sensor Health Audit & Deviations Table
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">🔬 Detailed Sensor Health & Z-Score Deviation Matrix</div>
        <div class="panel-sub">Real-time telemetry readings versus historical normal operating baseline for Engine #""" + f"{int(engine_id):02d}." + """</div>
        </div>""",
        unsafe_allow_html=True,
    )

    anom_df = health_info["anomaly_df"].copy()
    anom_display = pd.DataFrame({
        "Sensor Channel": anom_df["sensor"],
        "Subsystem / Description": anom_df["label"],
        "Current Value": anom_df["current"].apply(lambda v: f"{v:,.2f}"),
        "Normal Mean (μ)": anom_df["mean"].apply(lambda v: f"{v:,.2f}"),
        "Std Dev (σ)": anom_df["std"].apply(lambda v: f"{v:,.3f}"),
        "Z-Score (Dev)": anom_df["z"].apply(lambda v: f"{v:+.2f}σ"),
        "Health Status": anom_df["status"].apply(lambda s: f"🔴 {s}" if s == "Critical" else (f"🟡 {s}" if s == "Warning" else f"🟢 {s}")),
    })
    st.dataframe(anom_display, use_container_width=True, hide_index=True, height=290)

    # 6. Fleet-Wide Health Distribution & Priority Ranking
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🌐 Fleet Health Distribution & Priority Ranking ({fd_choice})</div>
        <div class="panel-sub">Health classification and urgency matrix across all engines in the current fleet dataset.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    fleet_df = get_fleet_health_summary(df, fd_choice)
    if not fleet_df.empty:
        n_healthy = int((fleet_df["tier"] == "Healthy").sum())
        n_monitor = int((fleet_df["tier"] == "Monitor").sum())
        n_warning = int((fleet_df["tier"] == "Warning").sum())
        n_critical = int((fleet_df["tier"] == "Critical").sum())
        total_fleet = len(fleet_df)

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-label">🟢 Healthy Engines</div>
                <div class="kpi-value" style="color:#16a34a;">{n_healthy} <span style="font-size:14px;color:{sub_box_txt};">({n_healthy/total_fleet*100:.0f}%)</span></div>
                <div class="kpi-sub">Score 80 – 100</div>
                </div>""", unsafe_allow_html=True)
        with fc2:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-label">🔵 Monitor Engines</div>
                <div class="kpi-value" style="color:#2563eb;">{n_monitor} <span style="font-size:14px;color:{sub_box_txt};">({n_monitor/total_fleet*100:.0f}%)</span></div>
                <div class="kpi-sub">Score 60 – 79</div>
                </div>""", unsafe_allow_html=True)
        with fc3:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-label">🟡 Warning Engines</div>
                <div class="kpi-value" style="color:#d97706;">{n_warning} <span style="font-size:14px;color:{sub_box_txt};">({n_warning/total_fleet*100:.0f}%)</span></div>
                <div class="kpi-sub">Score 40 – 59</div>
                </div>""", unsafe_allow_html=True)
        with fc4:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-label">🔴 Critical Engines</div>
                <div class="kpi-value" style="color:#dc2626;">{n_critical} <span style="font-size:14px;color:{sub_box_txt};">({n_critical/total_fleet*100:.0f}%)</span></div>
                <div class="kpi-sub">Score 0 – 39</div>
                </div>""", unsafe_allow_html=True)

        # Fleet Ranking Table (Sorted by lowest health score first to highlight risk)
        fleet_sorted = fleet_df.sort_values("health_score", ascending=True).reset_index(drop=True)
        fleet_table_display = pd.DataFrame({
            "Engine": fleet_sorted["engine"],
            "Health Score": fleet_sorted["health_score"].apply(lambda s: f"{s:.1f} / 100"),
            "Classification Tier": fleet_sorted.apply(lambda r: f"{r['icon']} {r['tier']}", axis=1),
            "Current Cycle": fleet_sorted["cycle"].apply(lambda c: f"{c} cyc"),
            "Forecasted RUL": fleet_sorted["rul"].apply(lambda r: f"{r:.0f} cyc"),
            "Sensor Stability": fleet_sorted["sensor_score"].apply(lambda s: f"{s:.1f} / 100"),
            "RUL Prognostic": fleet_sorted["rul_score"].apply(lambda r: f"{r:.1f} / 100"),
        })
        st.dataframe(fleet_table_display, use_container_width=True, hide_index=True, height=310)

    # 7. Actionable Maintenance & Operational Protocol
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">📋 Prescriptive Maintenance & Flight Operations Protocol</div>
        <div class="panel-sub">Standard Operating Procedures (SOP) tailored to current classification tier.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if tier == "Healthy":
        action_title = "🟢 Standard Flight Readiness Protocol"
        action_items = [
            "Maintain standard line-maintenance cycle intervals and automated health telemetry logging.",
            "Continue standard flight operations within unrestricted flight envelopes.",
            "Next scheduled borescope and hot-section inspection at standard 100-cycle overhaul interval.",
            "Verify all sensor calibrations match OEM nominal thresholds during regular scheduled downtime.",
        ]
        action_border = "#16a34a"
        action_bg = "#f0fdf4" if not _dm else "#1e293b"
    elif tier == "Monitor":
        action_title = "🔵 Enhanced Observational Protocol"
        action_items = [
            "Increase high-frequency sensor logging frequency from 1 Hz to 5 Hz on next flight leg.",
            "Flag engine for routine oil analysis and magnetic chip detector inspection within 10 cycles.",
            "Cross-reference compressor bleed valve and turbine outlet temperature logs for slight thermal creep.",
            "Notify maintenance operations control (MOC) of mild sensor drift; no immediate flight grounding required.",
        ]
        action_border = "#2563eb"
        action_bg = "#eff6ff" if not _dm else "#1e293b"
    elif tier == "Warning":
        action_title = "🟡 Pre-emptive Maintenance & Inspection Warning"
        action_items = [
            "Schedule focused borescope inspection of high-pressure turbine (HPT) blades within the next 15–20 cycles.",
            "Restrict extended twin-engine operational performance standards (ETOPS) routing if applicable.",
            "Prepare spare component staging (combustion nozzles, fuel flow meters, turbine shroud liners).",
            "Perform engine ground-run vibration survey to isolate acoustic and rotor spool anomalies.",
        ]
        action_border = "#d97706"
        action_bg = "#fffbeb" if not _dm else "#1e293b"
    else:  # Critical
        action_title = "🔴 Immediate Grounding & Overhaul Directive"
        action_items = [
            "GROUND ENGINE IMMEDIATELY: Halt flight dispatch and disconnect high-pressure bleed lines.",
            "Initiate priority teardown and borescope examination for thermal fatigue cracking or rotor imbalance.",
            "Coordinate with propulsion engineering to perform complete engine exchange with overhaul spare.",
            "Download full flight data recorder (FDR) telemetry packet for root-cause failure mode assessment.",
        ]
        action_border = "#dc2626"
        action_bg = "#fef2f2" if not _dm else "#1e293b"

    items_html = "".join([f"<li style='margin-bottom:8px;line-height:1.5;'>{it}</li>" for it in action_items])
    st.markdown(
f"""<div style="background:{action_bg};border-left:5px solid {action_border};border-radius:10px;padding:18px 22px;margin-bottom:15px;">
<div style="font-size:16px;font-weight:800;color:{action_border};margin-bottom:10px;">{action_title}</div>
<ul style="margin:0;padding-left:20px;font-size:14px;color:{banner_txt};">
{items_html}
</ul>
</div>""",
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE 9 — REAL-TIME ENGINE MONITORING (TASK 5)
# ==========================================================================
elif page == "📡 Real-Time Monitoring":

    max_cycles = len(engine_df)

    # Initialize streaming session states
    if "stream_engine_id" not in st.session_state or st.session_state.stream_engine_id != engine_id:
        st.session_state.stream_engine_id = engine_id
        st.session_state.stream_cycle = 1
        st.session_state.is_streaming = False

    if "stream_cycle" not in st.session_state:
        st.session_state.stream_cycle = 1
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False

    # Clamp cycle
    st.session_state.stream_cycle = max(1, min(int(st.session_state.stream_cycle), max_cycles))

    # Precompute fast batch arrays for the selected engine
    all_imp_preds, all_imp_lo, all_imp_hi, _, _ = predict_improved_rul_intervals_batch(
        imp_model, imp_scaler, imp_feats, q10_model, q90_model, q_meta, engine_df
    )
    all_traj_scores = compute_health_trajectory(engine_df, all_imp_preds)
    all_cycles = engine_df["cycle"].values.astype(int) if "cycle" in engine_df.columns else np.arange(1, max_cycles + 1)

    _dm = st.session_state.get("dark_mode", True)
    banner_txt = "#f8fafc" if _dm else "#0f172a"
    sub_box_bg = "#334155" if _dm else "#ffffff"
    sub_box_txt = "#cbd5e1" if _dm else "#475569"

    # -----------------------------------------------------------------------
    # 1. STREAM SIMULATION CONTROL CONSOLE
    # -----------------------------------------------------------------------
    st.markdown(
        """<div class="panel-card">
        <div class="panel-title">📡 Real-Time Telemetry Stream Controller</div>
        <div class="panel-sub">Simulate a live aircraft flight data stream cycle-by-cycle. Play automatically or scrub manually through operating cycles.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.2, 1, 1, 1, 1.2, 1.8])

    with ctrl_col1:
        if st.session_state.is_streaming:
            if st.button("⏸ Pause Stream", key="btn_pause_stream", use_container_width=True):
                st.session_state.is_streaming = False
                st.rerun()
        else:
            if st.button("▶ Play Stream", key="btn_play_stream", use_container_width=True):
                if st.session_state.stream_cycle >= max_cycles:
                    st.session_state.stream_cycle = 1
                st.session_state.is_streaming = True
                st.rerun()

    with ctrl_col2:
        if st.button("⏮ Step -1", key="btn_step_back", use_container_width=True):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = max(1, st.session_state.stream_cycle - 1)
            st.rerun()

    with ctrl_col3:
        if st.button("⏭ Step +1", key="btn_step_fwd", use_container_width=True):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = min(max_cycles, st.session_state.stream_cycle + 1)
            st.rerun()

    with ctrl_col4:
        if st.button("🔄 Reset", key="btn_reset_stream", use_container_width=True):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = 1
            st.rerun()

    with ctrl_col5:
        speed_opt = st.selectbox(
            "Stream Speed",
            ["Fast (0.2s)", "Normal (0.5s)", "Slow (1.0s)"],
            index=1,
            label_visibility="collapsed",
            key="speed_select",
        )
        delay_map = {"Fast (0.2s)": 0.2, "Normal (0.5s)": 0.5, "Slow (1.0s)": 1.0}
        stream_delay = delay_map[speed_opt]

    with ctrl_col6:
        if st.session_state.is_streaming:
            st.markdown(
                f"""<div style="background:#f0fdf4;border:1px solid #16a34a;color:#16a34a;border-radius:8px;padding:9px 12px;text-align:center;font-weight:800;font-size:12.5px;">
                🟢 LIVE STREAMING · Cycle {st.session_state.stream_cycle} / {max_cycles}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style="background:#eff6ff;border:1px solid #3b82f6;color:#1e40af;border-radius:8px;padding:9px 12px;text-align:center;font-weight:800;font-size:12.5px;">
                ⏸ STREAM PAUSED · Cycle {st.session_state.stream_cycle} / {max_cycles}
                </div>""", unsafe_allow_html=True)

    # Scrubber Slider
    _rt_lbl_col = "#9ca3af" if _dm else "#6b7280"
    st.markdown(f'<div style="color:{_rt_lbl_col};font-size:13.5px;font-weight:700;margin:8px 0 2px;">Flight Cycle Scrubber</div>', unsafe_allow_html=True)
    scrub_val = st.slider(
        "Flight Cycle Scrubber",
        min_value=1,
        max_value=max_cycles,
        value=int(st.session_state.stream_cycle),
        key="stream_cycle_slider",
        label_visibility="collapsed",
        help="Drag to scrub directly to any operating cycle in the engine lifespan.",
    )
    if scrub_val != st.session_state.stream_cycle and not st.session_state.is_streaming:
        st.session_state.stream_cycle = scrub_val
        st.rerun()

    # Quick Jump Shortcuts
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("🟢 Start of Service (Cycle 1)", use_container_width=True, key="qj_start"):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = 1
            st.rerun()
    with q2:
        mid_c = max(1, int(max_cycles * 0.45))
        if st.button(f"🔵 Mid-Life (Cycle {mid_c})", use_container_width=True, key="qj_mid"):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = mid_c
            st.rerun()
    with q3:
        warn_c = max(1, int(max_cycles * 0.75))
        if st.button(f"🟡 Degradation Phase (Cycle {warn_c})", use_container_width=True, key="qj_warn"):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = warn_c
            st.rerun()
    with q4:
        crit_c = max(1, int(max_cycles * 0.95))
        if st.button(f"🔴 Near End-of-Life (Cycle {crit_c})", use_container_width=True, key="qj_crit"):
            st.session_state.is_streaming = False
            st.session_state.stream_cycle = crit_c
            st.rerun()

    # -----------------------------------------------------------------------
    # 2. CURRENT STREAM STATE COMPUTATION
    # -----------------------------------------------------------------------
    cur_c = int(st.session_state.stream_cycle)
    cur_sub_df = engine_df.iloc[:cur_c]
    cur_row = cur_sub_df.iloc[-1]
    prev_row = cur_sub_df.iloc[-2] if len(cur_sub_df) > 1 else cur_row

    cur_rul = float(all_imp_preds[cur_c - 1])
    cur_lo = float(all_imp_lo[cur_c - 1])
    cur_hi = float(all_imp_hi[cur_c - 1])
    cur_health_score = float(all_traj_scores[cur_c - 1])
    cur_health_info = compute_engine_health_score(cur_sub_df, cur_rul)
    cur_tier = cur_health_info["tier"]
    cur_tier_color = cur_health_info["color"]
    cur_tier_icon = cur_health_info["icon"]

    # Risk Level classification
    if cur_rul < 30 or cur_health_score < 40:
        cur_risk_lbl = "CRITICAL RISK"
        cur_risk_col = "#dc2626"
        cur_risk_icon = "🔴"
        cur_risk_desc = "Imminent failure danger · Urgent action required"
    elif cur_rul < 75 or cur_health_score < 60:
        cur_risk_lbl = "WARNING RISK"
        cur_risk_col = "#d97706"
        cur_risk_icon = "🟡"
        cur_risk_desc = "Accelerated component wear · Plan inspection"
    elif cur_rul < 120 or cur_health_score < 80:
        cur_risk_lbl = "MODERATE RISK"
        cur_risk_col = "#2563eb"
        cur_risk_icon = "🔵"
        cur_risk_desc = "Mild operational drift · Elevated logging"
    else:
        cur_risk_lbl = "LOW / NORMAL RISK"
        cur_risk_col = "#16a34a"
        cur_risk_icon = "🟢"
        cur_risk_desc = "All parameters nominal · Flight cleared"

    # Maintenance Alert Message
    if cur_tier == "Critical":
        alert_title = f"🚨 CRITICAL ENGINE ALERT — FLIGHT CYCLE {cur_c}"
        alert_msg = f"Turbofan health collapsed to {cur_health_score:.1f}/100. Predicted Remaining Useful Life is depleted to {cur_rul:.0f} cycles. IMMEDIATE GROUNDING & OVERHAUL MANDATED."
        alert_bg = "#fef2f2" if not _dm else "#1e293b"
        alert_col = "#dc2626"
    elif cur_tier == "Warning":
        alert_title = f"⚠️ PRE-EMPTIVE MAINTENANCE WARNING — FLIGHT CYCLE {cur_c}"
        alert_msg = f"Noticeable thermodynamic and sensor drift detected. RUL decreased to {cur_rul:.0f} cycles. Schedule borescope inspection and hot-section staging within 15–20 cycles."
        alert_bg = "#fffbeb" if not _dm else "#1e293b"
        alert_col = "#d97706"
    elif cur_tier == "Monitor":
        alert_title = f"ℹ️ OPERATIONAL MONITORING NOTICE — FLIGHT CYCLE {cur_c}"
        alert_msg = f"Mild sensor fluctuation observed. Engine health index is {cur_health_score:.1f}/100. Turbofan cleared for flight with enhanced high-frequency telemetry logging."
        alert_bg = "#eff6ff" if not _dm else "#1e293b"
        alert_col = "#2563eb"
    else:
        alert_title = f"✅ ALL FLIGHT SYSTEMS OPTIMAL — FLIGHT CYCLE {cur_c}"
        alert_msg = f"All sensor channels operating within nominal envelopes. Overall health is {cur_health_score:.1f}/100 with {cur_rul:.0f} operating cycles remaining."
        alert_bg = "#f0fdf4" if not _dm else "#1e293b"
        alert_col = "#16a34a"

    # Real-Time Live Alert Banner (Zero indentation)
    st.markdown(
f"""<div style="background:{alert_bg};border:2px solid {alert_col};border-radius:14px;padding:18px 22px;margin:18px 0;box-shadow:0 4px 14px rgba(0,0,0,0.06);">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
<div>
<div style="font-size:16px;font-weight:900;color:{alert_col};margin-bottom:4px;">{alert_title}</div>
<div style="font-size:13.5px;color:{banner_txt};line-height:1.5;">{alert_msg}</div>
</div>
<div style="text-align:right;background:{sub_box_bg};padding:8px 16px;border-radius:10px;border:1px solid {alert_col}44;">
<span style="font-size:12px;font-weight:700;color:{sub_box_txt};">Telemetry Source:</span>
<span style="font-size:13px;font-weight:800;color:{alert_col};margin-left:4px;">Live Stream Feed #{int(engine_id):02d}</span>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 3. LIVE KPI STREAM CARDS
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">📡 Live Flight Cycle</div>
            <div class="kpi-value" style="color:#2563eb;">{cur_c} <span style="font-size:16px;color:{sub_box_txt};">/ {max_cycles}</span></div>
            <div class="kpi-sub">{(cur_c/max_cycles*100):.1f}% of recorded lifespan elapsed</div>
            </div>""", unsafe_allow_html=True)
    with k2:
        rul_delta = cur_rul - float(all_imp_preds[0])
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⏱️ Live Predicted RUL</div>
            <div class="kpi-value" style="color:#2563eb;">{cur_rul:,.0f} cyc</div>
            <div class="kpi-sub">80% CI: <b>{cur_lo:.0f}–{cur_hi:.0f}</b> &nbsp;·&nbsp; Δ <b style="color:{'#dc2626' if rul_delta < 0 else '#16a34a'};">{rul_delta:+.0f}</b></div>
            </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">🩺 Live Health Score</div>
            <div class="kpi-value" style="color:{cur_tier_color};">{cur_health_score:.1f} <span style="font-size:16px;color:{sub_box_txt};">/ 100</span></div>
            <div class="kpi-sub">Condition: <b style="color:{cur_tier_color};">{cur_tier_icon} {cur_tier}</b></div>
            </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">⚠️ Live Risk Status</div>
            <div class="kpi-value" style="font-size:22px;color:{cur_risk_col};">{cur_risk_icon} {cur_risk_lbl.split()[0]}</div>
            <div class="kpi-sub">{cur_risk_desc}</div>
            </div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 4. LIVE TELEMETRY STREAM OSCILLOSCOPE (Multi-Sensor Readings)
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">📈 Real-Time Multi-Sensor Telemetry Stream (Cycles 1 → {cur_c})</div>
        <div class="panel-sub">Continuous flight sensor telemetry readings actively feeding into AeroGuard prognostic intelligence models.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    fig_stream = go.Figure()

    # Core sensor traces up to current cycle
    stream_cycles_x = all_cycles[:cur_c]

    fig_stream.add_trace(go.Scatter(
        x=stream_cycles_x,
        y=cur_sub_df[hpc_col],
        name=f"HPC Temp ({hpc_col})",
        line=dict(color="#ef4444", width=2.5),
        mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f} °C<extra>HPC Temp</extra>",
    ))
    fig_stream.add_trace(go.Scatter(
        x=stream_cycles_x,
        y=cur_sub_df[lpc_col],
        name=f"LPC Temp ({lpc_col})",
        line=dict(color="#f97316", width=2),
        mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f} °C<extra>LPC Temp</extra>",
    ))
    fig_stream.add_trace(go.Scatter(
        x=stream_cycles_x,
        y=cur_sub_df[press_col],
        name=f"Combustor Pressure ({press_col})",
        line=dict(color="#3b82f6", width=2),
        mode="lines",
        yaxis="y2",
        hovertemplate="Cycle %{x}: %{y:.2f} psia<extra>Pressure</extra>",
    ))
    fig_stream.add_trace(go.Scatter(
        x=stream_cycles_x,
        y=cur_sub_df[rpm_col],
        name=f"Core Speed ({rpm_col})",
        line=dict(color="#8b5cf6", width=2),
        mode="lines",
        yaxis="y3",
        hovertemplate="Cycle %{x}: %{y:.1f} rpm<extra>RPM</extra>",
    ))

    # Mark active head point
    fig_stream.add_trace(go.Scatter(
        x=[cur_c],
        y=[float(cur_row[hpc_col])],
        mode="markers",
        name="Telemetry Head",
        marker=dict(size=14, color="#ef4444", line=dict(color="#ffffff", width=2.5)),
        hoverinfo="skip",
    ))

    fig_stream.update_layout(**styled_layout(
        height=380,
        margin=dict(l=10, r=40, t=20, b=10),
        xaxis=dict(title="Operating Cycle", range=[1, max_cycles]),
        yaxis=dict(title="Temperature (°C)", title_font=dict(color="#ef4444")),
        yaxis2=dict(title="Pressure (psia)", title_font=dict(color="#3b82f6"), overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Core Speed (rpm)", title_font=dict(color="#8b5cf6"), overlaying="y", side="right", position=0.95, showgrid=False, visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    st.plotly_chart(fig_stream, use_container_width=True, key="realtime_stream_chart")

    # -----------------------------------------------------------------------
    # 5. DUAL OSCILLOSCOPE: LIVE RUL DECAY & HEALTH SCORE TRAJECTORY
    # -----------------------------------------------------------------------
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown(
            f"""<div class="panel-card">
            <div class="panel-title">⏱️ Real-Time Predicted RUL Decay</div>
            <div class="panel-sub">Continuous prognostic forecast as flight hours accumulate.</div>
            </div>""", unsafe_allow_html=True)

        fig_rul_live = go.Figure()
        fig_rul_live.add_trace(go.Scatter(
            x=stream_cycles_x,
            y=all_imp_hi[:cur_c],
            mode="lines",
            name="Q90 Bound",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_rul_live.add_trace(go.Scatter(
            x=stream_cycles_x,
            y=all_imp_lo[:cur_c],
            mode="lines",
            name="80% Uncertainty",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.18)",
            hovertemplate="Cycle %{x} 80% CI lower: %{y:.0f} cyc<extra></extra>",
        ))
        fig_rul_live.add_trace(go.Scatter(
            x=stream_cycles_x,
            y=all_imp_preds[:cur_c],
            mode="lines",
            name="Predicted RUL",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Cycle %{x}: %{y:.0f} cycles<extra>Predicted</extra>",
        ))
        if "RUL" in engine_df.columns:
            fig_rul_live.add_trace(go.Scatter(
                x=stream_cycles_x,
                y=engine_df["RUL"].values[:cur_c],
                mode="lines",
                name="Ground Truth RUL",
                line=dict(color="#10b981", width=2, dash="dash"),
                hovertemplate="Cycle %{x}: %{y:.0f} cycles<extra>Ground Truth</extra>",
            ))
        fig_rul_live.add_trace(go.Scatter(
            x=[cur_c],
            y=[cur_rul],
            mode="markers+text",
            name="Current RUL",
            marker=dict(size=12, color="#2563eb", line=dict(color="#ffffff", width=2)),
            text=[f"  {cur_rul:.0f} cyc"],
            textposition="top right",
            textfont=dict(color="#0f172a"),
            cliponaxis=False,
            hoverinfo="skip",
        ))
        fig_rul_live.update_layout(**styled_layout(
            height=300,
            margin=dict(l=14, r=18, t=28, b=14),
            xaxis=dict(title="Operating Cycle", range=[1, max_cycles]),
            yaxis=dict(title="Remaining Cycles"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig_rul_live, use_container_width=True, key="live_rul_decay_chart")

    with d_col2:
        st.markdown(
            f"""<div class="panel-card">
            <div class="panel-title">🩺 Real-Time Health Index Trajectory</div>
            <div class="panel-sub">Multi-factor prognostic health score across degradation zones.</div>
            </div>""", unsafe_allow_html=True)

        fig_hlth_live = go.Figure()
        fig_hlth_live.add_hrect(y0=80, y1=100, fillcolor="#16a34a", opacity=0.10, line_width=0)
        fig_hlth_live.add_hrect(y0=60, y1=80, fillcolor="#2563eb", opacity=0.10, line_width=0)
        fig_hlth_live.add_hrect(y0=40, y1=60, fillcolor="#d97706", opacity=0.12, line_width=0)
        fig_hlth_live.add_hrect(y0=0, y1=40, fillcolor="#dc2626", opacity=0.12, line_width=0)

        fig_hlth_live.add_trace(go.Scatter(
            x=stream_cycles_x,
            y=all_traj_scores[:cur_c],
            mode="lines",
            name="Health Index",
            line=dict(color=cur_tier_color, width=3),
            hovertemplate="Cycle %{x}: %{y:.1f} / 100<extra>Health</extra>",
        ))
        fig_hlth_live.add_trace(go.Scatter(
            x=[cur_c],
            y=[cur_health_score],
            mode="markers+text",
            name="Current Health",
            marker=dict(size=12, color=cur_tier_color, line=dict(color="#ffffff", width=2)),
            text=[f"  {cur_health_score:.1f}"],
            textposition="top right",
            textfont=dict(color="#0f172a"),
            cliponaxis=False,
            hoverinfo="skip",
        ))
        fig_hlth_live.update_layout(**styled_layout(
            height=300,
            margin=dict(l=14, r=18, t=28, b=14),
            xaxis=dict(title="Operating Cycle", range=[1, max_cycles]),
            yaxis=dict(title="Health Score (0–100)", range=[0, 112]),
            showlegend=False,
        ))
        st.plotly_chart(fig_hlth_live, use_container_width=True, key="live_health_trajectory_chart")

    # -----------------------------------------------------------------------
    # 6. REAL-TIME SENSOR TELEMETRY READOUT MATRIX
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">🔬 Live Sensor Telemetry Matrix (Cycle {cur_c})</div>
        <div class="panel-sub">Instantaneous telemetry packet values and delta deviations relative to preceding cycle.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    sensor_cols_rt = [c for c in engine_df.columns if c.startswith("sensor_")]
    telemetry_rows = []
    for c in sensor_cols_rt:
        s_all = engine_df[c]
        v_now = float(cur_row[c])
        v_prev = float(prev_row[c])
        v_delta = v_now - v_prev
        z_v = (v_now - s_all.mean()) / max(s_all.std(), 1e-9)
        abs_z = abs(z_v)

        if abs_z > 2.5:
            st_sens = "🔴 Critical"
        elif abs_z > 1.5:
            st_sens = "🟡 Warning"
        else:
            st_sens = "🟢 Normal"

        telemetry_rows.append({
            "Sensor Channel": c,
            "Description": SENSOR_LABELS.get(c, c),
            "Current Reading": f"{v_now:,.2f}",
            "Delta (Δ vs Last)": f"{v_delta:+,.2f}",
            "Z-Score Deviation": f"{z_v:+.2f}σ",
            "Telemetry Status": st_sens,
        })

    rt_sens_df = pd.DataFrame(telemetry_rows)
    st.dataframe(rt_sens_df, use_container_width=True, hide_index=True, height=270)

    # -----------------------------------------------------------------------
    # 7. AUTOMATED REAL-TIME MAINTENANCE ALARMS & EVENT STREAM LOG
    # -----------------------------------------------------------------------
    st.markdown(
        f"""<div class="panel-card">
        <div class="panel-title">📜 Automated Real-Time Maintenance Event Stream Log</div>
        <div class="panel-sub">Chronological ledger of operational transitions, critical sensor spikes, and prognostic triggers up to Cycle {cur_c}.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Construct event stream up to cur_c
    event_logs = []
    # Event 1: Initial clearance
    event_logs.append({
        "Cycle": 1,
        "Timestamp": "00:00:00",
        "Severity": "🟢 Info",
        "Event Description": "Turbofan engine operational dispatch. All primary systems nominal.",
        "Maintenance Action": "Line clearance granted.",
    })

    prev_t = "Healthy"
    for i in range(1, cur_c):
        c_i = int(all_cycles[i])
        h_i = float(all_traj_scores[i])
        r_i = float(all_imp_preds[i])

        # Check tier transition
        if h_i >= 80:
            now_t = "Healthy"
        elif h_i >= 60:
            now_t = "Monitor"
        elif h_i >= 40:
            now_t = "Warning"
        else:
            now_t = "Critical"

        if now_t != prev_t:
            if now_t == "Monitor":
                event_logs.append({
                    "Cycle": c_i,
                    "Timestamp": f"{c_i//60:02d}:{(c_i*3)%60:02d}:14",
                    "Severity": "🔵 Advisory",
                    "Event Description": f"Engine transitioned to MONITOR tier (Score: {h_i:.1f}/100).",
                    "Maintenance Action": "Increase telemetry logging frequency to 5 Hz.",
                })
            elif now_t == "Warning":
                event_logs.append({
                    "Cycle": c_i,
                    "Timestamp": f"{c_i//60:02d}:{(c_i*3)%60:02d}:48",
                    "Severity": "🟡 Warning",
                    "Event Description": f"Engine entered WARNING degradation phase (Score: {h_i:.1f}/100, RUL: {r_i:.0f} cyc).",
                    "Maintenance Action": "Schedule borescope inspection within 20 cycles.",
                })
            elif now_t == "Critical":
                event_logs.append({
                    "Cycle": c_i,
                    "Timestamp": f"{c_i//60:02d}:{(c_i*3)%60:02d}:22",
                    "Severity": "🔴 Critical",
                    "Event Description": f"CRITICAL HEALTH COLLAPSE DETECTED (Score: {h_i:.1f}/100, RUL: {r_i:.0f} cyc).",
                    "Maintenance Action": "GROUND ENGINE: Immediate overhaul and blade teardown mandated.",
                })
            prev_t = now_t

    # Sort reverse chronological so latest event is at the top
    events_display_df = pd.DataFrame(event_logs).sort_values("Cycle", ascending=False).reset_index(drop=True)
    st.dataframe(events_display_df, use_container_width=True, hide_index=True, height=220)

    # -----------------------------------------------------------------------
    # 8. STREAMING AUTOMATION PLAYBACK LOOP
    # -----------------------------------------------------------------------
    if st.session_state.is_streaming:
        if cur_c < max_cycles:
            st.session_state.stream_cycle = cur_c + 1
            time.sleep(stream_delay)
            st.rerun()
        else:
            st.session_state.is_streaming = False
            st.success("✅ Flight stream completed: Engine reached full operating lifecycle.")




st.caption(
    f"AeroGuard · Safer Skies, Smarter Engines · Dataset: {fd_choice} · "
    f"Model: {'Improved HistGradientBoosting (R²=0.816)' if imp_model is not None else ('Loaded Baseline' if model is not None else 'Heuristic RUL')}"
)