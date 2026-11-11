# AeroGuard ✈️
### Real-Time Aircraft Engine Intelligence & Predictive Maintenance System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeroguard-aircraftenginehealthpredictionsystem-daymlhnceoegsjv.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/NASA-C--MAPSS%20Turbofan-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **AeroGuard** is an aerospace-grade telemetry and predictive maintenance command center that monitors aircraft turbofan engine health, detects early-stage mechanical anomalies, predicts Remaining Useful Life (RUL) with high precision, and automates FAA/EASA-aligned maintenance work orders.

🌐 **Live Dashboard**: [Launch AeroGuard on Streamlit Cloud](https://aeroguard-aircraftenginehealthpredictionsystem-daymlhnceoegsjv.streamlit.app/)

---

## 📌 Executive Overview

Modern commercial aircraft turbofan engines operate under extreme thermal, aerodynamic, and mechanical stress. Unplanned engine downtime disrupts flight schedules and presents severe safety hazards. **AeroGuard** bridges the gap between raw flight data recording (FDR) telemetry and operational maintenance dispatching by combining **feature-engineered machine learning models** with an **aerospace mission control interface**.

```
  [ Flight FDR Telemetry / Custom Upload ] ──► [ 73 Engineered Features ] ──► [ Gradient Boosting + CQR Q10/Q90 ]
                                                                                           │
  [ Multi-Channel Sensors ] ─────────────────► [ Rolling Z-Score Analytics ]               ▼
                                                                           [ Predicted RUL ± 80% Bounds + XAI ]
                                                                                           │
                                                                                           ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │                            AeroGuard Mission Control Console                           │
   ├────────────────────────────────────────────────────────────────────────────────────────┤
   │  • Failure Risk Classifier   • Multi-Factor Engine Health Score (0-100)                │
   │  • Turbofan Digital Twin     • Simulated Real-Time Sensor Stream                       │
   │  • XAI Waterfall Attribution • Custom Unseen FDR Ingestion (.csv/.txt/.tsv)            │
   │  • Condition Stress-Tester   • Automated Maintenance Decision SOPs                     │
   │  • Avionics Hotkey Nav (1-9) • Wide-Screen Telemetry Focus (Spacebar)                  │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Recent Platform Upgrades

### 1. 🧠 Explainable AI (XAI) — Per-Engine Root-Cause Attribution
* **Exact Tree-Path Decomposition (Saabas)**: Decomposes every individual engine's predicted RUL into $\text{base\_value} + \sum \text{feature contributions}$ across all 73 engineered telemetry features with machine-precision reconciliation ($\text{error} < 10^{-12}$).
* **Interactive Root-Cause Waterfall & Ledger**: Highlights top degrading (`−RUL`) and stabilizing (`+RUL`) thermodynamic signals alongside an inspectable 25-feature attribution ledger in **🎯 RUL Improvement**.

### 2. 📐 Conformalized 80% Quantile RUL Uncertainty Bounds (`[Q10, Q90]`)
* **Split Conformalized Quantile Regression (CQR)**: Trains dual quantile regressors (`alpha = 0.10` / `0.90`) calibrated with a CQR safety margin (`+0.0208` cycles) to deliver guaranteed **`80%` prediction intervals** (`79.93%` empirical test coverage, `109.66` cycles mean interval width).
* **End-to-End Uncertainty Awareness**: Enforces monotonic non-crossing intervals (`Q10 <= Point <= Q90`) rendered as shaded confidence ribbons across **Dashboard**, **RUL Improvement**, **Real-Time Monitoring**, and **Predictive Maintenance**.

### 3. 📤 Custom FDR Telemetry Uploader (Unseen Data Ingestion)
* **Universal NASA C-MAPSS Parser**: Accepts custom `.csv`, `.txt`, and `.tsv` flight logs in raw headerless space/comma format (`>= 6` columns, optional `RUL`), descriptive header CSVs, or AeroGuard-exported CSVs, with strict schema validation and `< 5`-cycle `NaN` protection across anomaly heatmaps and rolling Z-scores.
* **Live Switch & Revert Workflow**: Accessible both from the **Sidebar** (positioned directly below **Engine Unit Designation**) and **🗄️ Sensor Data** (including a downloadable sample CSV template, `data/sample_unseen_fdr_test.csv` with unseen engines `#101` & `#204`, and a one-click **Revert to Bundled Fleet** button).

### 4. 📦 Version-Independent Portable NumPy Bundles (`.npz`) & Cockpit UX Polish
* **Pure-NumPy Tree Inference**: Serializes ensemble scalers and decision trees into lightweight `.npz` artifacts (`16.9–35.3 KB`) loaded without `scikit-learn`/`numpy` pickle ABI coupling, ensuring seamless Streamlit Cloud execution for RUL predictions, `Q10`/`Q90` intervals, and XAI attribution.
* **High-Contrast Telemetry Canvas & Avionics HUD**: Soft whitish-grey chart canvases (`#eef2f6` / `#e5ebf3`) with crisp dark-slate typography and deep-merged legible legends, slate-grey widget/expander labels, an unclipped sidebar drawer toggle, and a directly clickable `[?] KEYMAP` HUD pill.

---

## 🚀 Key Functional Modules

| Module | Core Capability | Operational Output |
| :--- | :--- | :--- |
| **1. 🏠 Dashboard & Fleet Overview** | Fleet-wide RUL, Health Index, and CQR uncertainty triage | Highlights at-risk engines, live `Q10–Q90` envelopes, and sensor trends for selected or uploaded units. |
| **2. 📈 Engine Monitoring** | Single-engine sensor drill-down & multi-channel comparison | Inspects raw vs. normalized curves across individual or comparative telemetry channels. |
| **3. 🗄️ Sensor Data & Custom FDR** | Raw telemetry inspection, CSV export & unseen FDR upload | Parses custom `.csv`/`.txt`/`.tsv` flight logs and previews standardized 26-column C-MAPSS channels. |
| **4. 🛠️ Predictive Maintenance** | FAA/EASA-style dispatch clearance & decision engine | Automated maintenance directives, interactive **"What-If" stress-tester**, and fleet priority ranking. |
| **5. ⚠️ Risk Detection** | ML classification & multi-sensor baseline heatmap | Classifies engine condition into **Normal**, **Warning**, and **Critical** failure risk tiers. |
| **6. 🔍 Anomaly Detection** | Dynamic rolling $Z$-score statistical outlier detection | Flags sensor flutter, thermal creep, and pressure divergence across 21 telemetry channels (`NaN`-safe on short logs). |
| **7. 🎯 RUL Improvement & XAI** | 73-feature GBR, `Q10–Q90` CQR intervals & XAI waterfall | Predicts exact operating cycles remaining (**45.7% lower RMSE** than baseline) with per-engine root-cause attribution. |
| **8. 🩺 Engine Health Score** | Multi-factor prognostic health index ($0 - 100$) | Synthesizes RUL ($40\%$), sensor stability ($35\%$), and cycle fatigue ($25\%$) into **4 tiers**: Healthy, Monitor, Warning, Critical. |
| **9. 📡 Real-Time Monitoring** | Continuous simulated flight stream with playback controls | Play, pause, step, and scrub cycles in real time with dynamic live RUL decay, CQR bounds, and automated alarms. |

---

## 🎮 Avionics Hotkeys & Wide-Screen Mode

AeroGuard integrates cockpit Multi-Function Display (MFD) style keyboard shortcuts designed for rapid telemetry triage and hands-on-throttle-and-stick (HOTAS) operation. Operators can collapse the sidebar to maximize screen real estate (**+25% wider telemetry viewport**) and navigate between modules without ever touching the mouse.

| Shortcut Key | Function | Operational Action |
| :---: | :--- | :--- |
| <kbd>Spacebar</kbd> / <kbd>\\</kbd> | **Toggle Sidebar (Wide-Screen Mode)** | Instantly collapses the sidebar for maximum chart width or re-opens it. |
| <kbd>1</kbd> – <kbd>9</kbd> | **Direct Module Jump** | Instantly switches to modules `[1]` to `[9]` (operates identically whether sidebar is open or closed). |
| <kbd>→</kbd> or <kbd>]</kbd> | **Next Screen** | Cycles forward sequentially through telemetry modules. |
| <kbd>←</kbd> or <kbd>[</kbd> | **Previous Screen** | Cycles backward sequentially through telemetry modules. |
| <kbd>?</kbd> / <kbd>/</kbd> / <kbd>H</kbd> | **Flight Keymap HUD** | Displays an aerospace frosted-glass keymap overlay on screen. |
| <kbd>Esc</kbd> | **Dismiss Overlay** | Closes the keymap modal. |

> **Smart Input Guard**: Hotkeys are automatically suspended when the operator is typing inside text or number inputs to prevent unintended page switches.

---

## 📊 Model Performance & Benchmarks

AeroGuard's improved prognostic engine utilizes a **Feature-Engineered Gradient Boosting Pipeline** evaluated against the NASA C-MAPSS benchmark dataset:

| Evaluation Metric | Baseline Model | AeroGuard Improved Pipeline | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **RMSE (Root Mean Squared Error)** | `63.83 cycles` | **`34.66 cycles`** | **45.7% Error Reduction** 🟢 |
| **MAE (Mean Absolute Error)** | `48.15 cycles` | **`27.10 cycles`** | **43.7% Error Reduction** 🟢 |
| **$R^2$ Explained Variance** | `0.3743` | **`0.8155`** | **+0.4412 Accuracy Gain** 🟢 |
| **NASA Risk Penalty Score** | `19,321,064` | **`106,423`** | **99.4% Safer Predictions** 🟢 |
| **Feature Dimensionality** | 6 raw channels | **73 engineered features** | +67 degradation interaction signals |

### Why the Improved Pipeline Outperforms:
1. **Operational Cycle Ratio (`cycle_ratio`)**: Normalizes cumulative cycle wear against expected lifespan.
2. **Rolling Statistical Windows (`roll5_mean`, `roll5_std`)**: Eliminates sensor jitter while capturing true mechanical degradation slopes.
3. **Thermodynamic Domain Spreads**: Captures enthalpy and pressure drops across the High-Pressure Compressor (HPC), Combustor, and Turbines.

---

## 🩺 Engine Health Score & Maintenance Decision Logic

### 1. Composite Health Formula ($0 - 100$)
$$\text{Health Score} = 0.40 \times \text{Score}_{\text{RUL}} + 0.35 \times \text{Score}_{\text{Sensors}} + 0.25 \times \text{Score}_{\text{Cycle Fatigue}}$$

* **🟢 Healthy (80 – 100)**: Optimal operational condition; standard flight envelopes.
* **🔵 Monitor (60 – 79)**: Mild sensor variance; enhanced 5 Hz telemetry logging active.
* **🟡 Warning (40 – 59)**: Pre-emptive maintenance required; schedule shop inspection within 20 cycles.
* **🔴 Critical (0 – 39)**: Imminent failure risk; immediate aircraft grounding (AOG) and core overhaul.

### 2. Predictive Maintenance Urgency Index (MUI)
Synthesizes RUL exhaustion, physical wear, and anomalous sensor density into an actionable dispatch verdict:
* **🚫 Ground Aircraft**: Immediate core teardown & borescope inspection.
* **⚠️ Conditional Dispatch**: Max 20 cycles before mandatory maintenance.
* **ℹ️ Watchlist Clearance**: Cleared with turnaround observation protocol.
* **✅ Unrestricted Flight**: Normal airline dispatch status.

---

## 🛠️ Technology Stack

* **Core Platform**: Python 3.10+, Streamlit
* **Machine Learning & Uncertainty**: Scikit-learn (`GradientBoostingRegressor` point + Q10/Q90 quantile ensembles, Split Conformalized Quantile Regression), Portable Pure-NumPy Tree Bundles (`.npz`), Saabas Tree-Path XAI Attribution, NumPy, Pandas
* **Telemetry Visualization**: Plotly Graph Objects (Soft whitish-grey high-contrast canvas, waterfall XAI charts, CQR confidence envelopes, polar radar diagnostics)
* **Digital Twin**: Custom vector SVG turbofan cross-section schematic with real-time station callouts
* **Styling**: Mission Control Aerospace CSS (Dark-mode cockpit HUD contrasted with high-legibility telemetry panels and monospace instrumentation)
* **Avionics Hotkey Engine**: Native client-side bridge for Spacebar wide-screen toggling, 1–9 direct module jumps, clickable `[?] KEYMAP` HUD pill, and Smart Input Guard

---

## 📂 Project Repository Structure

```text
AeroGuard/
├── app.py                            # Complete Mission Control Dashboard & Telemetry Server
├── train_improved_model.py           # 73-Feature Gradient Boosting Training Pipeline
├── train_quantile_models.py          # CQR Q10/Q90 Quantile Regressor & Portable .npz Exporter
├── improved_rul_portable.npz         # Version-Independent Portable Improved GBR Bundle
├── improved_rul_q10_portable.npz     # Portable Q10 Lower Bound Quantile Bundle
├── improved_rul_q90_portable.npz     # Portable Q90 Upper Bound Quantile Bundle
├── baseline_rul_portable.npz         # Portable Baseline GBR Bundle
├── improved_rul_quantiles_meta.json  # Conformal Calibration & Coverage Metadata (79.93%)
├── improved_rul_model.pkl            # Local Scikit-learn Improved Model (R² = 0.816)
├── improved_rul_scaler.pkl           # Feature Normalization Scaler
├── improved_model_features.json      # 73-Feature Schema Definitions
├── model_comparison_metrics.csv      # Head-to-Head Benchmark Validation Data
├── feature_importances.csv           # Top Thermodynamic Feature Rankings
├── requirements.txt                  # Production Python Dependencies
├── README.md                         # Executive Project Documentation
└── data/                             # NASA C-MAPSS Datasets & Custom FDR Test Sample
    ├── test_FD001_descriptive.csv    # Sea Level, Single Failure Mode (HPC Degradation)
    ├── test_FD002_descriptive.csv    # 6 Flight Conditions, Single Failure Mode
    ├── test_FD003_descriptive.csv    # Sea Level, Two Failure Modes (HPC + Fan)
    ├── test_FD004_descriptive.csv    # 6 Flight Conditions, Two Failure Modes
    └── sample_unseen_fdr_test.csv    # Unseen 2-Engine Custom FDR Upload Sample (#101 & #204)
```

---

## ⚡ Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mayuresh-khare/AeroGuard-Aircraft_Engine_Health_Prediction_System.git
cd AeroGuard
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the console.

---

## 📄 License & Attribution

* **Dataset Source**: NASA Ames Prognostics Center of Excellence (PCoE) — Turbofan Engine Degradation Simulation (C-MAPSS).
* **License**: MIT Open Source License. Developed as an enterprise-grade aircraft predictive health monitoring solution.
