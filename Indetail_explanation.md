# Data Drift and Model Decay — Complete Project Documentation

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Workflow](#2-high-level-workflow)
3. [Directory Structure](#3-directory-structure)
4. [Step-by-Step Pipeline Walkthrough](#4-step-by-step-pipeline-walkthrough)
5. [File-by-File Deep Dive](#5-file-by-file-deep-dive)
   - [train.py](#51-trainpy--model-training)
   - [new.py](#52-newpy--synthetic-drift-data-generator)
   - [monitor.py](#53-monitorpy--cli-monitoring-pipeline)
   - [app.py](#54-apppy--streamlit-dashboard)
   - [requirements.txt](#55-requirementstxt)
6. [Datasets](#6-datasets)
7. [Key Concepts Explained](#7-key-concepts-explained)
   - [Data Drift](#data-drift)
   - [Model Decay](#model-decay)
   - [PSI](#psi-population-stability-index)
   - [KL Divergence](#kl-divergence)
   - [JS Divergence](#js-divergence)
   - [Drift Impact Score](#drift-impact-score)
8. [Generated Artifacts](#8-generated-artifacts)
9. [Running the Project](#9-running-the-project)

---

## 1. What This Project Does

This is a **Machine Learning Observability System**. It monitors a trained ML model in production to detect two problems:

- **Data Drift** — when the statistical distribution of incoming data changes compared to what the model was trained on (e.g., average customer age shifts, credit limits inflate).
- **Model Decay** — when the model's prediction performance degrades over time because the data it now sees no longer matches what it learned from.

The domain is **credit card default prediction** using the UCI Credit Card dataset. The model predicts whether a customer will default on their next payment (binary: 0 = no default, 1 = default).

The system has two monitoring interfaces:
- A **CLI script** (`monitor.py`) for automated/scheduled runs.
- An **interactive Streamlit web dashboard** (`app.py`) with charts, alerts, and configurable parameters.

---

## 2. High-Level Workflow

```
┌──────────────────────────────────────────────────────┐
│  STEP 1 — Generate test drift data  (new.py)        │
│  Reads UCI_Credit_Card.csv                          │
│  Applies known shifts (LIMIT_BAL +50%, AGE +2 …)   │
│  Saves → new_data.csv                               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STEP 2 — Train the model  (train.py)               │
│  Reads UCI_Credit_Card.csv                          │
│  Imputes + scales + trains RandomForest              │
│  Evaluates baseline F1 & Recall                     │
│  Saves → model.pkl, scaler.pkl,                     │
│           imputer.pkl, baseline.pkl                 │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STEP 3 — Monitor for drift & decay                 │
│                                                      │
│  Option A — CLI (monitor.py)                        │
│    Reads new_data.csv + all .pkl artifacts          │
│    Computes PSI per feature                         │
│    Predicts, measures F1 drop                       │
│    Computes Drift Impact Score                      │
│    Logs trend + triggers alert if score > 0.05      │
│                                                      │
│  Option B — Dashboard (app.py / Streamlit)          │
│    User uploads model.pkl + baseline.pkl + CSV      │
│    Computes PSI, KL, JS, Unified Drift              │
│    Shows feature-level drift ranked by importance   │
│    Shows performance history + trend chart          │
│    User can adjust α, threshold, rolling window     │
│    User can commit current batch to rolling baseline│
└──────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Data-Drift-and-Model-Decay/
├── train.py                       Training pipeline
├── monitor.py                     CLI monitoring pipeline
├── app.py                         Streamlit dashboard
├── new.py                         Synthetic drift data generator
├── UCI_Credit_Card.csv            Full dataset — 30,000 rows (baseline)
├── UCI_Credit_Card_sample_500.csv Small sample — 500 rows (quick testing)
├── new_data.csv                   Simulated production data (created by new.py)
├── requirements.txt               Python dependencies
├── .gitignore                     Excludes .pkl files and CSVs from git
└── README.md                      Original project readme

Generated at runtime (not in repo):
├── model.pkl                      Trained RandomForest
├── scaler.pkl                     Fitted StandardScaler
├── imputer.pkl                    Fitted SimpleImputer
├── baseline.pkl                   Baseline metadata + X_train distribution
└── performance_history.pkl        Running list of F1 scores across batches
```

---

## 4. Step-by-Step Pipeline Walkthrough

### Step A — Create drifted inference data (`new.py`)

`new.py` simulates what might happen months after deployment: the real-world data starts shifting. It copies the original dataset and applies deliberate changes:

| Column | Change Applied |
|--------|----------------|
| `LIMIT_BAL` | ×1.5 (credit limits inflated by 50%) |
| `BILL_AMT1` | ×1.3 (billing amounts up 30%) |
| `PAY_AMT1` | ×0.7 (payment amounts down 30%) |
| `AGE` | +2 years (customer base aged 2 years) |
| All numeric cols | + Gaussian noise (mean=0, std=0.05) |

The target column (`default.payment.next.month`) is intentionally **not modified** — labels remain clean so we can still evaluate model accuracy and isolate feature drift from label noise.

**Output:** `new_data.csv` — a 30,000-row CSV with known, controlled drift.

---

### Step B — Train baseline model (`train.py`)

1. Load `UCI_Credit_Card.csv`
2. Drop the `ID` column (it's an identifier, not a feature)
3. Split: 70% train / 30% test (random_state=42 for reproducibility)
4. Impute missing values using **mean strategy** (fitted on train, applied to test)
5. Scale features using **StandardScaler** (zero mean, unit variance — fitted on train)
6. Train a **RandomForestClassifier** (100 trees, random_state=42)
7. Evaluate on test set: compute **F1-score** and **Recall** as baseline benchmarks
8. Save 4 artifacts:
   - `model.pkl` — the trained classifier
   - `scaler.pkl` — the fitted scaler (must be used for all future data)
   - `imputer.pkl` — the fitted imputer (same)
   - `baseline.pkl` — a dict with `X_train` (raw, unscaled), `baseline_f1`, `baseline_recall`

The raw `X_train` is stored (not the scaled version) so it can be used as the **reference distribution** for statistical drift calculations later.

---

### Step C — CLI monitoring (`monitor.py`)

Run manually or on a schedule. Takes new data, compares distributions, measures performance drop.

1. Load all 4 artifacts
2. Load `new_data.csv`, drop `ID`, separate features from target
3. Compute **PSI** for every numeric feature (baseline distribution vs. new data)
4. Impute + scale new data using the **same** fitted imputer and scaler
5. Run model predictions
6. Compute current F1 and Recall; calculate performance drop = `baseline_f1 - current_f1`
7. Compute **Drift Impact Score** = `0.6 × avg_PSI + 0.4 × performance_drop`
8. Append current F1 to `performance_history.pkl`; fit a degree-1 polynomial on last 5 entries to detect trend direction
9. Trigger alert if `drift_impact_score > 0.05`
10. Return a result dict with all metrics

---

### Step D — Dashboard monitoring (`app.py`)

A Streamlit web app with file uploaders and interactive controls.

1. User uploads `model.pkl`, `baseline.pkl`, a new CSV
2. User sets: α (drift weight), alert threshold, rolling window size, target column name
3. Compute **PSI + KL + JS divergence** per feature
4. Combine into **Unified Drift Score** = `0.5×PSI + 0.25×KL + 0.25×JS`
5. Weight each feature's drift by its **model feature importance** → **Impact Rank**
6. Attempt predictions; compute F1 and performance drop
7. Drift Impact Score = `α × avg_unified_drift + (1-α) × perf_drop`
8. Detect decay if polynomial trend of F1 history < -0.01
9. Render 4 metric cards, alert banners, feature drift table, performance trend chart, drift metric breakdown chart
10. Optional: commit current batch to rolling baseline for dynamic reference

---

## 5. File-by-File Deep Dive

### 5.1 `train.py` — Model Training

**Run with:** `python train.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
import joblib
import logging
import warnings
```

**Single function: `train_and_save(filepath, target_col)`**

```python
def train_and_save(filepath='UCI_Credit_Card.csv', target_col='default.payment.next.month'):
    df = pd.read_csv(filepath)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])        # Remove identifier column

    X = df.drop(columns=[target_col])       # Features
    y = df[target_col]                      # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

`test_size=0.3` → 21,000 train rows, 9,000 test rows. `random_state=42` makes the split reproducible.

```python
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)   # Learn mean from train
    X_test_imputed  = imputer.transform(X_test)        # Apply same mean to test
```

`fit_transform` on training data means the imputer learns what the mean of each column is from training data only — preventing data leakage from test.

```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)  # Learn μ and σ from train
    X_test_scaled  = scaler.transform(X_test_imputed)       # Apply to test
```

StandardScaler transforms each feature to have mean=0 and std=1 based on the training distribution.

```python
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    baseline_f1     = f1_score(y_test, y_pred, zero_division=0)
    baseline_recall = recall_score(y_test, y_pred, zero_division=0)
```

F1-score = harmonic mean of precision and recall. It's preferred over accuracy for imbalanced classes (credit default datasets are imbalanced — most people don't default).

```python
    baseline_data = {
        'X_train': X_train.copy(),       # Raw (unscaled) training features
        'baseline_f1': baseline_f1,
        'baseline_recall': baseline_recall
    }

    joblib.dump(model,         'model.pkl')
    joblib.dump(scaler,        'scaler.pkl')
    joblib.dump(imputer,       'imputer.pkl')
    joblib.dump(baseline_data, 'baseline.pkl')
```

`X_train` is stored **raw (unscaled)** because PSI and other drift metrics compare raw distributions, not scaled ones. Comparing scaled features would mix the scaling transformation with the actual distribution shift.

---

### 5.2 `new.py` — Synthetic Drift Data Generator

**Run with:** `python new.py`

```python
import pandas as pd
import numpy as np

target_col = 'default.payment.next.month'
df = pd.read_csv("UCI_Credit_Card.csv")
future_df = df.copy()
np.random.seed(42)
```

`np.random.seed(42)` makes the noise generation reproducible.

```python
# Apply realistic drift
future_df['LIMIT_BAL'] = future_df['LIMIT_BAL'] * 1.5   # Credit limits inflated
future_df['BILL_AMT1'] = future_df['BILL_AMT1'] * 1.3   # Higher bills
future_df['PAY_AMT1']  = future_df['PAY_AMT1']  * 0.7   # Lower payments
future_df['AGE']       = future_df['AGE']        + 2    # Older customers
```

These multipliers simulate real-world macro trends (e.g., inflation driving up credit limits and bills, economic stress reducing payments, natural aging of the customer base).

```python
# Add mild noise to all numeric columns EXCEPT the target
num_cols = future_df.select_dtypes(include=np.number).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)   # Do NOT add noise to labels

future_df[num_cols] = future_df[num_cols] + np.random.normal(0, 0.05, future_df[num_cols].shape)
future_df.to_csv("new_data.csv", index=False)
```

The noise (std=0.05) is small — it simulates measurement variance / sensor noise without overwhelming the deliberate drift signal. Labels are left clean so monitoring can accurately evaluate model performance.

---

### 5.3 `monitor.py` — CLI Monitoring Pipeline

**Run with:** `python monitor.py --new_data new_data.csv`

#### Function: `calculate_psi(expected, actual, bins=10)`

```python
def calculate_psi(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    expected_min = expected.min()
    expected_max = expected.max()
    bins_edges = np.linspace(expected_min, expected_max + 1e-5, bins + 1)
```

Bin edges are defined by the **expected (baseline)** range. The `+1e-5` ensures the maximum value falls inside the last bin instead of being excluded.

```python
    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts,   _ = np.histogram(actual,   bins=bins_edges)

    expected_pct = expected_counts / len(expected)   # Proportion per bin
    actual_pct   = actual_counts   / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)  # Avoid log(0)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)
```

PSI formula: for each bin, `(actual% − expected%) × ln(actual% / expected%)`. The sum across all bins gives the PSI.

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 – 0.25 | Minor shift, monitor closely |
| > 0.25 | Major shift, retraining likely needed |

#### Function: `detect_drift_psi(baseline_df, new_df)`

```python
def detect_drift_psi(baseline_df, new_df):
    results = []
    numerical_cols = baseline_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in new_df.columns:
            expected = baseline_df[col].dropna().values
            actual   = new_df[col].dropna().values
            if len(expected) > 0 and len(actual) > 0:
                psi = calculate_psi(expected, actual)
                results.append({'Feature': col, 'PSI': psi})

    df_report = pd.DataFrame(results)
    if not df_report.empty:
        df_report = df_report.sort_values(by='PSI', ascending=False)
    return df_report
```

Loops over every numeric column and computes PSI. Returns a DataFrame sorted by PSI descending — the most drifted features appear first.

#### Function: `run_monitoring(...)`

```python
def run_monitoring(model_path, scaler_path, imputer_path, baseline_path, new_data_path, target_col):
    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    baseline = joblib.load(baseline_path)

    baseline_df     = baseline['X_train']
    baseline_f1     = baseline['baseline_f1']
    baseline_recall = baseline['baseline_recall']
```

Load all artifacts. `baseline_df` is the raw training data — used as the reference distribution for PSI.

```python
    new_df = pd.read_csv(new_data_path)
    if 'ID' in new_df.columns:
        new_df = new_df.drop(columns=['ID'])

    y_true = None
    if target_col in new_df.columns:
        y_true = new_df[target_col]
        X_new  = new_df.drop(columns=[target_col])
    else:
        X_new  = new_df.copy()
```

Separate features from labels. If there are no labels in the new data, performance metrics are skipped.

```python
    drift_report = detect_drift_psi(baseline_df, X_new)
    drift_score  = drift_report['PSI'].mean() if not drift_report.empty else 0.0
```

`drift_score` = average PSI across all features. A single summary number.

```python
    X_new_imputed  = imputer.transform(X_new)       # Apply same imputer (no re-fit)
    X_new_scaled   = scaler.transform(X_new_imputed) # Apply same scaler (no re-fit)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns)

    y_pred = model.predict(X_new_scaled_df)
```

Critical: we use `.transform()` (not `.fit_transform()`). Re-fitting would update the scaler's learned mean/std to match the new drifted data, which would hide the drift.

```python
    if y_true is not None:
        current_f1     = f1_score(y_true, y_pred, zero_division=0)
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        performance_drop = baseline_f1 - current_f1
```

Performance drop = how much F1 has fallen from the baseline. Positive means worse performance.

```python
    if current_f1 is not None:
        drift_impact_score = (0.6 * drift_score) + (0.4 * performance_drop)
    else:
        drift_impact_score = drift_score
```

The **Drift Impact Score** is a weighted combination: 60% statistical drift + 40% actual performance degradation. This acknowledges that not all statistical drift hurts performance (some features may drift but be unimportant to the model).

```python
    performance_history = []
    if os.path.exists('performance_history.pkl'):
        performance_history = joblib.load('performance_history.pkl')

    if current_f1 is not None:
        performance_history.append(current_f1)
        joblib.dump(performance_history, 'performance_history.pkl')
```

Persist F1 history across multiple monitoring runs. Each run appends to the list.

```python
    if len(performance_history) >= 2:
        last_n    = performance_history[-5:]                         # Last 5 batches
        trend_val = np.polyfit(range(len(last_n)), last_n, 1)[0]    # Slope of line
        if trend_val < 0:
            trend_status = "Model degrading"
        else:
            trend_status = "Stable/Improving"
```

`np.polyfit(..., 1)` fits a straight line (`y = m×x + b`) to the F1 scores. The returned `[0]` is the slope `m`. If slope is negative, performance is trending down.

```python
    impact_threshold = 0.05
    alert_triggered  = drift_impact_score > impact_threshold
```

Hardcoded in the CLI. Configurable in the dashboard.

**Return value:** a dict with all computed metrics — `drift_score`, `drift_report`, `current_f1`, `current_recall`, `performance_drop`, `drift_impact_score`, `trend_status`, `trend_val`, `alert_triggered`.

---

### 5.4 `app.py` — Streamlit Dashboard

**Run with:** `streamlit run app.py`

#### Drift metric functions

**`calculate_psi(expected, actual, bins=10)`**

Same logic as `monitor.py` but uses `np.clip` instead of `np.where` to replace zeros:
```python
e_pct = np.clip(e_pct, 1e-6, None)   # Clip to minimum 1e-6
a_pct = np.clip(a_pct, 1e-6, None)
```
Both approaches achieve the same goal — prevent `log(0)`.

**`calculate_kl_divergence(expected, actual, bins=10)`**

```python
def calculate_kl_divergence(expected, actual, bins=10):
    hist_e, _ = np.histogram(expected, bins=bins, density=True)   # density=True → prob density
    hist_a, _ = np.histogram(actual,   bins=bins, density=True)
    hist_e = np.clip(hist_e, 1e-6, None)
    hist_a = np.clip(hist_a, 1e-6, None)
    return entropy(hist_a, hist_e)   # scipy entropy = KL divergence
```

`entropy(P, Q)` from scipy computes `Σ P(i) × log(P/Q)` — the KL divergence from Q to P. This is **asymmetric**: KL(P||Q) ≠ KL(Q||P).

**`calculate_js_divergence(expected, actual, bins=10)`**

```python
def calculate_js_divergence(expected, actual, bins=10):
    m = 0.5 * (hist_e + hist_a)                          # Midpoint distribution M
    return 0.5 * entropy(hist_e, m) + 0.5 * entropy(hist_a, m)
```

JS divergence is the **symmetric** version: average of KL from each distribution to their midpoint. Always between 0 and 1 (when using log base 2) — easier to interpret.

**`get_unified_drift(expected, actual)`**

```python
def get_unified_drift(expected, actual):
    psi = calculate_psi(expected, actual)
    kl  = calculate_kl_divergence(expected, actual)
    js  = calculate_js_divergence(expected, actual)
    return (0.5 * psi) + (0.25 * kl) + (0.25 * js), psi, kl, js
```

A single blended score. PSI gets the most weight (0.5) because it's the industry standard for distribution shift. KL and JS each get 0.25 — they add signal about statistical distance.

#### Page config and styling

```python
st.set_page_config(page_title="Adaptive ML Observability", layout="wide")
```

`layout="wide"` uses the full browser width.

Custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`:
- Dark background: `linear-gradient(135deg, #020617 0%, #0f172a 100%)`
- Google Font: Outfit (loaded via `@import`)
- Glassmorphism metric cards: semi-transparent background + `backdrop-filter: blur(12px)`
- Metric values styled in sky blue (`#38bdf8`)

#### Sidebar controls

```python
with st.sidebar:
    alpha       = st.slider("Drift Influence (α)", 0.0, 1.0, 0.6)
    threshold   = st.number_input("Alert Threshold", value=0.1)
    window_size = st.number_input("Rolling Window size", value=5)
    target_col  = st.text_input("Target Column Name")
    model_file     = st.file_uploader("Model (.pkl)",           type=['pkl'])
    baseline_file  = st.file_uploader("Initial Baseline (.pkl)", type=['pkl'])
    new_data_file  = st.file_uploader("Incoming Data (.csv)",   type=['csv'])
```

- **α** — controls how much statistical drift vs. performance drop matters in the impact score. α=1 means drift only; α=0 means performance only.
- **Alert Threshold** — configurable. Default 0.1 (more conservative than CLI's 0.05).
- **Rolling Window** — how many past batches form the dynamic baseline.

If any of the three files is missing, the app shows a welcome message and stops.

#### Session state — rolling baseline

```python
if 'rolling_buffer' not in st.session_state:
    st.session_state.rolling_buffer = [baseline_df]   # Starts with original baseline
if 'perf_history' not in st.session_state:
    st.session_state.perf_history = []
```

Streamlit re-runs the entire script on every user interaction. `st.session_state` persists data across reruns within a session (like a server-side dict).

`rolling_buffer` starts containing the original baseline. When the user clicks "Commit to Rolling Baseline", the current batch is appended and old batches beyond `window_size` are dropped. This allows the reference distribution to gradually shift with expected seasonal changes while still catching anomalous drift.

#### Drift calculation

```python
current_reference = pd.concat(st.session_state.rolling_buffer[-window_size:])
```

Concatenate the last `window_size` batches into a single reference DataFrame.

```python
if hasattr(model, 'feature_importances_'):
    feature_importance = dict(zip(X_new.columns, model.feature_importances_))
elif hasattr(model, 'coef_'):
    feature_importance = dict(zip(X_new.columns, np.abs(model.coef_[0])))
```

Works with both tree-based models (`feature_importances_`) and linear models (`coef_`). If neither exists, importance defaults to 1.0 for all features.

```python
for col in X_new.select_dtypes(include=[np.number]).columns:
    if col in current_reference.columns:
        u_score, psi, kl, js = get_unified_drift(
            current_reference[col].dropna(), X_new[col].dropna()
        )
        importance = feature_importance.get(col, 1.0)
        drift_results.append({
            'Feature': col,
            'Unified Drift': u_score,
            'PSI': psi, 'KL': kl, 'JS': js,
            'Importance': importance,
            'Impact Rank': u_score * importance    # Drift × feature importance
        })
```

**Impact Rank** = `unified_drift × feature_importance`. A feature that drifts a lot but has near-zero importance to the model is ranked lower than a moderately drifted feature that the model relies on heavily. This focuses retraining attention where it actually matters.

#### Performance and decay

```python
try:
    y_pred = model.predict(X_new)    # Direct prediction (no scaling applied)
    if y_true is not None:
        current_perf = f1_score(y_true, y_pred, zero_division=0)
        perf_drop    = max(0, base_perf - current_perf)
        st.session_state.perf_history.append(current_perf)
except Exception as e:
    st.error(f"Prediction Error: {e}. Ensure the model and data columns match.")
```

**Note:** The dashboard calls `model.predict(X_new)` on **raw, unscaled** data. This will fail if the model was trained on scaled data (as it was in `train.py`). In a production setup, the scaler and imputer should also be uploaded and applied. The `try/except` catches this gracefully.

```python
drift_impact_score = (alpha * avg_unified_drift) + ((1 - alpha) * perf_drop)
```

Mirrors the CLI formula but uses user-configurable `alpha` and the unified drift score (which includes KL and JS, not just PSI).

```python
if len(st.session_state.perf_history) >= 3:
    trend_val = np.polyfit(range(len(history)), history, 1)[0]
    if trend_val < -0.01:
        decay_detected = True
```

Trend uses the **full** history (not just last 5 like CLI). Threshold is stricter: `-0.01` (the CLI triggers on any negative slope; the dashboard requires slope < -0.01).

#### Dashboard layout

```python
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unified Drift Score",  f"{avg_unified_drift:.4f}")
m2.metric("Drift Impact Score",   f"{drift_impact_score:.4f}")
m3.metric("Current F1-Score",     f"{current_perf:.4f}" if current_perf else "N/A")
m4.metric("Model Health Trend",   "Decline" if trend_val < 0 else "Stable",
          delta=f"{trend_val:.4f}")
```

4-column metric cards at the top — the "at a glance" view.

```python
if drift_impact_score > threshold:
    st.error("🚨 CRITICAL ALERT: ...")
elif decay_detected:
    st.warning("⚠️ DECAY WARNING: ...")
else:
    st.success("🟢 System Status: Stable.")
```

Three alert states, mutually exclusive in priority order.

**Left column (wider):**
- Feature-Level Drift table (top 10 by Impact Rank, color-gradient on Unified Drift column)
- Performance History line chart (with SMA-3 trend line)

**Right column:**
- Bar chart of average PSI, KL, JS across all features
- Info box explaining each metric

**Rolling baseline button:**
```python
if st.button("✅ Commit to Rolling Baseline"):
    st.session_state.rolling_buffer.append(X_new)
    if len(st.session_state.rolling_buffer) > window_size:
        st.session_state.rolling_buffer.pop(0)   # Drop oldest batch
    st.toast("Current batch added to reference buffer!")
```

---

### 5.5 `requirements.txt`

```
pandas        Data loading, manipulation, DataFrames
numpy         Array math, histograms, polyfit
scikit-learn  RandomForest, StandardScaler, SimpleImputer, f1_score, recall_score
joblib        Save/load .pkl files efficiently
streamlit     Web dashboard framework
matplotlib    Plotting (line charts, bar charts)
scipy         entropy() function for KL/JS divergence
```

**Install with:** `pip install -r requirements.txt`

---

## 6. Datasets

### `UCI_Credit_Card.csv` — Baseline training data

- **Size:** 30,000 rows × 25 columns
- **Source:** UCI Machine Learning Repository — Taiwan credit card dataset
- **Task:** Binary classification — predict credit default

| Column Group | Columns | Description |
|---|---|---|
| Identifier | `ID` | Row identifier — dropped before training |
| Target | `default.payment.next.month` | 0 = no default, 1 = default |
| Credit info | `LIMIT_BAL` | Credit limit amount |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Customer attributes |
| Payment status | `PAY_0`, `PAY_2` – `PAY_6` | Repayment status for past 6 months |
| Bill amounts | `BILL_AMT1` – `BILL_AMT6` | Statement balance for past 6 months |
| Payment amounts | `PAY_AMT1` – `PAY_AMT6` | Amount paid for past 6 months |

### `new_data.csv` — Simulated production data with drift

- **Size:** 30,000 rows × 25 columns (same structure)
- **Created by:** `new.py`
- **Drift pattern:** LIMIT_BAL +50%, BILL_AMT1 +30%, PAY_AMT1 −30%, AGE +2, plus small noise on all numerics
- **Labels:** Clean (not modified) — enables performance evaluation

### `UCI_Credit_Card_sample_500.csv`

- **Size:** 500 rows
- **Purpose:** Quick testing / demos without waiting for 30,000-row processing

---

## 7. Key Concepts Explained

### Data Drift

When the statistical properties of input features change between training time and inference time. The model was trained on data from distribution A, but now sees data from distribution B. Even if the model is perfect, if B ≠ A, predictions may be unreliable.

**Example in this project:** LIMIT_BAL was trained with a certain mean credit limit. In `new_data.csv`, that mean is 50% higher. The model never saw inputs that large during training.

### Model Decay

The gradual degradation of model performance over time — measured by a drop in F1-score, recall, or accuracy compared to the baseline evaluation. Decay can be caused by drift, but also by concept drift (the relationship between features and labels changes, e.g., economic conditions shift what predicts default).

### PSI (Population Stability Index)

The industry-standard metric for measuring distribution shift. Originally used in banking to monitor scorecards.

**Formula:**
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Applied per bin across the feature's range. A PSI of 0 means identical distributions; higher values mean more shift.

### KL Divergence

Kullback-Leibler Divergence — measures "how surprised" distribution P would be if it had been generated by distribution Q.

**Formula:**
```
KL(P || Q) = Σ P(i) × ln(P(i) / Q(i))
```

**Asymmetric** — KL(P||Q) ≠ KL(Q||P). Not bounded (can be very large).

### JS Divergence

Jensen-Shannon Divergence — the symmetric version of KL. Always between 0 and 1.

**Formula:**
```
M = 0.5 × (P + Q)
JS(P, Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
```

### Drift Impact Score

A combined metric that merges statistical drift with actual performance impact:

**CLI version:**
```
Drift Impact Score = 0.6 × avg_PSI + 0.4 × performance_drop
```

**Dashboard version (configurable):**
```
Drift Impact Score = α × unified_drift + (1 - α) × performance_drop
```

The weighting acknowledges that statistical drift alone doesn't always hurt the model — the performance component grounds the score in real-world impact.

---

## 8. Generated Artifacts

| File | Created by | Contents | Used by |
|---|---|---|---|
| `model.pkl` | `train.py` | Trained RandomForestClassifier | `monitor.py`, `app.py` |
| `scaler.pkl` | `train.py` | Fitted StandardScaler | `monitor.py` |
| `imputer.pkl` | `train.py` | Fitted SimpleImputer | `monitor.py` |
| `baseline.pkl` | `train.py` | Dict: `X_train`, `baseline_f1`, `baseline_recall` | `monitor.py`, `app.py` |
| `performance_history.pkl` | `monitor.py` | List of F1 scores across runs | `monitor.py` (appended each run) |
| `new_data.csv` | `new.py` | 30,000-row drifted dataset | `monitor.py`, `app.py` |

`.pkl` files and `.csv` files are excluded from git via `.gitignore` — they are regenerated locally.

---

## 9. Running the Project

### Setup

```bash
pip install -r requirements.txt
```

### Full workflow

```bash
# 1. Generate drifted test data
python new.py

# 2. Train the baseline model
python train.py

# 3a. Run CLI monitoring
python monitor.py --new_data new_data.csv

# 3b. OR launch the interactive dashboard
streamlit run app.py
```

### Dashboard usage

1. Open the Streamlit URL (default: `http://localhost:8501`)
2. In the sidebar, upload:
   - `model.pkl`
   - `baseline.pkl`
   - `new_data.csv`
3. Set Target Column to: `default.payment.next.month`
4. Adjust α, threshold, and window size as needed
5. View metrics, alerts, charts
6. Click "Commit to Rolling Baseline" to update the reference distribution

### Expected outputs after running `monitor.py`

```
INFO - Drift Score (Avg PSI): 0.xxxx
INFO - Performance F1    : 0.xxxx (Drop: 0.xxxx)
INFO - Drift Impact Score: 0.xxxx
INFO - Trend Analysis    : Model degrading / Stable/Improving
INFO - Alert Triggered   : Yes / No
```

The high PSI values for `LIMIT_BAL`, `BILL_AMT1`, `PAY_AMT1`, and `AGE` will confirm that the drift injected by `new.py` was detected.
