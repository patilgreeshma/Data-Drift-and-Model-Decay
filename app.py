import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from scipy.stats import entropy
from sklearn.metrics import f1_score, recall_score, accuracy_score

# -----------------------------------
# ADVANCED ANALYTICS UTILITIES
# -----------------------------------

def calculate_psi(expected, actual, bins=10):
    """Population Stability Index"""
    if len(expected) == 0 or len(actual) == 0: return 0.0
    expected_min, expected_max = expected.min(), expected.max()
    bins_edges = np.linspace(expected_min, expected_max + 1e-5, bins + 1)
    
    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts, _ = np.histogram(actual, bins=bins_edges)
    
    # Probabilities
    e_pct = expected_counts / len(expected)
    a_pct = actual_counts / len(actual)
    
    # Small constant to avoid log(0)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

def calculate_kl_divergence(expected, actual, bins=10):
    """Kullback-Leibler Divergence"""
    if len(expected) == 0 or len(actual) == 0: return 0.0
    hist_e, _ = np.histogram(expected, bins=bins, density=True)
    hist_a, _ = np.histogram(actual, bins=bins, density=True)
    hist_e = np.clip(hist_e, 1e-6, None)
    hist_a = np.clip(hist_a, 1e-6, None)
    return entropy(hist_a, hist_e)

def calculate_js_divergence(expected, actual, bins=10):
    """Jensen-Shannon Divergence (Symmetric KL)"""
    if len(expected) == 0 or len(actual) == 0: return 0.0
    hist_e, _ = np.histogram(expected, bins=bins, density=True)
    hist_a, _ = np.histogram(actual, bins=bins, density=True)
    hist_e = np.clip(hist_e, 1e-6, None)
    hist_a = np.clip(hist_a, 1e-6, None)
    
    m = 0.5 * (hist_e + hist_a)
    return 0.5 * entropy(hist_e, m) + 0.5 * entropy(hist_a, m)

def get_unified_drift(expected, actual):
    """Aggregates multiple drift metrics"""
    psi = calculate_psi(expected, actual)
    kl = calculate_kl_divergence(expected, actual)
    js = calculate_js_divergence(expected, actual)
    
    # Normalizing weights for a unified score
    # PSI is usually higher magnitude, JS/KL are 0-1 bounded (mostly)
    return (0.5 * psi) + (0.25 * kl) + (0.25 * js), psi, kl, js

# -----------------------------------
# STYLING & PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Adaptive ML Observability", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #020617 0%, #0f172a 100%); color: #f8fafc; }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1.5rem;
        border-radius: 20px;
        backdrop-filter: blur(12px);
    }
    
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
    
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Adaptive ML Observability Framework")
st.caption("Multi-Metric Drift Detection | Dynamic Baseline | Automated Decay Analysis")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    alpha = st.slider("Drift Influence (α)", 0.0, 1.0, 0.6, help="Weight given to drift vs performance in Impact Score")
    threshold = st.number_input("Alert Threshold", value=0.1, step=0.01)
    window_size = st.number_input("Rolling Window size", value=5, min_value=1)
    target_col = st.text_input("Target Column Name", placeholder="e.g. default.payment.next.month")
    
    st.divider()
    st.header("📂 Upload Artifacts")
    model_file = st.file_uploader("Model (.pkl)", type=['pkl'])
    baseline_file = st.file_uploader("Initial Baseline (.pkl)", type=['pkl'])
    new_data_file = st.file_uploader("Incoming Data (.csv)", type=['csv'])

if not (model_file and baseline_file and new_data_file):
    st.info("👋 Welcome! Please upload your Model, Baseline, and New Data to begin monitoring.")
    st.stop()

# -----------------------------------
# DATA PROCESSING
# -----------------------------------
@st.cache_resource
def load_model_objects(_m, _b):
    return joblib.load(_m), joblib.load(_b)

try:
    model, baseline_obj = load_model_objects(model_file, baseline_file)
    baseline_df = baseline_obj['X_train']
    base_perf = baseline_obj.get('baseline_f1', 0.8) # Default if missing
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Check for dynamic baseline storage
if 'rolling_buffer' not in st.session_state:
    st.session_state.rolling_buffer = [baseline_df]
if 'perf_history' not in st.session_state:
    st.session_state.perf_history = []

# Process incoming data
new_df = pd.read_csv(new_data_file)

# Dataset Agnostic Logic
if target_col and target_col in new_df.columns:
    y_true = new_df[target_col]
    X_new = new_df.drop(columns=[target_col])
    if 'ID' in X_new.columns: X_new = X_new.drop(columns=['ID']) # Common junk column
else:
    X_new = new_df.copy()
    if 'ID' in X_new.columns: X_new = X_new.drop(columns=['ID'])
    y_true = None
    st.warning("No target column found or specified. Performance metrics will be skipped.")

# DYNAMIC BASELINE: Use the buffer as reference
# Combine recent batches into a single reference distribution
current_reference = pd.concat(st.session_state.rolling_buffer[-window_size:])

# -----------------------------------
# DRIFT CALCULATION
# -----------------------------------
drift_results = []
feature_importance = {}

# Try to get feature importance from model
if hasattr(model, 'feature_importances_'):
    feature_importance = dict(zip(X_new.columns, model.feature_importances_))
elif hasattr(model, 'coef_'):
    feature_importance = dict(zip(X_new.columns, np.abs(model.coef_[0])))

for col in X_new.select_dtypes(include=[np.number]).columns:
    if col in current_reference.columns:
        u_score, psi, kl, js = get_unified_drift(current_reference[col].dropna(), X_new[col].dropna())
        importance = feature_importance.get(col, 1.0)
        drift_results.append({
            'Feature': col,
            'Unified Drift': u_score,
            'PSI': psi,
            'KL': kl,
            'JS': js,
            'Importance': importance,
            'Impact Rank': u_score * importance
        })

drift_summary_df = pd.DataFrame(drift_results).sort_values(by='Impact Rank', ascending=False)
avg_unified_drift = drift_summary_df['Unified Drift'].mean()

# -----------------------------------
# PERFORMANCE & DECAY
# -----------------------------------
# Attempt Prediction
# Note: This assumes model has a predict method and data matches features
try:
    # Minimal preprocessing if model expects array
    # In a real system, we'd apply the loaded Scaler/Imputer here.
    # For this snippet, we'll try direct prediction if scaler isn't provided or needed
    y_pred = model.predict(X_new)
    
    current_perf = None
    perf_drop = 0.0
    if y_true is not None:
        current_perf = f1_score(y_true, y_pred, zero_division=0)
        perf_drop = max(0, base_perf - current_perf)
        st.session_state.perf_history.append(current_perf)
except Exception as e:
    st.error(f"Prediction Error: {e}. Ensure the model and data columns match.")
    perf_drop = 0.0
    current_perf = 0.0

# Drift Impact Score calculation
drift_impact_score = (alpha * avg_unified_drift) + ((1 - alpha) * perf_drop)

# Moving Average & Trend
trend_val = 0
decay_detected = False
if len(st.session_state.perf_history) >= 3:
    history = st.session_state.perf_history
    # Trend using polyfit
    trend_val = np.polyfit(range(len(history)), history, 1)[0]
    if trend_val < -0.01: # Threshold for "Consistent Decline"
        decay_detected = True

# -----------------------------------
# DASHBOARD UI
# -----------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unified Drift Score", f"{avg_unified_drift:.4f}")
m2.metric("Drift Impact Score", f"{drift_impact_score:.4f}")
m3.metric("Current F1-Score", f"{current_perf:.4f}" if current_perf else "N/A")
m4.metric("Model Health Trend", "Decline" if trend_val < 0 else "Stable", delta=f"{trend_val:.4f}")

st.divider()

if drift_impact_score > threshold:
    st.error(f"🚨 **CRITICAL ALERT**: Drift Impact Score ({drift_impact_score:.4f}) exceeds threshold ({threshold}). Retraining recommended.")
elif decay_detected:
    st.warning("⚠️ **DECAY WARNING**: Consistent performance degradation detected over multiple runs.")
else:
    st.success("🟢 System Status: Stable. No significant impact detected.")

# Layout: Feature Analysis & Trends
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🔍 Feature-Level Drift Analysis")
    st.dataframe(drift_summary_df.head(10).style.background_gradient(subset=['Unified Drift'], cmap='OrRd'), use_container_width=True)
    
    st.subheader("📈 Performance History (Moving Trend)")
    if len(st.session_state.perf_history) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.style.use('dark_background')
        ax.plot(st.session_state.perf_history, marker='o', color='#38bdf8', label='Batch Performance')
        # Moving average
        if len(st.session_state.perf_history) >= 3:
            ma = pd.Series(st.session_state.perf_history).rolling(window=3).mean()
            ax.plot(ma, linestyle='--', color='#f59e0b', label='Trend (SMA 3)')
        
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        ax.legend()
        st.pyplot(fig)

with col_right:
    st.subheader("📊 Drift Metric Breakdown")
    # Small bar chart of avg values
    metrics_avg = {
        'PSI': drift_summary_df['PSI'].mean(),
        'KL': drift_summary_df['KL'].mean(),
        'JS': drift_summary_df['JS'].mean()
    }
    st.bar_chart(metrics_avg)
    
    st.info("""
    **Insight Engine:**
    - **PSI**: Measures volume stability.
    - **KL/JS**: Measures statistical 'distance' between distributions.
    - **Impact Rank**: Prioritizes features that have high drift AND high influence on the model.
    """)

# Button to commit current data to the Rolling Baseline
if st.button("✅ Commit to Rolling Baseline"):
    st.session_state.rolling_buffer.append(X_new)
    if len(st.session_state.rolling_buffer) > window_size:
        st.session_state.rolling_buffer.pop(0)
    st.toast("Current batch added to reference buffer!")
