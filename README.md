# 🛡️ Data Drift & Model Decay Monitoring System

An adaptive ML observability framework designed to detect **Data Drift**, **Model Decay**, and **Performance Degradation** in real-time. This project provides both a CLI pipeline and a high-fidelity Streamlit dashboard for monitoring machine learning models in production.

## 🚀 Overview

In production ML, data distributions often shift over time (Data Drift), leading to a decline in model accuracy (Model Decay). This system implements statistical checks (PSI, KL Divergence, JS Divergence) to identify these shifts early and trigger alerts before the model becomes unreliable.

### Key Features
- **Multi-Metric Drift Detection**: Calculates Population Stability Index (PSI), KL Divergence, and Jensen-Shannon Divergence.
- **Automated Decay Analysis**: Tracks performance trends (F1-score, Recall) and uses linear regression to detect consistent degradation.
- **Unified Drift Score**: Combines feature importance with statistical drift to prioritize high-impact feature shifts.
- **Dynamic Baseline**: Support for rolling baselines to adapt to expected seasonal shifts.
- **Interactive Dashboard**: Premium UI built with Streamlit for visual monitoring and alert management.

---

## 🛠️ Project Structure

```bash
├── app.py                # Streamlit Dashboard (UI)
├── train.py              # Model training & artifact generation
├── monitor.py            # CLI monitoring pipeline
├── UCI_Credit_Card.csv   # Baseline dataset (Example)
├── new_data.csv          # Inference dataset for testing
├── model.pkl             # Trained Random Forest model
├── baseline.pkl          # Training distribution metadata
├── scaler.pkl            # Preprocessing: StandardScaler
└── imputer.pkl           # Preprocessing: SimpleImputer
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📖 How to Use

### Step 1: Train the Baseline Model
Execute the training script to generate the model and baseline metadata.
```bash
python train.py
```
*This will create `model.pkl`, `scaler.pkl`, `imputer.pkl`, and `baseline.pkl`.*

### Step 2: Run Monitoring (CLI)
Analyze a new dataset for drift using the command line.
```bash
python monitor.py --new_data new_data.csv
```

### Step 3: Launch the Dashboard
Start the interactive UI to visualize drift metrics and model health.
```bash
streamlit run app.py
```

---

## 📊 Dashboard Features

- **Unified Drift Score**: A normalized score representing the overall "health" of your data distribution.
- **Drift Impact Score**: Weighs drift severity against actual performance drops.
- **Feature-Level Analysis**: Identifies exactly which features are drifting and their impact on the model.
- **Performance History**: Visualizes how model metrics (F1-score) evolve over time across multiple batches.
- **Alert System**: Triggers "Critical" or "Warning" status based on configurable thresholds.

---

## 🔬 Statistical Metrics Used

1. **PSI (Population Stability Index)**: Measures the magnitude of shift in the distribution of a variable between two points in time.
   - `PSI < 0.1`: No significant change.
   - `0.1 < PSI < 0.25`: Minor shift detected.
   - `PSI > 0.25`: Major shift; action required.
2. **KL Divergence**: Measures how one probability distribution differs from a second, reference distribution.
3. **Model Health Trend**: Uses the slope of the performance history to predict if the model is in a state of consistent decay.

---


