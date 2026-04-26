import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score
import joblib
import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_psi(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    expected_min = expected.min()
    expected_max = expected.max()
    bins_edges = np.linspace(expected_min, expected_max + 1e-5, bins + 1)
    
    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts, _ = np.histogram(actual, bins=bins_edges)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)

def detect_drift_psi(baseline_df, new_df):
    results = []
    numerical_cols = baseline_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in new_df.columns:
            expected = baseline_df[col].dropna().values
            actual = new_df[col].dropna().values
            if len(expected) > 0 and len(actual) > 0:
                psi = calculate_psi(expected, actual)
                results.append({'Feature': col, 'PSI': psi})
    
    df_report = pd.DataFrame(results)
    if not df_report.empty:
        df_report = df_report.sort_values(by='PSI', ascending=False)
    return df_report

def run_monitoring(model_path='model.pkl', scaler_path='scaler.pkl', 
                   imputer_path='imputer.pkl', baseline_path='baseline.pkl', 
                   new_data_path='new_data.csv', target_col='default.payment.next.month'):
    logging.info("Loading artifacts...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    baseline = joblib.load(baseline_path)
    
    baseline_df = baseline['X_train']
    baseline_f1 = baseline['baseline_f1']
    baseline_recall = baseline['baseline_recall']
    
    logging.info(f"Loading new data from {new_data_path}")
    new_df = pd.read_csv(new_data_path)
    if 'ID' in new_df.columns:
        new_df = new_df.drop(columns=['ID'])
        
    y_true = None
    if target_col in new_df.columns:
        y_true = new_df[target_col]
        X_new = new_df.drop(columns=[target_col])
    else:
        X_new = new_df.copy()
        
    logging.info("Computing PSI Drift...")
    drift_report = detect_drift_psi(baseline_df, X_new)
    drift_score = drift_report['PSI'].mean() if not drift_report.empty else 0.0
    
    logging.info("Preprocessing new data...")
    X_new_imputed = imputer.transform(X_new)
    X_new_scaled = scaler.transform(X_new_imputed)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns)
    
    logging.info("Predicting...")
    y_pred = model.predict(X_new_scaled_df)
    
    current_f1 = None
    current_recall = None
    performance_drop = 0.0
    
    if y_true is not None:
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        performance_drop = baseline_f1 - current_f1
    
    # Drift impact score
    if current_f1 is not None:
        drift_impact_score = (0.6 * drift_score) + (0.4 * performance_drop)
    else:
        drift_impact_score = drift_score
        
    # Performance History
    performance_history = []
    if os.path.exists('performance_history.pkl'):
        performance_history = joblib.load('performance_history.pkl')
        
    if current_f1 is not None:
        performance_history.append(current_f1)
        joblib.dump(performance_history, 'performance_history.pkl')
        
    trend_status = "Not enough data"
    trend_val = 0.0
    if len(performance_history) >= 2:
        last_n = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        trend_val = np.polyfit(range(len(last_n)), last_n, 1)[0]
        if trend_val < 0:
            trend_status = "Model degrading"
        else:
            trend_status = "Stable/Improving"
            
    impact_threshold = 0.05
    alert_triggered = drift_impact_score > impact_threshold
    
    logging.info(f"Drift Score (Avg PSI): {drift_score:.4f}")
    if current_f1 is not None:
        logging.info(f"Performance F1    : {current_f1:.4f} (Drop: {performance_drop:.4f})")
    logging.info(f"Drift Impact Score: {drift_impact_score:.4f}")
    logging.info(f"Trend Analysis    : {trend_status}")
    logging.info(f"Alert Triggered   : {'Yes' if alert_triggered else 'No'}")
    
    # Output to stdout or save to csv if needed, but returning dict is good
    return {
        'drift_score': drift_score,
        'drift_report': drift_report,
        'current_f1': current_f1,
        'current_recall': current_recall,
        'performance_drop': performance_drop,
        'drift_impact_score': drift_impact_score,
        'trend_status': trend_status,
        'trend_val': trend_val,
        'alert_triggered': alert_triggered
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run monitoring pipeline")
    parser.add_argument('--new_data', default='new_data.csv', help='Path to new data file')
    args = parser.parse_args()
    if os.path.exists(args.new_data):
        run_monitoring(new_data_path=args.new_data)
    else:
        logging.error(f"New data file {args.new_data} not found.")
