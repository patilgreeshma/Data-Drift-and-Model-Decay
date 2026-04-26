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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save(filepath='UCI_Credit_Card.csv', target_col='default.payment.next.month'):
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    logging.info("Preprocessing data...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    
    logging.info("Evaluating baseline model...")
    y_pred = model.predict(X_test_scaled)
    baseline_f1 = f1_score(y_test, y_pred, zero_division=0)
    baseline_recall = recall_score(y_test, y_pred, zero_division=0)
    
    baseline_data = {
        'X_train': X_train.copy(),
        'baseline_f1': baseline_f1,
        'baseline_recall': baseline_recall
    }
    
    logging.info(f"Baseline F1: {baseline_f1:.4f}, Recall: {baseline_recall:.4f}")
    
    logging.info("Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(baseline_data, 'baseline.pkl')
    logging.info("Training complete. Artifacts saved: model.pkl, scaler.pkl, imputer.pkl, baseline.pkl")

if __name__ == '__main__':
    train_and_save()
