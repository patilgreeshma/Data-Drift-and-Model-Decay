import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
import joblib
import logging
import warnings
import argparse

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save(filepath, target_col, prefix):
    logging.info(f"--- Training for {prefix} dataset ---")
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Clean data (common steps)
    df = df.replace('?', np.nan)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logging.info("Encoding categorical features...")
    encoders = {}
    
    # Encode Target if categorical
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        encoders['target'] = le_y
    
    # Encode Features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    logging.info("Preprocessing...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    logging.info("Training Model & Performing 5-Fold Cross-Validation...")
    model = XGBClassifier(random_state=42, n_estimators=100)
    
    # Perform CV on Training Data
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    logging.info(f"CV F1 Scores: {cv_scores}")
    logging.info(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Final Fit
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logging.info(f"Final Test F1 Score: {f1:.4f}")
    
    baseline_data = {
        'X_train': X_train.copy(),
        'baseline_f1': f1,
        'target_col': target_col
    }

    logging.info(f"Saving artifacts with prefix: {prefix}_")
    joblib.dump(model, f'{prefix}_model.pkl')
    joblib.dump(scaler, f'{prefix}_scaler.pkl')
    joblib.dump(imputer, f'{prefix}_imputer.pkl')
    joblib.dump(baseline_data, f'{prefix}_baseline.pkl')
    joblib.dump(encoders, f'{prefix}_encoders.pkl')
    logging.info(f"Done. Files: {prefix}_model.pkl, {prefix}_scaler.pkl, etc.")

if __name__ == '__main__':
    # Train Adult Dataset
    train_and_save(
        filepath='/Users/greeshmapatil/Desktop/ads/adult.csv', 
        target_col='income', 
        prefix='adult'
    )
    
    # Train Credit Card Dataset
    train_and_save(
        filepath='/Users/greeshmapatil/Desktop/ads/UCI_Credit_Card.csv', 
        target_col='default.payment.next.month', 
        prefix='credit'
    )
