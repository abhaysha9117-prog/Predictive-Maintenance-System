import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# Load test data
X_test = pd.read_csv("../data/processed/X_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

# Load model
rf_model = joblib.load("../models/random_forest_model.pkl")

# Predict probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.5, 0.4, 0.3, 0.2]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    print(f"\n=== Threshold: {t} ===")
    print(classification_report(y_test, y_pred))
