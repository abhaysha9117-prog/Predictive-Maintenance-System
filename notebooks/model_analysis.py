import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data
X_train = pd.read_csv("../data/processed/X_train.csv")

# Load trained model
rf_model = joblib.load("../models/random_forest_model.pkl") if False else None
