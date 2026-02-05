import joblib
import pandas as pd
import json

# Load model
model = joblib.load("maintenance_web/predictor/random_forest_model.pkl")

# Load feature names
with open("maintenance_web/predictor/feature_names.json", "r") as f:
    FEATURE_NAMES = json.load(f)

# Terminal inputs
cycle = float(input("Enter cycle: "))
sensor_2 = float(input("Enter sensor_2: "))
sensor_3 = float(input("Enter sensor_3: "))
sensor_4 = float(input("Enter sensor_4: "))

# Initialize all features
data = {f: 0.0 for f in FEATURE_NAMES}
data["cycle"] = cycle
data["sensor_2"] = sensor_2
data["sensor_3"] = sensor_3
data["sensor_4"] = sensor_4

# Estimate RUL
if "RUL" in data:
    data["RUL"] = max(0, 400 - cycle)

# Create DataFrame
input_df = pd.DataFrame([[data[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

# Predict probability
prob_failure = model.predict_proba(input_df)[0][1]

# Result
if prob_failure >= 0.30:
    print(f"⚠ FAILURE SOON (Risk: {prob_failure:.2%})")
else:
    print(f"✅ NORMAL (Risk: {prob_failure:.2%})")
