import joblib
import pandas as pd
import json
import os
from django.shortcuts import render
from django.conf import settings

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, "predictor", "random_forest_model.pkl")
FEATURE_PATH = os.path.join(settings.BASE_DIR, "predictor", "feature_names.json")

# Load model
model = joblib.load(MODEL_PATH)

# Load exact feature order
with open(FEATURE_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

def home(request):
    result = None

    if request.method == "POST":
        # User inputs
        cycle = float(request.POST.get("cycle"))
        sensor_2 = float(request.POST.get("sensor_2"))
        sensor_3 = float(request.POST.get("sensor_3"))
        sensor_4 = float(request.POST.get("sensor_4"))

        # Initialize ALL features with 0
        data = {feature: 0.0 for feature in FEATURE_NAMES}

        # Fill known values
        data["cycle"] = cycle
        data["sensor_2"] = sensor_2
        data["sensor_3"] = sensor_3
        data["sensor_4"] = sensor_4

        # Estimate RUL from cycle (important!)
        if "RUL" in data:
            data["RUL"] = max(0, 400 - cycle)

        # Create DataFrame with EXACT column order
        input_df = pd.DataFrame(
            [[data[f] for f in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        # Predict probability
        prob_failure = model.predict_proba(input_df)[0][1]

        # Custom threshold
        THRESHOLD = 0.30

        if prob_failure >= THRESHOLD:
            result = f"⚠ FAILURE SOON (Risk: {prob_failure:.2%})"
        else:
            result = f"✅ NORMAL (Risk: {prob_failure:.2%})"

    return render(request, "home.html", {"result": result})
