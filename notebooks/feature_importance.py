import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load training data
X_train = pd.read_csv("../data/processed/X_train.csv")

# Load saved model
rf_model = joblib.load("../models/random_forest_model.pkl")

# Get feature importance
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Print top 10 important features
print("Top 10 Important Features:")
print(feature_importance_df.head(10))

# Plot top 10
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df.head(10)["feature"],
    feature_importance_df.head(10)["importance"]
)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
