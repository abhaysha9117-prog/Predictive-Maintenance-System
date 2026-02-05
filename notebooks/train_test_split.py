import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load labeled data
df = pd.read_csv("../data/processed/train_labeled.csv")

# Unique engines
engine_ids = df["engine_id"].unique()

# Train-test split by engine
train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

# Create train and test sets
train_df = df[df["engine_id"].isin(train_engines)]
test_df = df[df["engine_id"].isin(test_engines)]

# Drop engine_id from features (not predictive)
X_train = train_df.drop(columns=["engine_id", "failure"])
y_train = train_df["failure"]

X_test = test_df.drop(columns=["engine_id", "failure"])
y_test = test_df["failure"]

# Save splits
X_train.to_csv("../data/processed/X_train.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

# Print shapes
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)
