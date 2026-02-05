import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("../data/processed/train_cleaned.csv")

# Calculate max cycle per engine
max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

# Merge back to main dataframe
df = df.merge(max_cycle, on="engine_id", how="left")

# Remaining Useful Life (RUL)
df["RUL"] = df["max_cycle"] - df["cycle"]

# Create binary failure label
THRESHOLD = 30
df["failure"] = df["RUL"].apply(lambda x: 1 if x <= THRESHOLD else 0)

# Drop helper column
df.drop(columns=["max_cycle"], inplace=True)

# Save final labeled dataset
df.to_csv("../data/processed/train_labeled.csv", index=False)

# Quick checks
print("Dataset shape:", df.shape)
print("\nFailure label distribution:")
print(df["failure"].value_counts())
