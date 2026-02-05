import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

# Column names (correct order)
columns = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load data correctly
train_df = pd.read_csv(
    "../data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Initial shape:", train_df.shape)

# Ensure correct data types
train_df["engine_id"] = train_df["engine_id"].astype(int)
train_df["cycle"] = train_df["cycle"].astype(int)

# Check constant sensors
sensor_cols = [col for col in train_df.columns if "sensor_" in col]

constant_sensors = [
    col for col in sensor_cols
    if train_df[col].std() == 0
]

print("\nConstant Sensors (to remove):")
print(constant_sensors)

# Drop constant sensors
train_df.drop(columns=constant_sensors, inplace=True)

print("\nShape after dropping constant sensors:", train_df.shape)

# Save cleaned data
train_df.to_csv("../data/processed/train_cleaned.csv", index=False)

print("\nCleaned data saved to data/processed/train_cleaned.csv")
