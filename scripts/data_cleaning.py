import pandas as pd
import numpy as np

print("Loading engineered dataset...")

data = pd.read_csv("output/final_ml_dataset.csv")

print("Original dataset shape:", data.shape)

# =========================
# REMOVE DUPLICATES
# =========================

data = data.drop_duplicates()

print("After removing duplicates:", data.shape)

# =========================
# HANDLE MISSING VALUES
# =========================

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna()

print("After removing missing values:", data.shape)

# =========================
# REMOVE OUTLIERS (IQR METHOD)
# =========================

numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_cols:

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower) & (data[col] <= upper)]

print("After removing outliers:", data.shape)

# =========================
# FINAL DATASET SAVE
# =========================

data.to_csv("output/clean_ml_dataset.csv", index=False)

print("Clean dataset saved to output/clean_ml_dataset.csv")