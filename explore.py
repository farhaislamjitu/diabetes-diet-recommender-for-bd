# ============================================
# NEW DATASET - EXPLORATION
# ============================================

import pandas as pd
import numpy as np

# ── 1. LOAD DATASET ───────────────────────────
df = pd.read_csv("diabetes_prediction/diabetes_clinical_dataset.csv")

# ── 2. BASIC INFO ─────────────────────────────
print("=== Shape (rows x columns) ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Target Column Value Counts ===")
# Check last column or any column named 'diabetes' or 'Outcome'
for col in df.columns:
    if "diab" in col.lower() or "outcome" in col.lower() or "target" in col.lower():
        print(f"\n'{col}' distribution:")
        print(df[col].value_counts())