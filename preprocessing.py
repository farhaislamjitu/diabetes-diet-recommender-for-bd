# ============================================
# PREPROCESSING — Ternary Label + Leakage-Free Pipeline
# ============================================
# Fixes vs v1:
#   1. Train/test split happens FIRST (before SMOTE/scaling/feature selection)
#   2. SMOTE fit only on training data
#   3. Scaler fit only on training data
#   4. Target converted to ternary: Non-Diabetic / Pre-Diabetic / Diabetic
#      - diabetes==1 (original)      -> "Diabetic"        (ground truth preserved)
#      - diabetes==0 AND HbA1c>=5.7 AND Glucose>=100 -> "Pre-Diabetic"
#      - diabetes==0 otherwise       -> "Non-Diabetic"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import joblib

RANDOM_STATE = 42

# ── 1. LOAD DATASET ───────────────────────────
df = pd.read_csv("diabetes_prediction/diabetes_clinical_dataset.csv")
print("=== STEP 1: Original Shape ===")
print(df.shape)

# ── 2. CREATE TERNARY LABEL (before dropping anything) ─────
def make_ternary(row):
    if row["diabetes"] == 1:
        return "Diabetic"
    if row["hbA1c_level"] >= 5.7 and row["blood_glucose_level"] >= 100:
        return "Pre-Diabetic"
    return "Non-Diabetic"

df["diabetes_ternary"] = df.apply(make_ternary, axis=1)
print("\n=== STEP 2: Ternary Label Distribution ===")
print(df["diabetes_ternary"].value_counts())

# ── 3. DROP IRRELEVANT COLUMNS ────────────────
drop_cols = ["year", "location", "clinical_notes", "diabetes"]  # drop old binary target too
df = df.drop(columns=drop_cols)

# ── 4. ENCODE CATEGORICAL COLUMNS ─────────────
print("\n=== STEP 3: Encoding Categorical Columns ===")
df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})

le_smoke = LabelEncoder()
df["smoking_history"] = le_smoke.fit_transform(df["smoking_history"])
print(f"Smoking history encoded -> classes: {list(le_smoke.classes_)}")

le_target = LabelEncoder()
df["diabetes_ternary"] = le_target.fit_transform(df["diabetes_ternary"])
print(f"Target encoded -> classes: {list(le_target.classes_)}  (0={le_target.classes_[0]}, "
      f"1={le_target.classes_[1]}, 2={le_target.classes_[2]})")

os.makedirs("diabetes_prediction/models", exist_ok=True)
joblib.dump(le_target, "diabetes_prediction/models/target_label_encoder.pkl")
joblib.dump(le_smoke, "diabetes_prediction/models/smoking_label_encoder.pkl")

# ── 5. SEPARATE FEATURES & TARGET ─────────────
X = df.drop("diabetes_ternary", axis=1)
y = df["diabetes_ternary"]

# ── 6. TRAIN/TEST SPLIT — BEFORE SMOTE/SCALING (leakage fix) ─
print("\n=== STEP 4: Train/Test Split (80:20, stratified) ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print("Train class dist:\n", y_train.value_counts())
print("Test class dist:\n", y_test.value_counts())

# ── 7. SMOTE — FIT ON TRAIN ONLY ──────────────
print("\n=== STEP 5: SMOTE (train set only) ===")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Before SMOTE (train):\n", y_train.value_counts())
print("After SMOTE (train):\n", y_train_res.value_counts())

# ── 8. SCALING — FIT ON TRAIN ONLY, TRANSFORM BOTH ─
print("\n=== STEP 6: Feature Scaling (fit on train only) ===")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
joblib.dump(scaler, "diabetes_prediction/models/scaler.pkl")
print("Scaling done. Scaler fit on training data only.")

# ── 9. VISUALIZATION: class balance before/after ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y_train.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["steelblue","orange","green"])
axes[0].set_title("Train — Before SMOTE")
axes[0].set_xticklabels(le_target.classes_, rotation=0)
y_train_res.value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["steelblue","orange","green"])
axes[1].set_title("Train — After SMOTE")
axes[1].set_xticklabels(le_target.classes_, rotation=0)
plt.suptitle("Ternary Class Distribution (Train Set Only)")
plt.tight_layout()
plt.savefig("diabetes_prediction/class_balance.png")
plt.close()

# ── 10. SAVE SPLIT DATA FOR NEXT STAGES ───────
X_train_scaled["target"] = y_train_res.values
X_test_scaled["target"] = y_test.values
X_train_scaled.to_csv("diabetes_prediction/train.csv", index=False)
X_test_scaled.to_csv("diabetes_prediction/test.csv", index=False)
print("\n=== Saved: train.csv, test.csv ===")
print("Preprocessing Complete! (leakage-free, ternary target)")