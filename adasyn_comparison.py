# ============================================
# ADASYN vs SMOTE COMPARISON (Class Imbalance Handling)
# ============================================
# Purpose: reproduce the SAME leakage-free train/test split used in
# preprocessing.py, but resample the training set with ADASYN instead of
# SMOTE, then train the same Stacking Ensemble architecture and compare
# against the SMOTE-based results -- especially the Diabetic class recall,
# since ADASYN focuses extra synthetic samples on harder-to-classify
# (borderline) minority examples.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)

RANDOM_STATE = 42
class_names = ["Diabetic", "Non-Diabetic", "Pre-Diabetic"]

# ── 1. LOAD & PREPARE DATA (identical to preprocessing.py) ────
print("=== STEP 1: Reproducing the same leakage-free split ===")
df = pd.read_csv("diabetes_prediction/diabetes_clinical_dataset.csv")

def make_ternary(row):
    if row["diabetes"] == 1:
        return "Diabetic"
    if row["hbA1c_level"] >= 5.7 and row["blood_glucose_level"] >= 100:
        return "Pre-Diabetic"
    return "Non-Diabetic"

df["diabetes_ternary"] = df.apply(make_ternary, axis=1)
df = df.drop(columns=["year", "location", "clinical_notes", "diabetes"])
df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})
df["smoking_history"] = LabelEncoder().fit_transform(df["smoking_history"])
df["diabetes_ternary"] = LabelEncoder().fit_transform(df["diabetes_ternary"])  # Diabetic=0, Non-Diabetic=1, Pre-Diabetic=2

X = df.drop("diabetes_ternary", axis=1)
y = df["diabetes_ternary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape} (identical split to preprocessing.py)")

# only the 4 features selected in feature_selection.py are used, for a fair
# apples-to-apples comparison against the reported SMOTE ensemble results
top_features = ["hbA1c_level", "blood_glucose_level", "age", "bmi"]

def build_and_evaluate(resampler, name):
    print(f"\n=== Resampling with {name} ===")
    X_res, y_res = resampler.fit_resample(X_train, y_train)
    print(f"After {name} (train class counts):")
    print(y_res.value_counts())

    scaler = StandardScaler()
    X_res_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)[top_features]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)[top_features]

    base_learners = [
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           subsample=0.8, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(n_estimators=150, max_depth=5, eval_metric="mlogloss",
                               random_state=RANDOM_STATE, n_jobs=-1)),
        ("lgbm", LGBMClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE,
                                 verbose=-1, n_jobs=-1)),
    ]
    stack = StackingClassifier(estimators=base_learners,
                                final_estimator=LogisticRegression(max_iter=1000),
                                cv=3, n_jobs=-1)
    print(f"Training Stacking Ensemble ({name} balanced data)...")
    stack.fit(X_res_scaled, y_res)

    y_pred = stack.predict(X_test_scaled)
    y_prob = stack.predict_proba(X_test_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-Score": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
    }
    print(f"\n{name} Ensemble Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Diabetic-class-specific recall (index 0) -- our known weak spot
    diabetic_recall = recall_score(y_test, y_pred, labels=[0], average="macro")
    metrics["Diabetic_Recall"] = diabetic_recall
    return metrics

# ── 2. RUN SMOTE (reference) ───────────────────
smote_metrics = build_and_evaluate(SMOTE(random_state=RANDOM_STATE), "SMOTE")

# ── 3. RUN ADASYN ───────────────────────────────
adasyn_metrics = build_and_evaluate(ADASYN(random_state=RANDOM_STATE), "ADASYN")

# ── 4. SIDE-BY-SIDE COMPARISON ─────────────────
print("\n" + "=" * 55)
print("=== FINAL COMPARISON: SMOTE vs ADASYN ===")
print("=" * 55)
print(f"{'Metric':<18} {'SMOTE':>10} {'ADASYN':>10} {'Better':>10}")
print("-" * 55)
for metric in smote_metrics:
    s, a = smote_metrics[metric], adasyn_metrics[metric]
    better = "ADASYN" if a > s else "SMOTE" if s > a else "Tie"
    print(f"{metric:<18} {s:>10.4f} {a:>10.4f} {better:>10}")