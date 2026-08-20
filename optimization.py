# ============================================
# MODEL OPTIMIZATION — Single Tuned Gradient Boosting
# (kept as a COMPARISON baseline against the Stacking Ensemble in model_training.py)
# ============================================
# Purpose: train ONE optimized Gradient Boosting model on the SAME leakage-free,
# ternary-labeled data used by the ensemble, so we can fairly report
# "single best-tuned model vs stacking ensemble" in the paper.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import os
import joblib

class_names = ["Diabetic", "Non-Diabetic", "Pre-Diabetic"]  # label-encoder order

# ── 1. LOAD LEAKAGE-FREE DATA ──────────────
print("=== STEP 1: Loading train/test data (already leakage-free) ===")
train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")

X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── 2. HYPERPARAMETER TUNING (single GB model) ─
print("\n=== STEP 2: Hyperparameter Tuning (Randomized Search) ===")
print("This may take a few minutes. Please wait...")

# Use a subsample for the search step only (large GB fits are slow); the
# final best_estimator_ is then refit on the FULL training set in Step 3.
search_sample = train.sample(n=min(20000, len(train)), random_state=42)
X_search = search_sample.drop("target", axis=1)
y_search = search_sample["target"]

param_dist = {
    "n_estimators"      : [100, 150, 200],
    "learning_rate"     : [0.05, 0.1, 0.2],
    "max_depth"         : [3, 4, 5],
    "min_samples_split" : [2, 5, 10],
    "subsample"         : [0.7, 0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator          = gb,
    param_distributions= param_dist,
    n_iter             = 10,
    cv                 = 3,
    scoring            = "f1_macro",
    n_jobs             = -1,
    random_state       = 42,
    verbose            = 1
)

search.fit(X_search, y_search)

print(f"\nBest Parameters Found (from {len(search_sample)}-row search sample):")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

# ── 2b. REFIT BEST PARAMS ON FULL TRAINING SET ─
print("\n=== STEP 2b: Refitting best model on FULL training set ===")
best_model = GradientBoostingClassifier(random_state=42, **search.best_params_)
best_model.fit(X_train, y_train)

# ── 3. EVALUATE OPTIMIZED SINGLE MODEL ────────
print("\n=== STEP 3: Optimized Single-Model Results (ternary) ===")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec  = recall_score(y_test, y_pred, average="macro")
f1   = f1_score(y_test, y_pred, average="macro")
auc  = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"Accuracy  (single tuned GB) : {acc:.4f}")
print(f"Precision (macro)           : {prec:.4f}")
print(f"Recall    (macro)           : {rec:.4f}")
print(f"F1-Score  (macro)           : {f1:.4f}")
print(f"ROC-AUC   (macro, ovr)      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 4. SINGLE MODEL vs STACKING ENSEMBLE ─
print("\n=== STEP 4: Single Tuned GB vs Stacking Ensemble (both on leakage-free data) ===")
print(f"{'Metric':<12} {'Single-GB':>12} {'Ensemble':>12} {'Change':>10}")
print("-" * 50)

# Ensemble numbers copied from model_training.py's printed evaluation
ensemble = {"Accuracy": 0.9715, "Precision": 0.9670,
            "Recall": 0.8957, "F1-Score": 0.9239, "ROC-AUC": 0.9891}
single   = {"Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}

for metric in ensemble:
    diff  = ensemble[metric] - single[metric]
    arrow = "Ensemble higher" if diff > 0 else "Single higher" if diff < 0 else "Same"
    print(f"{metric:<12} {single[metric]:>12.4f} {ensemble[metric]:>12.4f} {arrow:>15}")

# ── 5. CONFUSION MATRIX ───────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Optimized Single GB Model — Confusion Matrix (Ternary)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("diabetes_prediction/optimized_confusion_matrix.png")
plt.close()

print("\n=== Optimization Complete! ===")

# ── 6. SAVE OPTIMIZED SINGLE MODEL ────────────
os.makedirs("diabetes_prediction/models", exist_ok=True)
joblib.dump(best_model, "diabetes_prediction/models/optimized_single_model.pkl")
print("\n=== Optimized single model saved -> models/optimized_single_model.pkl ===")