# ============================================
# MODEL TRAINING — Stacking Ensemble (Ternary Classification)
# ============================================
# Base learners : Gradient Boosting, XGBoost, LightGBM
# Meta learner  : Logistic Regression
# Target        : 0=Diabetic, 1=Non-Diabetic, 2=Pre-Diabetic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)
import os
import joblib

RANDOM_STATE = 42

# ── 1. LOAD DATA ───────────────────────────────
train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")

X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

class_names = ["Diabetic", "Non-Diabetic", "Pre-Diabetic"]  # label-encoder order (alphabetical)

# ── 2. DEFINE BASE LEARNERS ────────────────────
print("=== STEP 1: Building Stacking Ensemble ===")
base_learners = [
    ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, subsample=0.8, random_state=RANDOM_STATE)),
    ("xgb", XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                           eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)),
    ("lgbm", LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
]

meta_learner = LogisticRegression(max_iter=1000)

stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=3,
    n_jobs=-1,
    passthrough=False,
)

# ── 3. TRAIN ────────────────────────────────────
print("Training stacking ensemble (this may take a few minutes)...")
stack_model.fit(X_train, y_train)
print("Training complete!")

# ── 4. PREDICT & EVALUATE ──────────────────────
y_pred = stack_model.predict(X_test)
y_prob = stack_model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print("\n=== STEP 2: Stacking Ensemble Evaluation (macro-averaged) ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 5. CONFUSION MATRIX ────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — Stacking Ensemble (Ternary)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("diabetes_prediction/confusion_matrix.png")
plt.close()

# ── 6. INDIVIDUAL BASE-LEARNER COMPARISON ──────
print("\n=== STEP 3: Individual Base Learner Performance (for comparison) ===")
for name, clf in stack_model.named_estimators_.items():
    pred = clf.predict(X_test)
    a = accuracy_score(y_test, pred)
    f = f1_score(y_test, pred, average="macro")
    print(f"{name:6s} -> Accuracy: {a:.4f} | Macro-F1: {f:.4f}")

# ── 7. SAVE MODEL ──────────────────────────────
os.makedirs("diabetes_prediction/models", exist_ok=True)
joblib.dump(stack_model, "diabetes_prediction/models/stacking_model.pkl")
print("\n=== Stacking ensemble saved -> models/stacking_model.pkl ===")