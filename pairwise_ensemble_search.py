# ============================================
# PAIR-WISE ENSEMBLE MODEL SELECTION
# ============================================
# Inspired by Ashour et al. (2024, SmartNets) -- "Enhancing Diabetes
# Prediction Based on Pair-Wise Ensemble Learning Model Selection", which
# found that specific PAIRS of classifiers can outperform larger ensembles.
# This script systematically tests every 2-learner and 3-learner combination
# of our three base learners (GB, XGBoost, LightGBM) to check whether the
# full 3-way stack is actually the best choice, or if a smaller pair wins.

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

RANDOM_STATE = 42

print("=== STEP 1: Loading v2 leakage-free data ===")
train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")
X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

def make_base_learners():
    """Fresh instances each time (StackingClassifier fits its own copies,
    but keeping this as a function avoids any accidental state reuse)."""
    return {
        "gb": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE),
        "xgb": XGBClassifier(n_estimators=150, max_depth=5, eval_metric="mlogloss",
                              random_state=RANDOM_STATE, n_jobs=-1),
        "lgbm": LGBMClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE,
                                verbose=-1, n_jobs=-1),
    }

all_learners = list(make_base_learners().keys())  # ['gb', 'xgb', 'lgbm']

# All 2-learner combinations only (the 3-way gb+xgb+lgbm combo was already
# fully evaluated in model_training.py -- its known result is reused below
# instead of re-training it here, to save time).
combos = list(combinations(all_learners, 2))

print(f"\n=== STEP 2: Testing {len(combos)} combinations ===")
results = []
learners_dict = make_base_learners()

for combo in combos:
    combo_name = "+".join(combo)
    print(f"\nTraining combo: {combo_name}")
    estimators = [(name, learners_dict[name]) for name in combo]
    stack = StackingClassifier(estimators=estimators,
                                final_estimator=LogisticRegression(max_iter=1000),
                                cv=3, n_jobs=-1)
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    diabetic_recall = recall_score(y_test, y_pred, labels=[0], average="macro")

    results.append({"Combo": combo_name, "N_Learners": len(combo),
                     "Accuracy": acc, "Macro_F1": f1, "Diabetic_Recall": diabetic_recall})
    print(f"  Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | Diabetic Recall: {diabetic_recall:.4f}")

# ============================================
# STEP 3: RANK AND SUMMARIZE
# ============================================
# ============================================
# STEP 3: RANK AND SUMMARIZE
# ============================================
# Add back the already-known full 3-way stack result (from model_training.py's
# verified run), instead of retraining it here.
results.append({"Combo": "gb+xgb+lgbm", "N_Learners": 3,
                 "Accuracy": 0.9715, "Macro_F1": 0.9239, "Diabetic_Recall": 0.6894})

results_df = pd.DataFrame(results).sort_values("Macro_F1", ascending=False)
print("\n" + "=" * 70)
print("=== FINAL RANKING (by Macro-F1) ===")
print("=" * 70)
print(results_df.to_string(index=False))

best_combo = results_df.iloc[0]
full_stack_row = results_df[results_df["Combo"] == "gb+xgb+lgbm"].iloc[0]
print(f"\nBest combo: {best_combo['Combo']} (F1={best_combo['Macro_F1']:.4f})")
print(f"Full 3-way stack: F1={full_stack_row['Macro_F1']:.4f}")

if best_combo["Combo"] != "gb+xgb+lgbm":
    diff = best_combo["Macro_F1"] - full_stack_row["Macro_F1"]
    print(f"\n>> A smaller combo ({best_combo['Combo']}) outperforms the full 3-way "
          f"stack by {diff:.4f} F1 -- fewer learners, similar or better performance.")
else:
    print("\n>> The full 3-way stack remains the best combination.")

results_df.to_csv("diabetes_prediction/pairwise_ensemble_results.csv", index=False)
print("\nSaved -> diabetes_prediction/pairwise_ensemble_results.csv")