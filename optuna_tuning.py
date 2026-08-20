# ============================================
# OPTUNA-BASED ENSEMBLE HYPERPARAMETER TUNING
# ============================================
# Inspired by "Optuna-Optimized Machine Learning Technique for Accurate
# Diabetes Prediction and Classification" (ICSES 2024). The current stacking
# ensemble (model_training.py) uses MANUALLY-selected hyperparameters for its
# 3 base learners. This script uses Optuna (TPE sampler) to jointly search
# for better hyperparameters, then compares the Optuna-tuned ensemble against
# the manually-tuned one on the same held-out test set.

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

RANDOM_STATE = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console output readable

print("=== STEP 1: Loading leakage-free data ===")
train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")
X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

# ── Use a SUBSAMPLE + internal train/val split for the search phase only
# (keeps each Optuna trial fast); the FINAL best model is refit on the FULL
# training set afterward, and only THEN evaluated on the true test set.
search_sample = train.sample(n=min(20000, len(train)), random_state=RANDOM_STATE)
X_search, y_search = search_sample.drop("target", axis=1), search_sample["target"]
X_tr, X_val, y_tr, y_val = train_test_split(X_search, y_search, test_size=0.2,
                                             random_state=RANDOM_STATE, stratify=y_search)

# ============================================
# STEP 2: OPTUNA OBJECTIVE (jointly tunes all 3 base learners)
# ============================================
def objective(trial):
    gb = GradientBoostingClassifier(
        n_estimators=trial.suggest_int("gb_n_estimators", 50, 250),
        learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("gb_max_depth", 2, 6),
        subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 250),
        learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 50, 250),
        learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("lgbm_max_depth", 2, 8),
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )

    stack = StackingClassifier(
        estimators=[("gb", gb), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    y_pred = stack.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")

print("\n=== STEP 2: Running Optuna search (this will take a while) ===")
study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
N_TRIALS = 20
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest trial: F1={study.best_value:.4f}")
print("Best hyperparameters found:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ============================================
# STEP 3: REFIT BEST CONFIG ON FULL TRAINING DATA
# ============================================
print("\n=== STEP 3: Refitting Optuna-tuned ensemble on FULL training set ===")
bp = study.best_params
gb_final = GradientBoostingClassifier(n_estimators=bp["gb_n_estimators"],
                                       learning_rate=bp["gb_learning_rate"],
                                       max_depth=bp["gb_max_depth"],
                                       subsample=bp["gb_subsample"],
                                       random_state=RANDOM_STATE)
xgb_final = XGBClassifier(n_estimators=bp["xgb_n_estimators"],
                           learning_rate=bp["xgb_learning_rate"],
                           max_depth=bp["xgb_max_depth"],
                           eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
lgbm_final = LGBMClassifier(n_estimators=bp["lgbm_n_estimators"],
                             learning_rate=bp["lgbm_learning_rate"],
                             max_depth=bp["lgbm_max_depth"],
                             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)

optuna_stack = StackingClassifier(
    estimators=[("gb", gb_final), ("xgb", xgb_final), ("lgbm", lgbm_final)],
    final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1,
)
optuna_stack.fit(X_train, y_train)

# ============================================
# STEP 4: EVALUATE ON TRUE TEST SET
# ============================================
y_pred = optuna_stack.predict(X_test)
y_prob = optuna_stack.predict_proba(X_test)

optuna_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": f1_score(y_test, y_pred, average="macro"),  # placeholder overwritten below
    "Recall": recall_score(y_test, y_pred, average="macro"),
    "F1-Score": f1_score(y_test, y_pred, average="macro"),
    "Diabetic_Recall": recall_score(y_test, y_pred, labels=[0], average="macro"),
}
from sklearn.metrics import precision_score, roc_auc_score
optuna_metrics["Precision"] = precision_score(y_test, y_pred, average="macro")
optuna_metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print("\n=== STEP 5: Optuna-Tuned Ensemble vs Manually-Tuned Ensemble ===")
manual_metrics = {"Accuracy": 0.9715, "Precision": 0.9670, "Recall": 0.8957,
                   "F1-Score": 0.9239, "ROC-AUC": 0.9891, "Diabetic_Recall": 0.6894}

print(f"{'Metric':<18} {'Manual':>10} {'Optuna':>10} {'Better':>10}")
print("-" * 52)
for metric in manual_metrics:
    m, o = manual_metrics[metric], optuna_metrics[metric]
    better = "Optuna" if o > m else "Manual" if m > o else "Tie"
    print(f"{metric:<18} {m:>10.4f} {o:>10.4f} {better:>10}")

import joblib
joblib.dump(optuna_stack, "diabetes_prediction/models/optuna_stacking_model.pkl")
print("\nSaved -> diabetes_prediction/models/optuna_stacking_model.pkl")
print("\n=== Optuna Tuning Complete! ===")