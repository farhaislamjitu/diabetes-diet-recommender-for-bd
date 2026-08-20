import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)

# Load leakage-free train/test data (same as used for the stacking ensemble)
train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")

X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

# Same base learners as the stacking ensemble, combined via soft voting instead of a meta-learner
gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42)
xgb = XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, eval_metric="mlogloss", random_state=42)
lgbm = LGBMClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)

voting_clf = VotingClassifier(
    estimators=[("gb", gb), ("xgb", xgb), ("lgbm", lgbm)],
    voting="soft"
)

voting_clf.fit(X_train, y_train)
pred = voting_clf.predict(X_test)
proba = voting_clf.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision (macro):", precision_score(y_test, pred, average="macro"))
print("Recall (macro):", recall_score(y_test, pred, average="macro"))
print("F1 (macro):", f1_score(y_test, pred, average="macro"))
print("ROC-AUC (macro, ovr):", roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
print()
print(classification_report(y_test, pred, target_names=["Diabetic","Non-Diabetic","Pre-Diabetic"]))

# Save for reuse (optional)
joblib.dump(voting_clf, "diabetes_prediction/models/voting_classifier.pkl")