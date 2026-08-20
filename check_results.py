import pandas as pd, joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report, confusion_matrix)

test = pd.read_csv("diabetes_prediction/test_final.csv")
X_test, y_test = test.drop("target", axis=1), test["target"]
model = joblib.load("diabetes_prediction/models/stacking_model.pkl")

pred = model.predict(X_test)
proba = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision (macro):", precision_score(y_test, pred, average="macro"))
print("Recall (macro):", recall_score(y_test, pred, average="macro"))
print("F1 (macro):", f1_score(y_test, pred, average="macro"))
print("ROC-AUC (macro, ovr):", roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
print()
print(classification_report(y_test, pred, target_names=["Diabetic","Non-Diabetic","Pre-Diabetic"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))