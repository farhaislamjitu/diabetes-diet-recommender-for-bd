# ============================================
# DUAL EXPLAINABILITY — SHAP (Global) + LIME (Local)
# ============================================
# SHAP : applied to the best-performing tree base-learner (TreeExplainer)
#        -> fast, exact, gives GLOBAL feature importance across the dataset
# LIME : applied to the full stacking ensemble (model-agnostic)
#        -> gives LOCAL, per-patient explanation of the final ensemble decision
# Together they give both a population-level and an individual-level view.

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ── 1. LOAD MODEL & DATA ───────────────────────
print("=== STEP 1: Loading model and data ===")
stack_model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
class_names = list(target_le.classes_)  # ['Diabetic','Non-Diabetic','Pre-Diabetic']

train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")
X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

feature_names = X_train.columns.tolist()
print(f"Features: {feature_names}")
print(f"Classes : {class_names}")

# ── 2. SHAP — GLOBAL EXPLANATION (best base learner) ─
print("\n=== STEP 2: SHAP (Global) — using best tree base-learner ===")
best_base_name = "xgb"  # chosen from model_training.py comparison (swap if a different base wins)
best_base_model = stack_model.named_estimators_[best_base_name]

explainer_shap = shap.TreeExplainer(best_base_model)
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values = explainer_shap.shap_values(X_sample)

plt.figure()
# multiclass shap_values -> list of arrays (one per class) for some tree explainers
if isinstance(shap_values, list):
    shap.summary_plot(shap_values, X_sample, class_names=class_names, show=False)
else:
    shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("diabetes_prediction/shap_summary.png")
plt.close()
print("Saved -> diabetes_prediction/shap_summary.png")

joblib.dump(explainer_shap, "diabetes_prediction/models/shap_explainer.pkl")

# ── 3. LIME — LOCAL EXPLANATION (full stacking ensemble) ─
print("\n=== STEP 3: LIME (Local) — explaining full stacking ensemble ===")
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    random_state=42,
)

# Explain one example patient from the test set
patient_idx = 0
patient = X_test.iloc[[patient_idx]]
pred_class_idx = stack_model.predict(patient)[0]
pred_proba = stack_model.predict_proba(patient)[0]

print(f"\nPatient data:\n{patient}")
print(f"Predicted class : {class_names[pred_class_idx]}")
print(f"Class probabilities: "
      f"{dict(zip(class_names, np.round(pred_proba, 4)))}")

lime_exp = lime_explainer.explain_instance(
    data_row=patient.values[0],
    predict_fn=stack_model.predict_proba,
    num_features=len(feature_names),
    top_labels=1,
)

print("\nLIME feature contributions for predicted class "
      f"'{class_names[pred_class_idx]}':")
for feature, weight in lime_exp.as_list(label=pred_class_idx):
    direction = "increases" if weight > 0 else "decreases"
    print(f"  {feature}: {weight:.4f}  ({direction} likelihood)")

lime_exp.save_to_file("diabetes_prediction/lime_explanation_patient0.html")
print("\nSaved -> diabetes_prediction/lime_explanation_patient0.html")

print("\n=== Dual Explainability (SHAP global + LIME local) Complete! ===")