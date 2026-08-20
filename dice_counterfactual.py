# ============================================
# DiCE COUNTERFACTUAL EXPLANATION LAYER
# ============================================
# Purpose: unlike SHAP/LIME (which explain WHY a prediction was made),
# DiCE answers "WHAT would need to change for the prediction to flip"
# e.g. "if HbA1c dropped from 7.2 to 6.0 and BMI dropped by 2 points,
#       the prediction would change from Pre-Diabetic/Diabetic to Non-Diabetic."
# This gives ACTIONABLE, patient-specific guidance that feeds directly
# into the diet-chart reasoning module.

import pandas as pd
import numpy as np
import joblib
import dice_ml
from dice_ml import Dice

# ── 1. LOAD MODEL, DATA, LABEL ENCODER ─────────
print("=== STEP 1: Loading model and data ===")
stack_model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
class_names = list(target_le.classes_)  # ['Diabetic','Non-Diabetic','Pre-Diabetic']
non_diabetic_idx = int(np.where(target_le.classes_ == "Non-Diabetic")[0][0])

train = pd.read_csv("diabetes_prediction/train_final.csv")
test = pd.read_csv("diabetes_prediction/test_final.csv")

feature_names = [c for c in train.columns if c != "target"]
print(f"Features: {feature_names}")
print(f"Classes : {class_names} (Non-Diabetic index = {non_diabetic_idx})")

# ── 2. BUILD DiCE DATA & MODEL WRAPPERS ────────
print("\n=== STEP 2: Setting up DiCE ===")
dice_data = dice_ml.Data(
    dataframe=train,
    continuous_features=feature_names,   # all 4 features (HbA1c, Glucose, Age, BMI) are continuous
    outcome_name="target",
)

dice_model = dice_ml.Model(model=stack_model, backend="sklearn", model_type="classifier")

# "random" method works generically with any sklearn-style predict_proba model
# (no gradient access needed, unlike a pure "gradient" method).
exp = Dice(dice_data, dice_model, method="random")

# ── 3. PICK A PATIENT WHO IS CURRENTLY AT RISK ─
print("\n=== STEP 3: Selecting an at-risk patient from the test set ===")
X_test = test[feature_names]
y_pred_all = stack_model.predict(X_test)

# find a test patient currently predicted Pre-Diabetic or Diabetic
at_risk_mask = y_pred_all != non_diabetic_idx
at_risk_idx = np.where(at_risk_mask)[0][0]
query_instance = X_test.iloc[[at_risk_idx]]

pred_class = class_names[y_pred_all[at_risk_idx]]
print(f"Selected patient (row {at_risk_idx}):")
print(query_instance)
print(f"Current prediction: {pred_class}")

# ── 4. GENERATE COUNTERFACTUALS -> target Non-Diabetic ─
print(f"\n=== STEP 4: Generating counterfactuals -> target class 'Non-Diabetic' ===")
# NOTE: only clinically/behaviorally modifiable features are allowed to vary.
# "age" cannot be changed by any diet/lifestyle intervention, so it is locked
# (kept fixed at the patient's actual value). Race is not in the final
# feature set (dropped earlier at the <5% importance stage), so no lock
# needed for it here.
modifiable_features = ["hbA1c_level", "blood_glucose_level", "bmi"]

# Keep suggested values within realistic clinical bounds (in the SCALED space
# the model was trained on), so DiCE cannot suggest e.g. BMI=51 as a "fix":
#   HbA1c 3.5-9.0%, Blood Glucose 80-300 mg/dL, BMI 15-45
permitted_range = {
    "hbA1c_level": [-2.09, 2.63],
    "blood_glucose_level": [-1.42, 2.80],
    "bmi": [-1.90, 2.29],
}

cf = exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_class=non_diabetic_idx,
    features_to_vary=modifiable_features,  # "age" excluded -> stays fixed
    permitted_range=permitted_range,
)

cf_df = cf.cf_examples_list[0].final_cfs_df
print("\nCounterfactual examples (values that would flip prediction to Non-Diabetic):")
print(cf_df)

# ── 5. HUMAN-READABLE SUMMARY (real clinical units, for diet-chart module) ──
print("\n=== STEP 5: Human-readable counterfactual summary (real units) ===")
scaler = joblib.load("diabetes_prediction/models/scaler.pkl")
# scaler was fit on ALL original columns (before feature selection); map our
# 4 selected features to their position in that original column order so we
# can manually invert (real = scaled * scale_ + mean_) using the matching
# mean_/scale_ entries only.
original_col_order = ["gender", "age", "race:AfricanAmerican", "race:Asian",
                       "race:Caucasian", "race:Hispanic", "race:Other",
                       "hypertension", "heart_disease", "smoking_history",
                       "bmi", "hbA1c_level", "blood_glucose_level"]
feat_idx = [original_col_order.index(f) for f in feature_names]

def inverse_row(row_vals):
    """row_vals: array-like of scaled feature values in `feature_names` order -> real units"""
    row_vals = np.asarray(row_vals, dtype=float)
    means = scaler.mean_[feat_idx]
    scales = scaler.scale_[feat_idx]
    return row_vals * scales + means

original_real = inverse_row(query_instance.iloc[0][feature_names].values)

for i, row in cf_df.iterrows():
    cf_real = inverse_row(row[feature_names].values)
    changes = []
    for j, feat in enumerate(feature_names):
        orig_val, new_val = original_real[j], cf_real[j]
        if abs(new_val - orig_val) > 0.05:
            direction = "increase" if new_val > orig_val else "decrease"
            changes.append(f"{feat}: {direction} from {orig_val:.2f} to {new_val:.2f}")
    summary = "; ".join(changes) if changes else "no material change needed"
    print(f"Counterfactual {i+1}: {summary}")

# ── 6. SAVE OUTPUT ──────────────────────────────
cf_df.to_csv("diabetes_prediction/dice_counterfactuals.csv", index=False)
print("\nSaved -> diabetes_prediction/dice_counterfactuals.csv")
print("\n=== DiCE Counterfactual Layer Complete! ===")