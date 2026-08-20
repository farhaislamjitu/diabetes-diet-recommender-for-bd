import joblib
import pandas as pd
import numpy as np
import shap
import os
import time
from dotenv import load_dotenv
from google import genai
import re

def clean_name(name):
    return re.sub(r"\s*\*\s*$", "", str(name)).strip()  # FCTB uses trailing "*" for cooked/composite dishes

# ============================================
# SETUP: LOAD MODEL, SCALER, ENCODER, FCTB FOOD DATABASE (ONCE)
# ============================================
print("=== Loading model, scaler, and FCTB food database ===")
model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
scaler = joblib.load("diabetes_prediction/models/scaler.pkl")
target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
explainer = joblib.load("diabetes_prediction/models/shap_explainer.pkl")
food_df = pd.read_csv("diabetes_prediction/diet_data/fctb_clustered.csv")

class_names = list(target_le.classes_)  # ['Diabetic', 'Non-Diabetic', 'Pre-Diabetic']

# ── DiCE COUNTERFACTUAL SETUP (Causal-DiCE Integration) ──
print("=== Setting up DiCE counterfactual explainer ===")
import dice_ml
from dice_ml import Dice

_train_for_dice = pd.read_csv("diabetes_prediction/train_final.csv")
_feature_names = [c for c in _train_for_dice.columns if c != "target"]
_dice_data = dice_ml.Data(dataframe=_train_for_dice, continuous_features=_feature_names,
                           outcome_name="target")
_dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
dice_exp = Dice(_dice_data, _dice_model, method="random")

_non_diabetic_idx = int(np.where(target_le.classes_ == "Non-Diabetic")[0][0])
_modifiable_features = ["hbA1c_level", "blood_glucose_level", "bmi"]
# realistic clinical bounds in the SCALED space (same as dice_counterfactual.py)
_permitted_range = {
    "hbA1c_level": [-2.09, 2.63],
    "blood_glucose_level": [-1.42, 2.80],
    "bmi": [-1.90, 2.29],
}
_original_col_order = ["gender", "age", "race:AfricanAmerican", "race:Asian",
                        "race:Caucasian", "race:Hispanic", "race:Other",
                        "hypertension", "heart_disease", "smoking_history",
                        "bmi", "hbA1c_level", "blood_glucose_level"]
_feat_idx = [_original_col_order.index(f) for f in _feature_names]

def get_counterfactual_summary(patient_df_row):
    """Run DiCE targeting Non-Diabetic and return a short natural-language
    summary of the smallest realistic change. Returns None on failure."""
    try:
        cf = dice_exp.generate_counterfactuals(
            patient_df_row, total_CFs=1, desired_class=_non_diabetic_idx,
            features_to_vary=_modifiable_features, permitted_range=_permitted_range,
        )
        cf_row = cf.cf_examples_list[0].final_cfs_df.iloc[0]
        means = scaler.mean_[_feat_idx]
        scales = scaler.scale_[_feat_idx]
        orig_real = patient_df_row.iloc[0][_feature_names].values * scales + means
        cf_real = cf_row[_feature_names].values * scales + means

        changes = []
        for j, feat in enumerate(_feature_names):
            delta = cf_real[j] - orig_real[j]
            # Only keep changes in the CLINICALLY SENSIBLE direction: HbA1c,
            # Glucose, and BMI should only ever be suggested to DECREASE for
            # a diabetes-risk context. A DiCE-suggested increase (a quirk of
            # its boundary search, not real medical advice) is dropped.
            if feat in ("hbA1c_level", "blood_glucose_level", "bmi") and delta >= -0.3:
                continue
            if abs(delta) > 0.3:
                changes.append(f"{feat} from {orig_real[j]:.1f} to {cf_real[j]:.1f}")
        if not changes:
            return None
        return "; ".join(changes)
    except Exception as e:
        print(f"  [DiCE skipped: {e}]")
        return None


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

fried_keywords = ["fry", "fried"]

def is_fried(name):
    name = str(name).lower()
    return any(kw in name for kw in fried_keywords)

# ============================================
# SAMPLE PATIENTS (3 different risk profiles)
# ============================================
patients = [
    {
        "label": "Patient A (expected: Low risk / Non-diabetic)",
        "age": 28, "gender": "Female", "weight_kg": 55, "height_cm": 160,
        "activity_level": "Moderate",
        "hba1c_level": 5.2, "blood_glucose_level": 90, "bmi": 21.5
    },
    {
        "label": "Patient B (expected: Medium risk / Pre-diabetic)",
        "age": 50, "gender": "Male", "weight_kg": 82, "height_cm": 168,
        "activity_level": "Sedentary",
        "hba1c_level": 6.0, "blood_glucose_level": 130, "bmi": 26.5
    },
    {
        "label": "Patient C (expected: High risk / Diabetic)",
        "age": 45, "gender": "Male", "weight_kg": 78, "height_cm": 170,
        "activity_level": "Moderate",
        "hba1c_level": 7.2, "blood_glucose_level": 165, "bmi": 27.0
    }
]

# ============================================
# FOOD SELECTION FUNCTIONS (FCTB column names)
# ============================================
def get_candidates(meal_type_filter, role, allowed_gi, apply_gi_filter):
    cands = food_df[
        (food_df["Meal_Type"] == meal_type_filter) &
        (food_df["Food_Role"] == role) &
        (food_df["Is_Standalone"] == "Yes")
    ].copy()
    if apply_gi_filter:
        cands = cands[cands["GI_Category"].isin(allowed_gi)]
    return cands

def compose_meal_combo(meal_type_filter, calorie_budget, carb_target, risk_tier, allowed_gi):
    carb_cands = get_candidates(meal_type_filter, "Carb", allowed_gi, apply_gi_filter=False)
    if risk_tier == "High":
        nf = carb_cands[~carb_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0:
            carb_cands = nf
    carb_item = None
    if not carb_cands.empty:
        carb_cands = carb_cands.copy()
        carb_cands["diff"] = (carb_cands["Carbohydrate_g"] - carb_target).abs()
        if risk_tier == "High":
            penalty = carb_cands["Diet_Cluster"].apply(lambda c: 15 if c == "Limit" else 0)
            carb_cands["diff"] = carb_cands["diff"] + penalty
        carb_item = carb_cands.sort_values("diff").iloc[0]

    remaining_cal = calorie_budget - (carb_item["Energy_kcal"] if carb_item is not None else 0)

    protein_cands = get_candidates(meal_type_filter, "Protein", allowed_gi, apply_gi_filter=True)
    if risk_tier == "High":
        nf = protein_cands[~protein_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0:
            protein_cands = nf
    protein_item = None
    if not protein_cands.empty:
        protein_cands = protein_cands.copy()
        target_protein_cal = remaining_cal * 0.6
        protein_cands["diff"] = (protein_cands["Energy_kcal"] - target_protein_cal).abs()
        protein_item = protein_cands.sort_values("diff").iloc[0]

    remaining_cal2 = remaining_cal - (protein_item["Energy_kcal"] if protein_item is not None else 0)

    veg_cands = get_candidates(meal_type_filter, "Vegetable", allowed_gi, apply_gi_filter=True)
    if veg_cands.empty and meal_type_filter == "Breakfast":
        # Not enough dedicated breakfast vegetables — fall back to the
        # general Lunch/Dinner vegetable pool
        veg_cands = get_candidates("Lunch/Dinner", "Vegetable", allowed_gi, apply_gi_filter=True)
    if risk_tier == "High":
        nf = veg_cands[~veg_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0:
            veg_cands = nf
    veg_item = None
    if not veg_cands.empty:
        veg_cands = veg_cands.copy()
        veg_cands["diff"] = (veg_cands["Energy_kcal"] - max(remaining_cal2, 0)).abs()
        veg_item = veg_cands.sort_values("diff").iloc[0]

    def add_portion(item, target_cal):
        if item is None:
            return None
        item = item.copy()
        base_cal = item["Energy_kcal"]
        multiplier = target_cal / base_cal if base_cal > 0 else 1.0
        multiplier = max(0.5, min(2.0, round(multiplier * 2) / 2))
        item["Portion_Multiplier"] = multiplier
        item["Adj_Calories"] = base_cal * multiplier
        item["Adj_Carbs"] = item["Carbohydrate_g"] * multiplier
        return item

    carb_item = add_portion(carb_item, carb_item["Energy_kcal"] if carb_item is not None else 0)
    protein_item = add_portion(protein_item, remaining_cal * 0.6)
    veg_item = add_portion(veg_item, max(remaining_cal2, 0))

    # CARB-BUDGET CORRECTION: Protein/Vegetable items also carry some carbs
    # (e.g. peas, lentils) that weren't counted when picking the Carb item.
    # If the combined meal overshoots the carb target by >15%, shrink the
    # Carb item's portion (down to a 0.5x floor) to compensate.
    total_meal_carbs = sum(
        item["Adj_Carbs"] for item in [carb_item, protein_item, veg_item] if item is not None
    )
    if carb_item is not None and total_meal_carbs > carb_target * 1.15:
        other_carbs = total_meal_carbs - carb_item["Adj_Carbs"]
        room_for_carb_item = max(carb_target - other_carbs, 0)
        base_carb_per_unit = carb_item["Carbohydrate_g"]
        if base_carb_per_unit > 0:
            new_multiplier = room_for_carb_item / base_carb_per_unit
            new_multiplier = max(0.5, min(carb_item["Portion_Multiplier"], round(new_multiplier * 2) / 2))
            carb_item["Portion_Multiplier"] = new_multiplier
            carb_item["Adj_Calories"] = carb_item["Energy_kcal"] * new_multiplier
            carb_item["Adj_Carbs"] = base_carb_per_unit * new_multiplier

    return {"Carb": carb_item, "Protein": protein_item, "Vegetable": veg_item}

def compose_snack(calorie_budget, risk_tier, allowed_gi):
    veg_cands = get_candidates("Snack", "Vegetable", allowed_gi, apply_gi_filter=True)
    carb_cands = get_candidates("Snack", "Carb", allowed_gi, apply_gi_filter=False)
    all_cands = pd.concat([veg_cands, carb_cands])
    if risk_tier == "High":
        nf = all_cands[~all_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0:
            all_cands = nf
    if all_cands.empty:
        return {"Snack": None}
    all_cands = all_cands.copy()
    all_cands["diff"] = (all_cands["Energy_kcal"] - calorie_budget).abs()
    item = all_cands.sort_values("diff").iloc[0].copy()
    base_cal = item["Energy_kcal"]
    multiplier = calorie_budget / base_cal if base_cal > 0 else 1.0
    multiplier = max(0.5, min(2.0, round(multiplier * 2) / 2))
    item["Portion_Multiplier"] = multiplier
    item["Adj_Calories"] = base_cal * multiplier
    item["Adj_Carbs"] = item["Carbohydrate_g"] * multiplier
    return {"Snack": item}

def format_options(combo):
    lines = []
    for role, item in combo.items():
        if item is not None:
            portion_text = f"{item['Portion_Multiplier']:.1f}x serving"
            lines.append(f"- [{role}] {clean_name(item['Food_Name_English'])} — USE EXACTLY {portion_text} "
                          f"({item['Adj_Calories']:.0f} kcal, {item['Adj_Carbs']:.1f}g carbs, "
                          f"GI: {item['GI_Category']}, Diet Group: {item['Diet_Cluster']})")
    return "\n".join(lines) if lines else "No suitable options found."

# ============================================
# MAIN: PROCESS EACH PATIENT
# ============================================
for patient in patients:
    print("\n" + "="*60)
    print(patient["label"])
    print("="*60)

    gender_map = {"Female": 0, "Male": 1, "Other": 2}

    # Build full 13-column input matching the scaler's training structure
    full_input = pd.DataFrame([{
        "gender": gender_map.get(patient["gender"], 0),
        "age": patient["age"],
        "race:AfricanAmerican": 0,
        "race:Asian": 0,
        "race:Caucasian": 0,
        "race:Hispanic": 0,
        "race:Other": 0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": 0,
        "bmi": patient["bmi"],
        "hbA1c_level": patient["hba1c_level"],
        "blood_glucose_level": patient["blood_glucose_level"]
    }])

    scaled_array = scaler.transform(full_input)
    scaled_df = pd.DataFrame(scaled_array, columns=full_input.columns)
    patient_df = scaled_df[["hbA1c_level", "blood_glucose_level", "age", "bmi"]]

    pred_idx = model.predict(patient_df)[0]
    pred_proba = model.predict_proba(patient_df)[0]
    prediction = class_names[pred_idx]
    confidence = pred_proba[pred_idx]

    print(f"Prediction: {prediction} | Confidence: {confidence:.2%}")
    print(f"Full probabilities: {dict(zip(class_names, np.round(pred_proba, 4)))}")

    # ============================================
    # SOFT-PROBABILISTIC TIERING (Borderline detection)
    # ============================================
    # A patient near the decision boundary (low confidence in the predicted
    # class) is flagged as "Borderline" even though the hard prediction is
    # a single class. Non-Diabetic predictions get a stricter 85% bar,
    # since under-flagging a borderline Non-Diabetic case is the costliest
    # mistake (a person who should get preventive advice, doesn't).
    if prediction == "Non-Diabetic":
        is_borderline = confidence < 0.85
    else:
        is_borderline = confidence < 0.65
    if is_borderline:
        print(f"  >> BORDERLINE case flagged (confidence {confidence:.1%} near decision boundary)")

    # ============================================
    # CHECK IF NON-DIABETIC: ASK IF PREVENTIVE CHART WANTED
    # ============================================
    if prediction == "Non-Diabetic" and not is_borderline:
        print(f"\n{patient['label']}: Prediction is Non-Diabetic (Confidence: {confidence:.2%})")
        wants_chart = input("This patient is NOT diabetic. Generate a preventive diet chart anyway? (yes/no): ").strip().lower()
        if wants_chart != "yes":
            print(f"\n--- HEALTH MESSAGE ---")
            print(f"Great news! Your test results show a low diabetes risk. "
                  f"Your current health markers (HbA1c: {patient['hba1c_level']}%, "
                  f"Glucose: {patient['blood_glucose_level']} mg/dL) are within a healthy range. "
                  f"Keep maintaining a balanced diet and active lifestyle to stay this way!")
            print("(Skipping detailed diet chart generation)")
            continue
    elif prediction == "Non-Diabetic" and is_borderline:
        print(f"\n{patient['label']}: Non-Diabetic but BORDERLINE (confidence {confidence:.1%} < 85%) "
              f"-> generating a preventive chart automatically (no prompt needed).")

    # ============================================
    # RISK TIER — now mapped DIRECTLY from the ternary prediction
    # ============================================
    RISK_TIER_MAP = {
        "Non-Diabetic": ("Low", 0.475, "any"),
        "Pre-Diabetic": ("Medium", 0.375, "low_medium"),
        "Diabetic":     ("High", 0.28, "low_only"),
    }
    risk_tier, carb_percent, gi_restriction = RISK_TIER_MAP[prediction]

    if gi_restriction == "low_only":
        allowed_gi = ["Low"]
    elif gi_restriction == "low_medium":
        allowed_gi = ["Low", "Medium"]
    else:
        allowed_gi = ["Low", "Medium", "High"]

    print(f"Risk Tier: {risk_tier} | Carb%: {carb_percent:.0%} | GI: {gi_restriction}")

    if patient["gender"] == "Male":
        bmr = (10 * patient["weight_kg"]) + (6.25 * patient["height_cm"]) - (5 * patient["age"]) + 5
    else:
        bmr = (10 * patient["weight_kg"]) + (6.25 * patient["height_cm"]) - (5 * patient["age"]) - 161

    activity_multipliers = {"Sedentary": 1.2, "Moderate": 1.55, "Active": 1.725}
    tdee = bmr * activity_multipliers[patient["activity_level"]]
    daily_carb_grams = (tdee * carb_percent) / 4

    print(f"TDEE: {tdee:.0f} kcal | Daily Carb Target: {daily_carb_grams:.0f} g")

    meal_split = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
    meal_budgets = {m: tdee * pct for m, pct in meal_split.items()}
    meal_carb_targets = {m: daily_carb_grams * pct for m, pct in meal_split.items()}

    diet_plan = {}
    diet_plan["Breakfast"] = compose_meal_combo("Breakfast", meal_budgets["Breakfast"], meal_carb_targets["Breakfast"], risk_tier, allowed_gi)
    diet_plan["Lunch"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Lunch"], meal_carb_targets["Lunch"], risk_tier, allowed_gi)
    diet_plan["Dinner"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Dinner"], meal_carb_targets["Dinner"], risk_tier, allowed_gi)
    diet_plan["Snack"] = compose_snack(meal_budgets["Snack"], risk_tier, allowed_gi)

    total_cal = sum(item["Adj_Calories"] for combo in diet_plan.values() for item in combo.values() if item is not None)
    total_carb = sum(item["Adj_Carbs"] for combo in diet_plan.values() for item in combo.values() if item is not None)
    print(f"Daily Total: {total_cal:.0f} kcal (target {tdee:.0f}) | {total_carb:.1f}g carbs (target {daily_carb_grams:.0f})")

    # ============================================
    # SHAP EXPLANATION (best base-learner, as in shap_explain.py)
    # ============================================
    patient_shap = explainer.shap_values(patient_df)
    if isinstance(patient_shap, list):
        # older-style: list of arrays, one per class
        class_shap = patient_shap[pred_idx][0]
    elif patient_shap.ndim == 3:
        # newer-style: single array shaped (n_samples, n_features, n_classes)
        class_shap = patient_shap[0, :, pred_idx]
    else:
        class_shap = patient_shap[0]

    shap_lines = []
    for feature, value in zip(patient_df.columns, class_shap):
        direction = "increases" if value > 0 else "decreases"
        shap_lines.append(f"{feature} {direction} likelihood of '{prediction}' (impact: {abs(value):.2f})")
    shap_text = "\n".join(shap_lines)

    # ============================================
    # CAUSAL-DiCE: counterfactual "what would help" guidance
    # ============================================
    counterfactual_text = "N/A"
    if prediction != "Non-Diabetic":
        print("  Generating DiCE counterfactual guidance...")
        cf_summary = get_counterfactual_summary(patient_df)
        if cf_summary:
            counterfactual_text = cf_summary
            print(f"  DiCE suggestion: {cf_summary}")

    borderline_note = (
        f"\nNOTE: This patient's result is BORDERLINE (confidence {confidence:.0%}, close to "
        f"the next risk category). Mention this gently and recommend closer monitoring / earlier "
        f"re-testing." if is_borderline else ""
    )

    prompt = f"""You are a clinical nutrition assistant creating a personalized diabetic diet chart
for a patient in Bangladesh. Use ONLY the food options provided below.

PATIENT PROFILE:
- Age: {patient['age']}, Gender: {patient['gender']}
- Weight: {patient['weight_kg']} kg, Height: {patient['height_cm']} cm
- Activity Level: {patient['activity_level']}
- HbA1c: {patient['hba1c_level']}%, Blood Glucose: {patient['blood_glucose_level']} mg/dL, BMI: {patient['bmi']}

PREDICTION RESULT:
- Prediction: {prediction} (Confidence: {confidence:.1%})
- Risk Tier: {risk_tier}{borderline_note}

WHY THIS PREDICTION:
{shap_text}

WHAT WOULD IMPROVE THIS PATIENT'S RISK TIER (counterfactual guidance, mention briefly and
encouragingly if not "N/A"):
{counterfactual_text}

DAILY TARGETS: {tdee:.0f} kcal/day, {daily_carb_grams:.0f}g carbs/day

AVAILABLE FOOD OPTIONS:

Breakfast (~{meal_budgets['Breakfast']:.0f} kcal):
{format_options(diet_plan['Breakfast'])}

Lunch (~{meal_budgets['Lunch']:.0f} kcal):
{format_options(diet_plan['Lunch'])}

Dinner (~{meal_budgets['Dinner']:.0f} kcal):
{format_options(diet_plan['Dinner'])}

Snack (~{meal_budgets['Snack']:.0f} kcal):
{format_options(diet_plan['Snack'])}

IMPORTANT: All meat, fish, and egg items listed above must be understood as
COOKED (boiled, grilled, or lightly fried as appropriate) before eating — never
suggest eating them raw, even if the underlying nutrient data source labels them
"raw" (that label refers to the pre-cooking nutrient basis only).

TASK: Write a friendly daily diet chart. For each meal, recommend the food combination
from the options given, using EXACTLY the portion sizes specified (e.g., "1.5x serving" means
one and a half servings — do NOT invent your own serving counts or quantities). Briefly explain
why each choice is good. End with a short, warm 2-3 sentence summary of their risk level in
plain language (no jargon).
"""

    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    print("\n--- DIET CHART ---")
    print(response.text)

    time.sleep(4)

print("\n=== All patients processed! ===")