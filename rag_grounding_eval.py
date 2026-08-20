# ============================================
# RAG GROUNDING / HALLUCINATION EVALUATION
# ============================================
# Inspired by Vavken et al. (2025, IEEE BigData) -- "Evaluation of LLMs in
# Retrieving Food and Nutritional Context for RAG Systems". We quantitatively
# measure whether the Gemini-generated diet chart stays GROUNDED to the exact
# candidate food items and portion sizes we provided (our RAG design promises
# "no hallucination" since Gemini only reformats pre-selected foods -- this
# script checks that promise numerically instead of just assuming it).
#
# Two metrics are computed per patient:
#   1. Food Grounding Rate  = % of candidate food items that appear in the
#      generated chart text (did Gemini use what it was given, and not
#      invent or drop items?)
#   2. Portion Fidelity Rate = % of those matched items where the EXACT
#      portion multiplier we specified (e.g. "1.5x") also appears verbatim
#      near the food name (did Gemini respect the exact serving size?)

import joblib
import pandas as pd
import numpy as np
import os
import re
import time
from dotenv import load_dotenv
from google import genai

print("=== STEP 1: Loading model, scaler, and FCTB food database ===")
model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
scaler = joblib.load("diabetes_prediction/models/scaler.pkl")
target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
food_df = pd.read_csv("diabetes_prediction/diet_data/fctb_clustered.csv")
class_names = list(target_le.classes_)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

fried_keywords = ["fry", "fried"]

def is_fried(name):
    return any(kw in str(name).lower() for kw in fried_keywords)

def clean_name(name):
    return re.sub(r"\s*\*\s*$", "", str(name)).strip()

def clean_for_match(name):
    # first significant phrase before a comma, lowercase -- forgiving match
    return clean_name(name).split(",")[0].strip().lower()

# ============================================
# (same food-selection functions as diet_constraints.py)
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
    if risk_tier == "High":
        nf = veg_cands[~veg_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0:
            veg_cands = nf
    veg_item = None
    if not veg_cands.empty:
        veg_cands = veg_cands.copy()
        veg_cands["diff"] = (veg_cands["Energy_kcal"] - max(remaining_cal2, 0)).abs()
        veg_item = veg_cands.sort_values("diff").iloc[0]

    carb_item = add_portion(carb_item, carb_item["Energy_kcal"] if carb_item is not None else 0)
    protein_item = add_portion(protein_item, remaining_cal * 0.6)
    veg_item = add_portion(veg_item, max(remaining_cal2, 0))
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
    item = add_portion(item, calorie_budget)
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

RISK_TIER_MAP = {
    "Non-Diabetic": ("Low", 0.475, "any"),
    "Pre-Diabetic": ("Medium", 0.375, "low_medium"),
    "Diabetic":     ("High", 0.28, "low_only"),
}

patients = [
    {"label": "Patient A", "age": 28, "gender": "Female", "weight_kg": 55, "height_cm": 160,
     "activity_level": "Moderate", "hba1c_level": 5.2, "blood_glucose_level": 90, "bmi": 21.5},
    {"label": "Patient B", "age": 50, "gender": "Male", "weight_kg": 82, "height_cm": 168,
     "activity_level": "Sedentary", "hba1c_level": 6.0, "blood_glucose_level": 130, "bmi": 26.5},
    {"label": "Patient C", "age": 45, "gender": "Male", "weight_kg": 78, "height_cm": 170,
     "activity_level": "Moderate", "hba1c_level": 7.2, "blood_glucose_level": 165, "bmi": 27.0},
]

# ============================================
# MAIN EVALUATION LOOP
# ============================================
all_results = []

for patient in patients:
    print(f"\n{'='*60}\n{patient['label']}\n{'='*60}")

    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    full_input = pd.DataFrame([{
        "gender": gender_map.get(patient["gender"], 0), "age": patient["age"],
        "race:AfricanAmerican": 0, "race:Asian": 0, "race:Caucasian": 0,
        "race:Hispanic": 0, "race:Other": 0, "hypertension": 0,
        "heart_disease": 0, "smoking_history": 0, "bmi": patient["bmi"],
        "hbA1c_level": patient["hba1c_level"], "blood_glucose_level": patient["blood_glucose_level"]
    }])
    scaled_df = pd.DataFrame(scaler.transform(full_input), columns=full_input.columns)
    patient_df = scaled_df[["hbA1c_level", "blood_glucose_level", "age", "bmi"]]

    pred_idx = model.predict(patient_df)[0]
    prediction = class_names[pred_idx]
    risk_tier, carb_percent, gi_restriction = RISK_TIER_MAP[prediction]
    allowed_gi = {"low_only": ["Low"], "low_medium": ["Low", "Medium"],
                  "any": ["Low", "Medium", "High"]}[gi_restriction]
    print(f"Prediction: {prediction} | Risk Tier: {risk_tier}")

    if patient["gender"] == "Male":
        bmr = (10 * patient["weight_kg"]) + (6.25 * patient["height_cm"]) - (5 * patient["age"]) + 5
    else:
        bmr = (10 * patient["weight_kg"]) + (6.25 * patient["height_cm"]) - (5 * patient["age"]) - 161
    activity_multipliers = {"Sedentary": 1.2, "Moderate": 1.55, "Active": 1.725}
    tdee = bmr * activity_multipliers[patient["activity_level"]]
    daily_carb_grams = (tdee * carb_percent) / 4
    meal_split = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
    meal_budgets = {m: tdee * pct for m, pct in meal_split.items()}
    meal_carb_targets = {m: daily_carb_grams * pct for m, pct in meal_split.items()}

    diet_plan = {}
    diet_plan["Breakfast"] = compose_meal_combo("Breakfast", meal_budgets["Breakfast"], meal_carb_targets["Breakfast"], risk_tier, allowed_gi)
    diet_plan["Lunch"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Lunch"], meal_carb_targets["Lunch"], risk_tier, allowed_gi)
    diet_plan["Dinner"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Dinner"], meal_carb_targets["Dinner"], risk_tier, allowed_gi)
    diet_plan["Snack"] = compose_snack(meal_budgets["Snack"], risk_tier, allowed_gi)

    prompt = f"""You are a clinical nutrition assistant creating a personalized diabetic diet chart
for a patient in Bangladesh. Use ONLY the food options provided below.

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

TASK: Write a friendly daily diet chart. For each meal, recommend the food combination
from the options given, using EXACTLY the portion sizes specified (e.g., "1.5x serving" means
one and a half servings — do NOT invent your own serving counts or quantities). Briefly explain
why each choice is good. End with a short, warm 2-3 sentence summary of their risk level in
plain language (no jargon).
"""

    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    chart_text = response.text

    # ── GROUNDING CHECK ──────────────────────────
    all_candidate_items = []
    for combo in diet_plan.values():
        for item in combo.values():
            if item is not None:
                all_candidate_items.append(item)

    matched = 0
    portion_matched = 0
    for item in all_candidate_items:
        key = clean_for_match(item["Food_Name_English"])
        expected_portion = f"{item['Portion_Multiplier']:.1f}x"
        if key in chart_text.lower():
            matched += 1
            # check the exact portion string appears somewhere near/within the text
            if expected_portion in chart_text.lower():
                portion_matched += 1
        else:
            print(f"  [NOT GROUNDED] '{item['Food_Name_English']}' not found in output text")

    total = len(all_candidate_items)
    grounding_rate = matched / total if total else 0
    portion_fidelity = portion_matched / matched if matched else 0

    print(f"Food Grounding Rate: {matched}/{total} = {grounding_rate:.1%}")
    print(f"Portion Fidelity Rate (of grounded items): {portion_matched}/{matched} = {portion_fidelity:.1%}")

    all_results.append({
        "Patient": patient["label"], "Prediction": prediction,
        "Total_Candidates": total, "Grounded": matched,
        "Grounding_Rate": grounding_rate, "Portion_Fidelity_Rate": portion_fidelity
    })

    time.sleep(4)

# ============================================
# FINAL SUMMARY
# ============================================
results_df = pd.DataFrame(all_results)
print("\n" + "=" * 60)
print("=== RAG GROUNDING EVALUATION SUMMARY ===")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nOverall Average Grounding Rate: {results_df['Grounding_Rate'].mean():.1%}")
print(f"Overall Average Portion Fidelity Rate: {results_df['Portion_Fidelity_Rate'].mean():.1%}")

results_df.to_csv("diabetes_prediction/rag_grounding_eval_results.csv", index=False)
print("\nSaved -> diabetes_prediction/rag_grounding_eval_results.csv")