import joblib
import pandas as pd
import numpy as np
import re
import os
import json
import time
from dotenv import load_dotenv
from google import genai

# ============================================
# SETUP (reusing the same pipeline as diet_constraints.py)
# ============================================
print("=== Loading model, scaler, and FCTB food database ===")
model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
scaler = joblib.load("diabetes_prediction/models/scaler.pkl")
target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
food_df = pd.read_csv("diabetes_prediction/diet_data/fctb_clustered.csv")
class_names = list(target_le.classes_)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def clean_name(name):
    return re.sub(r"\s*\*\s*$", "", str(name)).strip()

fried_keywords = ["fry", "fried"]
def is_fried(name):
    return any(kw in str(name).lower() for kw in fried_keywords)

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
        if len(nf) > 0: carb_cands = nf
    carb_item = None
    if not carb_cands.empty:
        carb_cands = carb_cands.copy()
        carb_cands["diff"] = (carb_cands["Carbohydrate_g"] - carb_target).abs()
        if risk_tier == "High":
            penalty = carb_cands["Diet_Cluster"].apply(lambda c: 15 if c == "Limit" else 0)
            carb_cands["diff"] += penalty
        carb_item = carb_cands.sort_values("diff").iloc[0]
    remaining_cal = calorie_budget - (carb_item["Energy_kcal"] if carb_item is not None else 0)

    protein_cands = get_candidates(meal_type_filter, "Protein", allowed_gi, apply_gi_filter=True)
    animal_groups = ["09 Fish, shellfish and their products",
                      "10 Meat, poultry and their products", "11 Eggs and their products"]
    is_raw_animal = (protein_cands["Food_Group"].isin(animal_groups)) & \
                     (protein_cands["Food_Name_English"].str.contains("raw", case=False, na=False))
    safe_protein = protein_cands[~is_raw_animal]
    if len(safe_protein) > 0: protein_cands = safe_protein
    if risk_tier == "High":
        nf = protein_cands[~protein_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0: protein_cands = nf
    protein_item = None
    if not protein_cands.empty:
        protein_cands = protein_cands.copy()
        target_protein_cal = remaining_cal * 0.6
        protein_cands["diff"] = (protein_cands["Energy_kcal"] - target_protein_cal).abs()
        protein_item = protein_cands.sort_values("diff").iloc[0]
    remaining_cal2 = remaining_cal - (protein_item["Energy_kcal"] if protein_item is not None else 0)

    veg_cands = get_candidates(meal_type_filter, "Vegetable", allowed_gi, apply_gi_filter=True)
    if veg_cands.empty and meal_type_filter == "Breakfast":
        veg_cands = get_candidates("Lunch/Dinner", "Vegetable", allowed_gi, apply_gi_filter=True)
    if risk_tier == "High":
        nf = veg_cands[~veg_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0: veg_cands = nf
    veg_item = None
    if not veg_cands.empty:
        veg_cands = veg_cands.copy()
        veg_cands["diff"] = (veg_cands["Energy_kcal"] - max(remaining_cal2, 0)).abs()
        veg_item = veg_cands.sort_values("diff").iloc[0]

    carb_item = add_portion(carb_item, carb_item["Energy_kcal"] if carb_item is not None else 0)
    protein_item = add_portion(protein_item, remaining_cal * 0.6)
    veg_item = add_portion(veg_item, max(remaining_cal2, 0))

    total_meal_carbs = sum(item["Adj_Carbs"] for item in [carb_item, protein_item, veg_item] if item is not None)
    if carb_item is not None and total_meal_carbs > carb_target * 1.15:
        other_carbs = total_meal_carbs - carb_item["Adj_Carbs"]
        room = max(carb_target - other_carbs, 0)
        base = carb_item["Carbohydrate_g"]
        if base > 0:
            new_mult = max(0.5, min(carb_item["Portion_Multiplier"], round((room / base) * 2) / 2))
            carb_item["Portion_Multiplier"] = new_mult
            carb_item["Adj_Calories"] = carb_item["Energy_kcal"] * new_mult
            carb_item["Adj_Carbs"] = base * new_mult

    return {"Carb": carb_item, "Protein": protein_item, "Vegetable": veg_item}

def compose_snack(calorie_budget, risk_tier, allowed_gi):
    veg_cands = get_candidates("Snack", "Vegetable", allowed_gi, apply_gi_filter=True)
    carb_cands = get_candidates("Snack", "Carb", allowed_gi, apply_gi_filter=False)
    all_cands = pd.concat([veg_cands, carb_cands])
    if risk_tier == "High":
        nf = all_cands[~all_cands["Food_Name_English"].apply(is_fried)]
        if len(nf) > 0: all_cands = nf
    if all_cands.empty: return {"Snack": None}
    all_cands = all_cands.copy()
    all_cands["diff"] = (all_cands["Energy_kcal"] - calorie_budget).abs()
    item = all_cands.sort_values("diff").iloc[0].copy()
    base_cal = item["Energy_kcal"]
    mult = max(0.5, min(2.0, round((calorie_budget / base_cal if base_cal > 0 else 1.0) * 2) / 2))
    item["Portion_Multiplier"] = mult
    item["Adj_Calories"] = base_cal * mult
    item["Adj_Carbs"] = item["Carbohydrate_g"] * mult
    return {"Snack": item}

def format_options(combo):
    lines = []
    for role, item in combo.items():
        if item is not None:
            protein_adj = item["Protein_g"] * item["Portion_Multiplier"]
            fibre_adj = item["Fibre_g"] * item["Portion_Multiplier"]
            lines.append(f"- [{role}] {clean_name(item['Food_Name_English'])} — USE EXACTLY {item['Portion_Multiplier']:.1f}x serving "
                          f"({item['Adj_Calories']:.0f} kcal, {item['Adj_Carbs']:.1f}g carbs, "
                          f"{protein_adj:.1f}g protein, {fibre_adj:.1f}g fibre, GI: {item['GI_Category']})")
    return "\n".join(lines) if lines else "No suitable options found."

RISK_TIER_MAP = {"Non-Diabetic": ("Low", 0.475, "any"), "Pre-Diabetic": ("Medium", 0.375, "low_medium"), "Diabetic": ("High", 0.28, "low_only")}

# ============================================
# STEP 1: GENERATE SAMPLE DIET CHARTS TO EVALUATE
# ============================================
test_patients = [
    {"label": "Patient A", "age": 28, "gender": "Female", "weight_kg": 55, "height_cm": 160,
     "activity_level": "Moderate", "hba1c_level": 5.2, "blood_glucose_level": 90, "bmi": 21.5},
    {"label": "Patient B", "age": 50, "gender": "Male", "weight_kg": 82, "height_cm": 168,
     "activity_level": "Sedentary", "hba1c_level": 6.0, "blood_glucose_level": 130, "bmi": 26.5},
    {"label": "Patient C", "age": 45, "gender": "Male", "weight_kg": 78, "height_cm": 170,
     "activity_level": "Moderate", "hba1c_level": 7.2, "blood_glucose_level": 165, "bmi": 27.0},
]

evaluation_records = []

print("\n=== STEP 1: Generating diet charts for evaluation ===")
for patient in test_patients:
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    full_input = pd.DataFrame([{
        "gender": gender_map.get(patient["gender"], 0), "age": patient["age"],
        "race:AfricanAmerican": 0, "race:Asian": 0, "race:Caucasian": 0, "race:Hispanic": 0, "race:Other": 0,
        "hypertension": 0, "heart_disease": 0, "smoking_history": 0,
        "bmi": patient["bmi"], "hbA1c_level": patient["hba1c_level"], "blood_glucose_level": patient["blood_glucose_level"]
    }])
    scaled_df = pd.DataFrame(scaler.transform(full_input), columns=full_input.columns)
    patient_df = scaled_df[["hbA1c_level", "blood_glucose_level", "age", "bmi"]]
    pred_idx = model.predict(patient_df)[0]
    prediction = class_names[pred_idx]

    risk_tier, carb_percent, gi_restriction = RISK_TIER_MAP[prediction]
    allowed_gi = {"low_only": ["Low"], "low_medium": ["Low", "Medium"], "any": ["Low", "Medium", "High"]}[gi_restriction]

    if patient["gender"] == "Male":
        bmr = (10*patient["weight_kg"]) + (6.25*patient["height_cm"]) - (5*patient["age"]) + 5
    else:
        bmr = (10*patient["weight_kg"]) + (6.25*patient["height_cm"]) - (5*patient["age"]) - 161
    activity_multipliers = {"Sedentary": 1.2, "Moderate": 1.55, "Active": 1.725}
    tdee = bmr * activity_multipliers[patient["activity_level"]]
    daily_carb_grams = (tdee * carb_percent) / 4
    meal_split = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
    meal_budgets = {m: tdee*pct for m, pct in meal_split.items()}
    meal_carb_targets = {m: daily_carb_grams*pct for m, pct in meal_split.items()}

    diet_plan = {}
    diet_plan["Breakfast"] = compose_meal_combo("Breakfast", meal_budgets["Breakfast"], meal_carb_targets["Breakfast"], risk_tier, allowed_gi)
    diet_plan["Lunch"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Lunch"], meal_carb_targets["Lunch"], risk_tier, allowed_gi)
    diet_plan["Dinner"] = compose_meal_combo("Lunch/Dinner", meal_budgets["Dinner"], meal_carb_targets["Dinner"], risk_tier, allowed_gi)
    diet_plan["Snack"] = compose_snack(meal_budgets["Snack"], risk_tier, allowed_gi)

    context_text = ""
    for meal_name in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        context_text += f"\n{meal_name} (~{meal_budgets[meal_name]:.0f} kcal):\n{format_options(diet_plan[meal_name])}\n"

    gen_prompt = f"""You are a clinical nutrition assistant creating a diet chart for a patient in
Bangladesh. Use ONLY the food options provided below.

AVAILABLE FOOD OPTIONS:
{context_text}

TASK: Write a friendly daily diet chart. For each meal, recommend the food combination
from the options given, using EXACTLY the portion sizes specified. Briefly explain why
each choice is good. Keep it concise (under 200 words)."""

    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=gen_prompt)
    generated_chart = response.text

    evaluation_records.append({
        "patient_label": patient["label"],
        "context": context_text,
        "generated_chart": generated_chart,
    })
    print(f"  Generated chart for {patient['label']}")
    time.sleep(2)

# ============================================
# STEP 2: LLM-AS-JUDGE — CLAIM EXTRACTION + FAITHFULNESS SCORING
# (RAGAS-inspired methodology)
# ============================================
print("\n=== STEP 2: Running LLM-as-Judge faithfulness evaluation ===")

all_results = []

for record in evaluation_records:
    judge_prompt = f"""You are an impartial fact-checking judge. You will be given a CONTEXT
(a list of approved food items with exact nutritional data) and a GENERATED DIET CHART
that was supposed to be written using ONLY that context.

Your task (following the RAGAS faithfulness methodology):
1. Break the GENERATED DIET CHART down into a list of atomic factual claims (short,
   standalone statements — e.g., "Breakfast includes 1.0x serving of Bread", "The lunch
   meal provides approximately 680 kcal", "Bitter gourd is recommended for its low
   glycemic impact"). Ignore purely stylistic/greeting sentences.
2. For EACH claim, judge whether it is "Supported" (can be directly verified from the
   CONTEXT) or "Not Supported" (cannot be verified from the CONTEXT, or contradicts it —
   this includes invented food items, invented numbers, or invented health claims not
   grounded in the given data).

CONTEXT:
{record['context']}

GENERATED DIET CHART:
{record['generated_chart']}

Respond with ONLY a valid JSON array, no extra text, in this exact format:
[{{"claim": "...", "verdict": "Supported"}}, {{"claim": "...", "verdict": "Not Supported"}}, ...]
"""

    try:
        judge_response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=judge_prompt)
        text = judge_response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        claims = json.loads(text.strip())

        supported = sum(1 for c in claims if c.get("verdict", "").strip().lower() == "supported")
        total = len(claims)
        faithfulness = supported / total if total > 0 else None

        all_results.append({
            "patient_label": record["patient_label"],
            "total_claims": total,
            "supported_claims": supported,
            "faithfulness_score": faithfulness,
            "claims_detail": claims,
        })
        print(f"\n{record['patient_label']}: {supported}/{total} claims supported "
              f"(Faithfulness = {faithfulness:.2%})" if faithfulness is not None else "  Could not compute")

        unsupported = [c for c in claims if c.get("verdict", "").strip().lower() != "supported"]
        if unsupported:
            print("  Unsupported claims:")
            for c in unsupported:
                print(f"    - {c['claim']}")

    except Exception as e:
        print(f"  Error judging {record['patient_label']}: {e}")

    time.sleep(2)

# ============================================
# STEP 3: AGGREGATE FAITHFULNESS SCORE
# ============================================
valid_scores = [r["faithfulness_score"] for r in all_results if r["faithfulness_score"] is not None]
if valid_scores:
    overall = sum(valid_scores) / len(valid_scores)
    print(f"\n=== OVERALL LLM-as-Judge Faithfulness Score: {overall:.2%} (averaged across {len(valid_scores)} diet charts) ===")

# Save detailed results
results_df = pd.DataFrame([{
    "patient_label": r["patient_label"],
    "total_claims": r["total_claims"],
    "supported_claims": r["supported_claims"],
    "faithfulness_score": r["faithfulness_score"],
} for r in all_results])
results_df.to_csv("diabetes_prediction/llm_judge_faithfulness_results.csv", index=False)
print("\nSaved -> diabetes_prediction/llm_judge_faithfulness_results.csv")