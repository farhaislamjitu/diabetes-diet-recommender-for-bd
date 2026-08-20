import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
from dotenv import load_dotenv
from google import genai
from rdflib import Graph

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Diabetes Diet Recommender", page_icon="🍽️", layout="centered")

# ============================================
# LOAD RESOURCES (cached so it only loads once)
# ============================================
@st.cache_resource
def load_resources():
    model = joblib.load("diabetes_prediction/models/stacking_model.pkl")
    scaler = joblib.load("diabetes_prediction/models/scaler.pkl")
    target_le = joblib.load("diabetes_prediction/models/target_label_encoder.pkl")
    explainer = joblib.load("diabetes_prediction/models/shap_explainer.pkl")
    food_df = pd.read_csv("diabetes_prediction/diet_data/fctb_clustered.csv")
    return model, scaler, target_le, explainer, food_df

@st.cache_resource
def load_knowledge_graph():
    """Load the RDF food knowledge graph. Returns None if unavailable,
    so the app can fall back to pandas-based filtering."""
    try:
        g = Graph()
        g.parse("diabetes_prediction/diet_data/food_knowledge_graph.ttl", format="turtle")
        return g
    except Exception as e:
        st.warning(f"Knowledge graph unavailable, using rule-based fallback: {e}")
        return None

model, scaler, target_le, explainer, food_df = load_resources()
kg = load_knowledge_graph()

KG_MEAL_MAP = {"Lunch/Dinner": "Lunch_Dinner", "Dessert": "Dessert",
               "Breakfast": "Breakfast", "Snack": "Snack", "Beverage": "Beverage"}
class_names = list(target_le.classes_)

@st.cache_resource
def load_dice():
    import dice_ml
    from dice_ml import Dice
    train_for_dice = pd.read_csv("diabetes_prediction/train_final.csv")
    feature_names = [c for c in train_for_dice.columns if c != "target"]
    dice_data = dice_ml.Data(dataframe=train_for_dice, continuous_features=feature_names, outcome_name="target")
    dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice_exp = Dice(dice_data, dice_model, method="random")
    return dice_exp, feature_names

dice_exp, dice_feature_names = load_dice()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_name(name):
    return re.sub(r"\s*\*\s*$", "", str(name)).strip()

fried_keywords = ["fry", "fried"]

def is_fried(name):
    return any(kw in str(name).lower() for kw in fried_keywords)

_non_diabetic_idx = int(np.where(target_le.classes_ == "Non-Diabetic")[0][0])
_modifiable_features = ["hbA1c_level", "blood_glucose_level", "bmi"]
_permitted_range = {
    "hbA1c_level": [-2.09, 2.63],
    "blood_glucose_level": [-1.42, 2.80],
    "bmi": [-1.90, 2.29],
}
_original_col_order = ["gender", "age", "race:AfricanAmerican", "race:Asian",
                        "race:Caucasian", "race:Hispanic", "race:Other",
                        "hypertension", "heart_disease", "smoking_history",
                        "bmi", "hbA1c_level", "blood_glucose_level"]
_feat_idx = [_original_col_order.index(f) for f in dice_feature_names]

def get_counterfactual_summary(patient_df_row):
    try:
        cf = dice_exp.generate_counterfactuals(
            patient_df_row, total_CFs=1, desired_class=_non_diabetic_idx,
            features_to_vary=_modifiable_features, permitted_range=_permitted_range,
        )
        cf_row = cf.cf_examples_list[0].final_cfs_df.iloc[0]
        means = scaler.mean_[_feat_idx]
        scales = scaler.scale_[_feat_idx]
        orig_real = patient_df_row.iloc[0][dice_feature_names].values * scales + means
        cf_real = cf_row[dice_feature_names].values * scales + means
        changes = []
        for j, feat in enumerate(dice_feature_names):
            delta = cf_real[j] - orig_real[j]
            if feat in ("hbA1c_level", "blood_glucose_level", "bmi") and delta >= -0.3:
                continue
            if abs(delta) > 0.3:
                changes.append(f"{feat} from {orig_real[j]:.1f} to {cf_real[j]:.1f}")
        return "; ".join(changes) if changes else None
    except Exception:
        return None

KG_SOURCE_LOG = []

def get_candidates_kg(meal_type_filter, role, allowed_gi, apply_gi_filter):
    """Retrieve candidate foods via SPARQL over the RDF knowledge graph.
    Returns None on failure/empty so caller can fall back to pandas."""
    if kg is None:
        return None
    try:
        meal_slug = KG_MEAL_MAP.get(meal_type_filter, meal_type_filter.replace("/", "_"))
        gi_filter = ""
        if apply_gi_filter and allowed_gi:
            gi_values = " ".join(f"food:{gi}" for gi in allowed_gi)
            gi_filter = f"VALUES ?gi {{ {gi_values} }}"
        query = f"""
        PREFIX food: <http://example.org/food-ontology#>
        SELECT ?label WHERE {{
            ?f a food:Food ;
               rdfs:label ?label ;
               food:hasRole food:{role} ;
               food:hasMealType food:{meal_slug} ;
               food:isStandalone true .
            {"?f food:hasGICategory ?gi ." if apply_gi_filter and allowed_gi else ""}
            {gi_filter}
        }}
        """
        labels = [str(row.label) for row in kg.query(query)]
        if not labels:
            return None
        cands = food_df[food_df["Food_Name_English"].isin(labels)].copy()
        return cands if not cands.empty else None
    except Exception:
        return None

def get_candidates(meal_type_filter, role, allowed_gi, apply_gi_filter):
    kg_result = get_candidates_kg(meal_type_filter, role, allowed_gi, apply_gi_filter)
    if kg_result is not None:
        KG_SOURCE_LOG.append("KG")
        return kg_result
    KG_SOURCE_LOG.append("Fallback")
    # Fallback: rule-based pandas filtering (used if KG is unavailable or empty)
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
    animal_groups = ["09 Fish, shellfish and their products",
                      "10 Meat, poultry and their products",
                      "11 Eggs and their products"]
    is_raw_animal = (protein_cands["Food_Group"].isin(animal_groups)) & \
                     (protein_cands["Food_Name_English"].str.contains("raw", case=False, na=False))
    safe_protein = protein_cands[~is_raw_animal]
    if len(safe_protein) > 0:
        protein_cands = safe_protein
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

    carb_item = add_portion(carb_item, carb_item["Energy_kcal"] if carb_item is not None else 0)
    protein_item = add_portion(protein_item, remaining_cal * 0.6)
    veg_item = add_portion(veg_item, max(remaining_cal2, 0))

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

RISK_TIER_MAP = {
    "Non-Diabetic": ("Low", 0.475, "any"),
    "Pre-Diabetic": ("Medium", 0.375, "low_medium"),
    "Diabetic": ("High", 0.28, "low_only"),
}

def run_prediction(patient):
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    full_input = pd.DataFrame([{
        "gender": gender_map.get(patient["gender"], 0), "age": patient["age"],
        "race:AfricanAmerican": 0, "race:Asian": 0, "race:Caucasian": 0,
        "race:Hispanic": 0, "race:Other": 0, "hypertension": 0, "heart_disease": 0,
        "smoking_history": 0, "bmi": patient["bmi"],
        "hbA1c_level": patient["hba1c_level"], "blood_glucose_level": patient["blood_glucose_level"]
    }])
    scaled_array = scaler.transform(full_input)
    scaled_df = pd.DataFrame(scaled_array, columns=full_input.columns)
    patient_df = scaled_df[["hbA1c_level", "blood_glucose_level", "age", "bmi"]]

    pred_idx = model.predict(patient_df)[0]
    pred_proba = model.predict_proba(patient_df)[0]
    prediction = class_names[pred_idx]
    confidence = pred_proba[pred_idx]

    if prediction == "Non-Diabetic":
        is_borderline = confidence < 0.85
    else:
        is_borderline = confidence < 0.65

    return {
        "patient": patient, "patient_df": patient_df, "pred_idx": pred_idx,
        "prediction": prediction, "confidence": confidence,
        "full_probabilities": dict(zip(class_names, pred_proba)),
        "is_borderline": is_borderline,
    }

def generate_diet_chart(result):
    KG_SOURCE_LOG.clear()
    patient = result["patient"]
    patient_df = result["patient_df"]
    prediction = result["prediction"]
    pred_idx = result["pred_idx"]
    confidence = result["confidence"]
    is_borderline = result["is_borderline"]

    risk_tier, carb_percent, gi_restriction = RISK_TIER_MAP[prediction]
    allowed_gi = {"low_only": ["Low"], "low_medium": ["Low", "Medium"], "any": ["Low", "Medium", "High"]}[gi_restriction]

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

    patient_shap = explainer.shap_values(patient_df)
    if isinstance(patient_shap, list):
        class_shap = patient_shap[pred_idx][0]
    elif patient_shap.ndim == 3:
        class_shap = patient_shap[0, :, pred_idx]
    else:
        class_shap = patient_shap[0]
    shap_lines = []
    for feature, value in zip(patient_df.columns, class_shap):
        direction = "increases" if value > 0 else "decreases"
        shap_lines.append(f"{feature} {direction} likelihood of '{prediction}' (impact: {abs(value):.2f})")
    shap_text = "\n".join(shap_lines)

    counterfactual_text = "N/A"
    if prediction != "Non-Diabetic":
        cf_summary = get_counterfactual_summary(patient_df)
        if cf_summary:
            counterfactual_text = cf_summary

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

    return {
        "chart_text": response.text, "shap_lines": shap_lines, "risk_tier": risk_tier,
        "tdee": tdee, "daily_carb_grams": daily_carb_grams, "counterfactual_text": counterfactual_text,
        "kg_hits": KG_SOURCE_LOG.count("KG"), "fallback_hits": KG_SOURCE_LOG.count("Fallback"),
    }

# ============================================
# PAGE ROUTING
# ============================================
if "page" not in st.session_state:
    st.session_state["page"] = "form"

# --------------------------------------------
# PAGE 1: INPUT FORM
# --------------------------------------------
if st.session_state["page"] == "form":
    st.title("🍽️ Diabetes Risk & Diet Recommendation System")
    st.caption("Ternary classification (Non-Diabetic / Pre-Diabetic / Diabetic) · Stacking Ensemble · SHAP · DiCE Counterfactuals · Bangladeshi FCTB Food Data")
    st.write("Enter patient details to predict diabetes risk and get a personalized diet chart.")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
        with col2:
            activity_level = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])
            hba1c_level = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
            blood_glucose_level = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=400, value=120)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        patient = {
            "age": age, "gender": gender, "weight_kg": weight_kg, "height_cm": height_cm,
            "activity_level": activity_level, "hba1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level, "bmi": bmi
        }
        with st.spinner("Predicting..."):
            result = run_prediction(patient)
        st.session_state["result"] = result
        st.session_state["chart_data"] = None
        st.session_state["page"] = "confirm" if (result["prediction"] == "Non-Diabetic" and not result["is_borderline"]) else "generate"
        st.rerun()

# --------------------------------------------
# PAGE 1.5: CONFIRM (only for confidently non-diabetic patients)
# --------------------------------------------
elif st.session_state["page"] == "confirm":
    result = st.session_state["result"]
    st.subheader("📊 Prediction Result")
    st.metric("Prediction", "Non-Diabetic", f"{result['confidence']:.1%} confidence")
    st.info(f"Good news! Low diabetes risk. HbA1c: {result['patient']['hba1c_level']}%, "
            f"Glucose: {result['patient']['blood_glucose_level']} mg/dL are within healthy range.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate preventive diet chart"):
            st.session_state["page"] = "generate"
            st.rerun()
    with col2:
        if st.button("No thanks, just show result"):
            st.session_state["page"] = "results"
            st.rerun()

# --------------------------------------------
# PAGE 2: GENERATE CHART (runs the Gemini call)
# --------------------------------------------
elif st.session_state["page"] == "generate":
    result = st.session_state["result"]
    if result["is_borderline"] and result["prediction"] == "Non-Diabetic":
        st.info(f"Non-Diabetic but BORDERLINE (confidence {result['confidence']:.1%}) — generating a preventive chart automatically.")
    with st.spinner("Generating your personalized diet chart..."):
        chart_data = generate_diet_chart(result)
    st.session_state["chart_data"] = chart_data
    st.session_state["page"] = "results"
    st.rerun()

# --------------------------------------------
# PAGE 3: RESULTS (prediction only)
# --------------------------------------------
elif st.session_state["page"] == "results":
    result = st.session_state["result"]
    chart_data = st.session_state.get("chart_data")
    prediction = result["prediction"]
    confidence = result["confidence"]

    st.title("📊 Prediction Result")

    if st.button("⬅ Back to form (new patient)"):
        st.session_state["page"] = "form"
        st.rerun()

    st.metric("Prediction", prediction, f"{confidence:.1%} confidence")
    if result["is_borderline"]:
        st.warning("⚠️ This result is BORDERLINE (close to the next risk category) — closer monitoring recommended.")

    with st.expander("Full class probabilities"):
        for cls, prob in result["full_probabilities"].items():
            st.write(f"- {cls}: {prob:.1%}")

    if chart_data:
        st.subheader("🎯 Daily Targets")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Tier", chart_data["risk_tier"])
        c2.metric("Calorie Target", f"{chart_data['tdee']:.0f} kcal")
        c3.metric("Carb Target", f"{chart_data['daily_carb_grams']:.0f} g")

        with st.expander("🔍 Why this prediction? (SHAP Explanation)"):
            for line in chart_data["shap_lines"]:
                st.write(f"- {line}")

        if chart_data["counterfactual_text"] != "N/A":
            with st.expander("🔄 What would improve this risk tier? (DiCE Counterfactual)"):
                st.write(chart_data["counterfactual_text"])

        st.write("")
        if st.button("📋 View Full Diet Chart →"):
            st.session_state["page"] = "diet_chart"
            st.rerun()
    else:
        st.write("No diet chart was generated for this session.")

# --------------------------------------------
# PAGE 4: DIET CHART (separate page)
# --------------------------------------------
elif st.session_state["page"] == "diet_chart":
    chart_data = st.session_state["chart_data"]

    st.title("📋 Your Personalized Diet Chart")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back to Prediction Result"):
            st.session_state["page"] = "results"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over (new patient)"):
            st.session_state["page"] = "form"
            st.rerun()

    kg_hits = chart_data.get("kg_hits", 0)
    fallback_hits = chart_data.get("fallback_hits", 0)
    total = kg_hits + fallback_hits
    if total > 0:
        st.caption(f"🔗 Source: Knowledge Graph ({kg_hits}/{total} queries) "
                    f"· Rule-based Fallback ({fallback_hits}/{total} queries)")
    st.markdown(chart_data["chart_text"])