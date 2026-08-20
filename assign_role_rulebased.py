# ============================================
# STEP 3: FOOD_GROUP -> FOOD_ROLE (rule-based mapping)
# ============================================
# Most Food_Group values map RELIABLY to a role without needing Gemini.
# Only a few ambiguous groups are left as "Needs_Review" for a Gemini
# fallback step next (much cheaper than classifying all 380 items).

import pandas as pd

print("=== STEP 1: Loading FCTB dataset (with GI already added) ===")
df = pd.read_csv("diabetes_prediction/diet_data/fctb_with_gi.csv")
print(f"Loaded {df.shape[0]} food items")

# ── 2. RELIABLE GROUP -> ROLE MAPPING ──────────
GROUP_TO_ROLE = {
    "01 Cereals and their products": "Carb",
    "05 Starchy roots, tubers and their products": "Carb",
    "02 Pulses, legumes and their products": "Protein",
    "09 Fish, shellfish and their products": "Protein",
    "10 Meat, poultry and their products": "Protein",
    "11 Eggs and their products": "Protein",
    "03 Vegetables and their products": "Vegetable",
    "04 Leafy vegetables": "Vegetable",
    "08 Fruits": "Vegetable",  # treated as a light side/fruit role
}

# ── 3. GROUPS THAT SHOULD NEVER BE A MAIN DISH ──
# (condiments, pure fats/oils are never a standalone meal component)
NOT_STANDALONE_GROUPS = [
    "07 Spices, condiments and herbs",
    "13 Fat and oils",
]

# ── 4. GROUPS THAT NEED GEMINI (ambiguous role) ─
AMBIGUOUS_GROUPS = [
    "06 Nuts, seeds and their products",
    "12 Milk and its product",
    "14 Beverages",
    "15 Miscellaneous",
]

def assign_role(group):
    if group in GROUP_TO_ROLE:
        return GROUP_TO_ROLE[group]
    if group in NOT_STANDALONE_GROUPS:
        return "Condiment"  # will be excluded from meal composition entirely
    return "Needs_Review"  # ambiguous groups -> Gemini fallback next step

def assign_standalone(group):
    if group in NOT_STANDALONE_GROUPS:
        return "No"
    if group in AMBIGUOUS_GROUPS:
        return "Needs_Review"
    return "Yes"

print("\n=== STEP 2: Assigning Food_Role and Is_Standalone (rule-based) ===")
df["Food_Role"] = df["Food_Group"].apply(assign_role)
df["Is_Standalone"] = df["Food_Group"].apply(assign_standalone)

# ── 5. SUMMARY ──────────────────────────────────
print("\n=== Food_Role distribution ===")
print(df["Food_Role"].value_counts())

print("\n=== Is_Standalone distribution ===")
print(df["Is_Standalone"].value_counts())

needs_review = df[(df["Food_Role"] == "Needs_Review") | (df["Is_Standalone"] == "Needs_Review")]
print(f"\n=== Items needing Gemini review: {len(needs_review)} (out of {len(df)}) ===")
print(needs_review["Food_Group"].value_counts())

# ── 6. SAVE ──────────────────────────────────────
df.to_csv("diabetes_prediction/diet_data/fctb_with_role_partial.csv", index=False)
print("\n=== Saved -> diet_data/fctb_with_role_partial.csv ===")
print("Next step: Gemini classification will run ONLY on the 'Needs_Review' items.")