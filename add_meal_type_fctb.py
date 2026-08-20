# ============================================
# STEP 5: MEAL_TYPE ASSIGNMENT for FCTB
# ============================================
# Strategy: start from a Food_Group-based DEFAULT meal type (more reliable
# than pure keyword guessing), then apply keyword OVERRIDES for items that
# clearly belong to a different meal type (e.g. sweet items -> Dessert,
# bread/rice-flakes -> Breakfast, roasted nuts/popcorn -> Snack).
# Order matters: Dessert/Beverage checked first, then Breakfast, then Snack;
# anything left over falls back to the group default.

import re
import pandas as pd

print("=== STEP 1: Loading FCTB dataset (with roles) ===")
df = pd.read_csv("diabetes_prediction/diet_data/fctb_with_roles.csv")
print(f"Loaded {df.shape[0]} food items")

# ── 2. GROUP -> DEFAULT MEAL TYPE ───────────────
GROUP_DEFAULT_MEAL_TYPE = {
    "14 Beverages": "Beverage",
    "06 Nuts, seeds and their products": "Snack",
}
DEFAULT_FALLBACK = "Lunch/Dinner"

# ── 3. KEYWORD OVERRIDES (Bangladeshi-food-relevant) ─
dessert_keywords = ["payesh", "sandesh", "misti", "rasgulla",
                     "kheer", "halwa", "pudding", "jalebi", "condensed, sweetened",
                     "curd, sweetened", "biscuit, sweet"]

breakfast_keywords = ["bread", "toast", "rice flakes", "vermicelli",
                       "semolina", "egg", "omelette", "porridge"]

snack_keywords = ["popcorn", "puffed", "chips", "roasted", "biscuit",
                   "cookie", "wafer", "fry", "fried"]

def has_keyword(name_lower, keyword_list):
    for kw in keyword_list:
        if kw in name_lower:  # substring match is fine here (multi-word phrases)
            return True
    return False

def assign_meal_type(row):
    name_lower = row["Food_Name_English"].lower()
    group = row["Food_Group"]

    # condiments/non-standalone items don't really need a meaningful meal type,
    # but assign one anyway for completeness (won't be used in meal composition)
    if has_keyword(name_lower, dessert_keywords):
        return "Dessert"
    if has_keyword(name_lower, breakfast_keywords):
        return "Breakfast"
    # "fry"/"fried" vegetable side dishes (e.g. "Gourd, bitter, fry") are
    # still a Lunch/Dinner side dish, not a snack -- only treat fried/roasted
    # items as Snack when they are NOT a Vegetable-role side dish.
    if has_keyword(name_lower, snack_keywords) and row.get("Food_Role") != "Vegetable":
        return "Snack"

    return GROUP_DEFAULT_MEAL_TYPE.get(group, DEFAULT_FALLBACK)

print("\n=== STEP 2: Assigning Meal_Type ===")
df["Meal_Type"] = df.apply(assign_meal_type, axis=1)

# ── 4. SUMMARY ──────────────────────────────────
print("\n=== Meal_Type distribution ===")
print(df["Meal_Type"].value_counts())

print("\n=== Sample from each Meal_Type ===")
for mt in df["Meal_Type"].unique():
    print(f"\n--- {mt} ---")
    print(df[df["Meal_Type"] == mt]["Food_Name_English"].head(6).tolist())

# ── 5. SAVE FINAL FOOD DATABASE ─────────────────
df.to_csv("diabetes_prediction/diet_data/fctb_final.csv", index=False)
print("\n=== Saved -> diet_data/fctb_final.csv ===")
print(f"Final database: {df.shape[0]} items, {df.shape[1]} columns")