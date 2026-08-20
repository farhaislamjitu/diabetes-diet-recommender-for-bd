# ============================================
# STEP 2: GI ESTIMATION for FCTB (Food Composition Table for Bangladesh)
# ============================================
# Same validated formula as before (Aidara et al., 2019, Nutrients),
# just adapted to FCTB's column names.

import pandas as pd

print("=== STEP 1: Loading FCTB dataset ===")
df = pd.read_csv(
    "diabetes_prediction/diet_data/fctb_bangladesh_clean.csv",
    sep="\t",
    quotechar='"'
)
print(f"Loaded {df.shape[0]} food items")

# ── 1b. CLEAN NUTRIENT COLUMNS ──────────────────
# Some values are stored as strings like "[0.9]" (means: estimated/imputed
# value in the original FCTB documentation) or "Tr" (trace amount, i.e.
# a very small, non-zero quantity). Convert both to plain numbers so the
# GI formula's arithmetic works.
nutrient_cols = ["Energy_kcal", "Water_g", "Protein_g", "Fat_g",
                  "Carbohydrate_g", "Fibre_g", "Ash_g", "Calcium_mg",
                  "Iron_mg", "Sodium_mg"]

def clean_numeric(value):
    if pd.isna(value):
        return value
    if isinstance(value, (int, float)):
        return value
    value = str(value).strip()
    if value == "Tr":
        return 0.01  # trace amount -> treat as a very small nonzero value
    value = value.strip("[]")  # "[0.9]" -> "0.9"
    try:
        return float(value)
    except ValueError:
        return None  # anything else unexpected -> missing

print("\n=== STEP 1b: Cleaning bracketed/trace values in nutrient columns ===")
for col in nutrient_cols:
    before_bad = df[col].apply(lambda x: isinstance(x, str)).sum()
    df[col] = df[col].apply(clean_numeric)
    if before_bad > 0:
        print(f"  {col}: cleaned {before_bad} non-numeric text values")

# ── 2. HANDLE MISSING FIBRE (needed for the GI formula) ──
# A few rows have missing Fibre_g; treat missing as 0 (conservative -- slightly
# raises estimated GI for those items rather than crashing or guessing).
df["Fibre_g"] = df["Fibre_g"].fillna(0)

# ── 3. GLYCEMIC CARBS (Total Carbs - Fibre) ────
df["Glycemic_Carbs"] = df["Carbohydrate_g"] - df["Fibre_g"]
df["Glycemic_Carbs"] = df["Glycemic_Carbs"].clip(lower=0)

# ── 4. GI FORMULA (Aidara et al., 2019) ────────
print("\n=== STEP 2: Estimating GI using validated regression formula ===")
numerator = df["Glycemic_Carbs"] * 100
denominator = (df["Glycemic_Carbs"]
               + 0.6 * df["Fat_g"]
               + 0.6 * df["Protein_g"]
               + 0.3 * df["Fibre_g"])

df["Estimated_GI"] = (numerator / denominator.replace(0, 1)).round(1)
df["Estimated_GI"] = df["Estimated_GI"].clip(lower=0, upper=100)

# ── 5. GI CATEGORY ──────────────────────────────
def gi_category(gi):
    if gi <= 55:
        return "Low"
    elif gi <= 69:
        return "Medium"
    else:
        return "High"

df["GI_Category"] = df["Estimated_GI"].apply(gi_category)

# ── 6. SUMMARY ──────────────────────────────────
print("\n=== STEP 3: GI Category Distribution ===")
print(df["GI_Category"].value_counts())

print("\n=== Sample Results (first 10 items) ===")
print(df[["Food_Name_English", "Carbohydrate_g", "Fibre_g",
          "Estimated_GI", "GI_Category"]].head(10))

# ── 7. SAVE ──────────────────────────────────────
df.to_csv("diabetes_prediction/diet_data/fctb_with_gi.csv", index=False)
print("\n=== Saved -> diet_data/fctb_with_gi.csv ===")
print(f"Final database: {df.shape[0]} items, {df.shape[1]} columns")