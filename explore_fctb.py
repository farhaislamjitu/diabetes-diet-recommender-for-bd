# ============================================
# STEP 1: EXPLORE FCTB (Food Composition Table for Bangladesh)
# ============================================

import pandas as pd

# NOTE: this file is TAB-separated (not comma), with double-quote
# quoting for fields that contain commas (e.g. "Barley, whole-grain, raw")
print("=== STEP 1: Loading FCTB dataset ===")
df = pd.read_csv(
    "diabetes_prediction/diet_data/fctb_bangladesh_clean.csv",
    sep="\t",
    quotechar='"'
)
print(f"Loaded {df.shape[0]} food items, {df.shape[1]} columns")

print("\n=== STEP 2: Columns ===")
print(df.columns.tolist())

print("\n=== STEP 3: Food_Group distribution ===")
print(df["Food_Group"].value_counts())

print("\n=== STEP 4: Missing values per column ===")
print(df.isnull().sum())

print("\n=== STEP 5: Sample rows ===")
print(df[["Food_Name_English", "Food_Group", "Energy_kcal",
          "Protein_g", "Fat_g", "Carbohydrate_g", "Fibre_g"]].head(10))

print("\n=== Exploration Complete! ===")