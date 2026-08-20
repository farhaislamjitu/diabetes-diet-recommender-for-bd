import openpyxl
import re
import pandas as pd

# ============================================
# STEP 1: LOAD THE EXCEL FILE
# ============================================
print("=== STEP 1: Loading FCTB Excel file ===")
wb = openpyxl.load_workbook(
    "diabetes_prediction/diet_data/FCDB_7_4_14.xlsx",
    read_only=True, data_only=True
)
ws = wb["UserDB_Main_table"]
print(f"Loaded sheet 'UserDB_Main_table'")

# ============================================
# STEP 2: DEFINE PATTERNS TO IDENTIFY ROW TYPES
# ============================================
# Actual food rows have codes like "01_0001"
code_pattern = re.compile(r"^\d{2}_\d{4}$")
# Food group header rows look like "01 Cereals and their products"
group_pattern = re.compile(r"^\d{2}\s+\D")

# ============================================
# STEP 3: HELPER TO PARSE THE ENERGY COLUMN
# ============================================
# Energy column stores values like "(324)1360" meaning (324 kcal) 1360 kJ
def parse_energy(val):
    if val is None or val == "":
        return None
    val = str(val)
    match = re.match(r"\((\d+\.?\d*)\)", val)
    if match:
        return float(match.group(1))
    try:
        return float(val)
    except ValueError:
        return None

# ============================================
# STEP 4: EXTRACT FOOD ROWS, TRACKING CURRENT GROUP
# ============================================
print("\n=== STEP 2: Extracting food entries ===")
current_group = ""
rows_out = []

for row in ws.iter_rows(min_row=3, values_only=True):
    code = row[0]
    name = row[1]

    if code is None and name is None:
        continue  # empty row, skip

    code_str = str(code).strip() if code else ""

    if code_pattern.match(code_str):
        # This is an actual food data row
        rows_out.append({
            "Code": code_str,
            "Food_Name_English": name,
            "Food_Name_Bengali": row[2],
            "Scientific_Name": row[3],
            "Food_Group": current_group,
            "Edible_Portion": row[5],
            "Energy_kcal": parse_energy(row[6]),
            "Water_g": row[7],
            "Protein_g": row[8],
            "Fat_g": row[9],
            "Carbohydrate_g": row[10],
            "Fibre_g": row[11],
            "Ash_g": row[12],
            "Calcium_mg": row[13],
            "Iron_mg": row[14],
            "Sodium_mg": row[18] if len(row) > 18 else None,
        })
    elif group_pattern.match(code_str) and name is None:
        # This is a food-group header row (e.g. "01 Cereals and their products")
        current_group = code_str
    # else: this is a metadata row ("SD or min-max", "n") — skip it

print(f"Total food rows extracted: {len(rows_out)}")

# ============================================
# STEP 5: CONVERT TO DATAFRAME AND CLEAN
# ============================================
print("\n=== STEP 3: Cleaning data ===")
df = pd.DataFrame(rows_out)

print(f"Shape before cleaning: {df.shape}")

# Check missing values in key columns
key_cols = ["Energy_kcal", "Protein_g", "Fat_g", "Carbohydrate_g", "Fibre_g"]
print("\nMissing values in key columns:")
print(df[key_cols].isnull().sum())

# Drop rows with missing Energy (only 1 expected)
df = df.dropna(subset=["Energy_kcal"])
print(f"\nShape after dropping missing-Energy rows: {df.shape}")

# ============================================
# STEP 6: SUMMARY
# ============================================
print("\n=== STEP 4: Food Group Distribution ===")
print(df["Food_Group"].value_counts())

print("\n=== Sample rows ===")
print(df[["Food_Name_English", "Food_Name_Bengali", "Food_Group",
          "Energy_kcal", "Protein_g", "Carbohydrate_g"]].head(10).to_string(index=False))

# ============================================
# STEP 7: SAVE CLEANED CSV
# ============================================
df.to_csv("diabetes_prediction/diet_data/fctb_bangladesh_clean.csv", index=False, encoding="utf-8-sig")
df.to_csv("diabetes_prediction/diet_data/fctb_bangladesh_clean.csv", index=False, sep="\t", encoding="utf-8-sig")