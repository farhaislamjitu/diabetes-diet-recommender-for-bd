# ============================================
# STEP 4: GEMINI CLASSIFICATION (only for ambiguous "Needs_Review" items)
# ============================================
# Only ~46 items (Nuts, Milk, Beverages, Miscellaneous groups) need Gemini,
# instead of all 380 -- much cheaper and faster than the old approach.

import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai

# ── 1. SETUP ────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ── 2. LOAD DATA, SPLIT REVIEW vs ALREADY-DONE ──
print("=== STEP 1: Loading data ===")
df = pd.read_csv("diabetes_prediction/diet_data/fctb_with_role_partial.csv")
print(f"Loaded {df.shape[0]} items total")

needs_review_mask = (df["Food_Role"] == "Needs_Review") | (df["Is_Standalone"] == "Needs_Review")
review_df = df[needs_review_mask].copy()
print(f"Items needing Gemini classification: {len(review_df)}")

food_names = review_df["Food_Name_English"].tolist()

# ── 3. BATCH CLASSIFY WITH GEMINI ───────────────
print("\n=== STEP 2: Classifying with Gemini ===")
BATCH_SIZE = 50
results = {}

for i in range(0, len(food_names), BATCH_SIZE):
    batch = food_names[i:i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1
    print(f"Processing batch {batch_num}...")

    item_list_text = "\n".join([f"{idx+1}. {name}" for idx, name in enumerate(batch)])

    prompt = f"""You are a nutrition expert familiar with Bangladeshi food. For each food item
below, classify TWO things:

1. "role": what role this food plays in a meal:
   - "Carb" = a carbohydrate staple
   - "Protein" = a protein-rich item (meat, fish, egg, lentils, milk, nuts as protein source)
   - "Vegetable" = a vegetable/fruit side or light item
   - "Mixed" = a combined dish with significant carb AND protein
   - "Condiment" = not a meal component at all (pure oil, spice, seasoning, sweetener)

2. "standalone": whether this is something a person would actually eat/drink as part of
   a meal or as a snack/beverage ("Yes"), OR if it's purely an ingredient/additive that is
   never consumed by itself ("No")

Respond with ONLY a valid JSON object, no extra text, in this exact format:
{{"1": {{"role": "Protein", "standalone": "Yes"}}, "2": {{"role": "Condiment", "standalone": "No"}}, ...}}

Food items:
{item_list_text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        batch_result = json.loads(text)
        for idx, name in enumerate(batch):
            key = str(idx + 1)
            item = batch_result.get(key, {"role": "Mixed", "standalone": "Yes"})
            results[name] = item

    except Exception as e:
        print(f"  Error on batch {batch_num}: {e}")
        for name in batch:
            results[name] = {"role": "Mixed", "standalone": "Yes"}

    time.sleep(4)

# ── 4. APPLY GEMINI RESULTS BACK TO THE MAIN DF ─
print("\n=== STEP 3: Merging Gemini results back ===")
for name, item in results.items():
    idx = df[df["Food_Name_English"] == name].index
    df.loc[idx, "Food_Role"] = item.get("role", "Mixed")
    df.loc[idx, "Is_Standalone"] = item.get("standalone", "Yes")

# ── 5. FINAL SUMMARY ────────────────────────────
print("\n=== Final Food_Role distribution ===")
print(df["Food_Role"].value_counts())
print("\n=== Final Is_Standalone distribution ===")
print(df["Is_Standalone"].value_counts())

remaining_review = df[(df["Food_Role"] == "Needs_Review") | (df["Is_Standalone"] == "Needs_Review")]
print(f"\nRemaining unclassified (should be 0): {len(remaining_review)}")

# ── 6. SAVE ──────────────────────────────────────
df.to_csv("diabetes_prediction/diet_data/fctb_with_roles.csv", index=False)
print("\n=== Saved -> diet_data/fctb_with_roles.csv ===")
print("Gemini classification complete!")