# ============================================
# STEP 6: FOOD CLUSTERING for FCTB (K-Means, k=3)
# ============================================
# Same logic as before: cluster foods by calorie/carb/fibre/protein/GI profile,
# then label clusters as Diabetic-Safe / Moderate / Limit using a normalized
# risk score, with a hard rule that high-calorie clusters can never be "Safe".

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ── 1. LOAD FCTB (with roles + meal types) ─────
print("=== STEP 1: Loading FCTB dataset ===")
df = pd.read_csv("diabetes_prediction/diet_data/fctb_final.csv")
print(f"Loaded {df.shape[0]} food items")

# ── 2. SELECT FEATURES FOR CLUSTERING ──────────
features = ["Energy_kcal", "Carbohydrate_g", "Fibre_g", "Protein_g", "Estimated_GI"]
X = df[features].copy()

# ── 3. SCALE FEATURES ──────────────────────────
print("\n=== STEP 2: Scaling features ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. ELBOW METHOD (for reference/report) ─────
print("\n=== STEP 3: Finding optimal k (Elbow Method) ===")
inertias = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}: inertia={km.inertia_:.1f}")

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k (FCTB)")
plt.tight_layout()
plt.savefig("diabetes_prediction/clustering_elbow_fctb.png")
plt.close()
print("Saved elbow chart -> diabetes_prediction/clustering_elbow_fctb.png")

# ── 5. FINAL MODEL (k=3) ───────────────────────
print("\n=== STEP 4: Training final K-Means model (k=3) ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ── 6. LABEL CLUSTERS BY RISK SCORE ────────────
cluster_summary = df.groupby("Cluster")[["Estimated_GI", "Carbohydrate_g", "Energy_kcal"]].mean()

gi_norm = (cluster_summary["Estimated_GI"] - cluster_summary["Estimated_GI"].min()) / \
          (cluster_summary["Estimated_GI"].max() - cluster_summary["Estimated_GI"].min()) * 100
carb_norm = (cluster_summary["Carbohydrate_g"] - cluster_summary["Carbohydrate_g"].min()) / \
            (cluster_summary["Carbohydrate_g"].max() - cluster_summary["Carbohydrate_g"].min()) * 100
cal_norm = (cluster_summary["Energy_kcal"] - cluster_summary["Energy_kcal"].min()) / \
           (cluster_summary["Energy_kcal"].max() - cluster_summary["Energy_kcal"].min()) * 100

cluster_summary["risk_score"] = (gi_norm * 0.4) + (carb_norm * 0.4) + (cal_norm * 0.2)

# HARD RULE: a cluster averaging >400 kcal cannot be "Diabetic-Safe"
CALORIE_SAFE_LIMIT = 400
sorted_by_risk = cluster_summary.sort_values("risk_score").index.tolist()

safe_eligible = [c for c in sorted_by_risk if cluster_summary.loc[c, "Energy_kcal"] <= CALORIE_SAFE_LIMIT]
safe_cluster = safe_eligible[0] if safe_eligible else sorted_by_risk[0]

remaining = [c for c in sorted_by_risk if c != safe_cluster]
label_map = {safe_cluster: "Diabetic-Safe"}
label_map[remaining[0]] = "Moderate"
label_map[remaining[1]] = "Limit"
labels_ordered = ["Diabetic-Safe", "Moderate", "Limit"]

df["Diet_Cluster"] = df["Cluster"].map(label_map)

# ── 7. SUMMARY ──────────────────────────────────
print("\n=== STEP 5: Cluster Summary (mean values) ===")
print(cluster_summary)

print("\n=== STEP 6: Diet_Cluster Distribution ===")
print(df["Diet_Cluster"].value_counts())

print("\n=== Sample items from each cluster ===")
for label in labels_ordered:
    print(f"\n--- {label} ---")
    sample = df[df["Diet_Cluster"] == label][
        ["Food_Name_English", "Energy_kcal", "Carbohydrate_g", "Estimated_GI"]
    ].head(5)
    print(sample.to_string(index=False))

# ── 8. SAVE ──────────────────────────────────────
df = df.drop(columns=["Cluster"])
df.to_csv("diabetes_prediction/diet_data/fctb_clustered.csv", index=False)
print("\n=== Saved -> diet_data/fctb_clustered.csv ===")
print(f"Final database: {df.shape[0]} items, {df.shape[1]} columns")