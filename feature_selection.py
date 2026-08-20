# ============================================
# FEATURE SELECTION — Train Data Only (leakage-free)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ── 1. LOAD TRAIN DATA ONLY ────────────────────
train = pd.read_csv("diabetes_prediction/train.csv")
test = pd.read_csv("diabetes_prediction/test.csv")

X_train = train.drop("target", axis=1)
y_train = train["target"]

# ── 2. FEATURE IMPORTANCE (fit on TRAIN only) ──
print("=== STEP 1: Feature Importance (Random Forest, train only) ===")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importance)

plt.figure(figsize=(10, 5))
importance.plot(kind="bar", color="darkorange", edgecolor="black")
plt.title("Feature Importance (Random Forest, Train Only)")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("diabetes_prediction/feature_importance.png")
plt.close()

# ── 3. SELECT TOP FEATURES ─────────────────────
top_features = importance[importance > 0.05].index.tolist()
print(f"\n=== STEP 2: Selected Features (importance > 5%) ===")
print(top_features)

# ── 4. APPLY SAME FEATURE SET TO TRAIN & TEST ──
train_final = train[top_features + ["target"]]
test_final = test[top_features + ["target"]]

train_final.to_csv("diabetes_prediction/train_final.csv", index=False)
test_final.to_csv("diabetes_prediction/test_final.csv", index=False)
print(f"\nSaved train_final.csv {train_final.shape}, test_final.csv {test_final.shape}")