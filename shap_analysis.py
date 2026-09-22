import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
import os

# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_PATH = r"D:\PowerGrid_BD\sylhet_2023_2025_prediction_dataset.csv"
PREDICTION_PATH = r"D:\PowerGrid_BD\xgboost_2025_predictions.csv"

OUTPUT_DIR = r"D:\PowerGrid_BD\shap_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "demand_mw",
    "previous_load_shed",
    "load_shed_3day_avg",
    "load_shed_7day_avg",
    "previous_demand",
    "demand_3day_avg",
    "demand_change",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
]

TARGET = "next_day_risk"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
pred_df = pd.read_csv(PREDICTION_PATH)

df["date"] = pd.to_datetime(df["date"])
pred_df["date"] = pd.to_datetime(pred_df["date"])

print(f"Full dataset shape: {df.shape}")
print(f"Prediction file shape: {pred_df.shape}")


# ============================================================
# 4. CREATE TRAIN / TEST SPLIT
# ============================================================

train_df = df[df["year"].isin([2023, 2024])].copy()
test_df = df[df["year"] == 2025].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nTraining data:")
print(f"Rows: {len(X_train)}")

print("\n2025 test data:")
print(f"Rows: {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\n2025 test class distribution:")
print(y_test.value_counts())


# ============================================================
# 5. TRAIN THE SAME XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

print("XGBoost training completed.")


# ============================================================
# 6. VERIFY PREDICTIONS
# ============================================================

test_prob = model.predict_proba(X_test)[:, 1]

print("\nPrediction probability range:")
print(f"Minimum: {test_prob.min():.6f}")
print(f"Maximum: {test_prob.max():.6f}")
print(f"Mean:    {test_prob.mean():.6f}")


# ============================================================
# 7. SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

explainer = shap.TreeExplainer(model)

shap_output = explainer(X_test)

# SHAP Explanation object
shap_values = shap_output

print("SHAP calculation completed.")

print(f"SHAP matrix shape: {shap_values.values.shape}")


# ============================================================
# 8. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs_shap})

importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(
    drop=True
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

importance_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")

importance_df.to_csv(importance_path, index=False)

print(f"\nSaved: {importance_path}")


# ============================================================
# 9. SHAP BAR PLOT
# ============================================================

print("\nCreating SHAP bar plot...")

plt.figure(figsize=(10, 7))

shap.plots.bar(shap_values, max_display=12, show=False)

plt.title("SHAP Feature Importance - XGBoost\n" "Sylhet Next-Day Load-Shedding Risk")

plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_bar.png")

plt.savefig(bar_path, dpi=300, bbox_inches="tight")

plt.close()

print(f"Saved: {bar_path}")


# ============================================================
# 10. SHAP BEESWARM PLOT
# ============================================================

print("\nCreating SHAP beeswarm plot...")

plt.figure(figsize=(10, 8))

shap.plots.beeswarm(shap_values, max_display=12, show=False)

plt.title("SHAP Summary Plot - XGBoost\n" "Sylhet Next-Day Load-Shedding Risk")

plt.tight_layout()

beeswarm_path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")

plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")

plt.close()

print(f"Saved: {beeswarm_path}")


# ============================================================
# 11. FIND HIGHEST-RISK 2025 DAY
# ============================================================

highest_idx = np.argmax(test_prob)

highest_date = test_df.iloc[highest_idx]["date"]
highest_probability = test_prob[highest_idx]
actual_risk = test_df.iloc[highest_idx][TARGET]

print("\n" + "=" * 70)
print("HIGHEST-RISK 2025 PREDICTION")
print("=" * 70)

print(f"Date:              {highest_date.date()}")
print(f"Predicted risk:    {highest_probability:.4f}")
print(f"Actual next-day risk: {actual_risk}")


# ============================================================
# 12. WATERFALL PLOT FOR HIGHEST-RISK DAY
# ============================================================

print("\nCreating SHAP waterfall plot...")

single_explanation = shap_values[highest_idx]

plt.figure(figsize=(12, 8))

shap.plots.waterfall(single_explanation, max_display=12, show=False)

plt.tight_layout()

waterfall_path = os.path.join(OUTPUT_DIR, "shap_highest_risk_waterfall.png")

plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")

plt.close()

print(f"Saved: {waterfall_path}")


# ============================================================
# 13. SAVE ROW-LEVEL SHAP VALUES
# ============================================================

print("\nSaving row-level SHAP values...")

shap_values_df = pd.DataFrame(shap_values.values, columns=FEATURES)

shap_values_df.insert(0, "date", test_df["date"].values)

shap_values_df["actual_next_day_risk"] = y_test.values
shap_values_df["predicted_probability"] = test_prob

shap_path = os.path.join(OUTPUT_DIR, "shap_values_2025.csv")

shap_values_df.to_csv(shap_path, index=False)

print(f"Saved: {shap_path}")


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. shap_feature_importance.csv")
print("2. shap_feature_importance_bar.png")
print("3. shap_beeswarm.png")
print("4. shap_highest_risk_waterfall.png")
print("5. shap_values_2025.csv")
