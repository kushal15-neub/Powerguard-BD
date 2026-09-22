import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# =========================================================
# 1. LOAD DATA
# =========================================================

file = r"d:\PowerGrid_BD\sylhet_2023_2025_prediction_dataset.csv"

df = pd.read_csv(file)

df["date"] = pd.to_datetime(df["date"])


# =========================================================
# 2. FEATURES
# =========================================================

features = [
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

target = "next_day_risk"


# =========================================================
# 3. TRAIN / VALIDATION / TEST
# =========================================================

# 2023 = training
train = df[df["date"].dt.year == 2023].copy()

# 2024 = validation
validation = df[df["date"].dt.year == 2024].copy()

# 2025 = final test
test = df[df["date"].dt.year == 2025].copy()


X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]


print("\n========================================")
print("DATA SPLIT")
print("========================================")

print("Training:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

print("\nTraining target:")
print(y_train.value_counts())

print("\nValidation target:")
print(y_val.value_counts())

print("\nTest target:")
print(y_test.value_counts())


# =========================================================
# 4. TRAIN XGBOOST
# =========================================================

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


# =========================================================
# 5. VALIDATION PROBABILITIES
# =========================================================

val_prob = model.predict_proba(X_val)[:, 1]


# =========================================================
# 6. THRESHOLD SEARCH
# =========================================================

threshold_results = []

thresholds = np.arange(0.10, 0.91, 0.05)


for threshold in thresholds:

    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(y_val, val_pred, zero_division=0)

    recall = recall_score(y_val, val_pred, zero_division=0)

    f1 = f1_score(y_val, val_pred, zero_division=0)

    threshold_results.append(
        {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
    )


threshold_df = pd.DataFrame(threshold_results)


# =========================================================
# 7. SHOW RESULTS
# =========================================================

print("\n========================================")
print("THRESHOLD RESULTS")
print("========================================")

print(threshold_df.to_string(index=False))


# =========================================================
# 8. SELECT THRESHOLD USING F1
# =========================================================

best_row = threshold_df.loc[threshold_df["f1"].idxmax()]

best_threshold = best_row["threshold"]


print("\n========================================")
print("SELECTED THRESHOLD")
print("========================================")

print("Threshold:", best_threshold)

print("Validation Precision:", round(best_row["precision"], 4))

print("Validation Recall:", round(best_row["recall"], 4))

print("Validation F1:", round(best_row["f1"], 4))


# =========================================================
# 9. RETRAIN ON 2023 + 2024
# =========================================================

train_final = df[df["date"].dt.year <= 2024].copy()

X_train_final = train_final[features]
y_train_final = train_final[target]


final_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

final_model.fit(X_train_final, y_train_final)


# =========================================================
# 10. FINAL 2025 TEST
# =========================================================

test_prob = final_model.predict_proba(X_test)[:, 1]


test_pred = (test_prob >= best_threshold).astype(int)


# =========================================================
# 11. FINAL METRICS
# =========================================================

precision = precision_score(y_test, test_pred, zero_division=0)

recall = recall_score(y_test, test_pred, zero_division=0)

f1 = f1_score(y_test, test_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, test_prob)

pr_auc = average_precision_score(y_test, test_prob)


print("\n========================================")
print("FINAL 2025 TEST RESULT")
print("========================================")

print(f"Threshold: {best_threshold:.2f}")

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1 Score: {f1:.4f}")

print(f"ROC-AUC: {roc_auc:.4f}")

print(f"PR-AUC: {pr_auc:.4f}")


print("\nConfusion Matrix:")

print(confusion_matrix(y_test, test_pred))


# =========================================================
# 12. SAVE PREDICTIONS
# =========================================================

test_results = test[["date", "demand_mw", "load_shed_mw", "next_day_risk"]].copy()

test_results["predicted_probability"] = test_prob

test_results["predicted_risk"] = test_pred

test_results.to_csv(r"d:\PowerGrid_BD\xgboost_2025_predictions.csv", index=False)


print("\nPredictions saved:")
print(r"d:\PowerGrid_BD\xgboost_2025_predictions.csv")
