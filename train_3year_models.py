import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# 1. LOAD DATA
# =========================================================

file = r"d:\PowerGrid_BD\sylhet_2023_2025_prediction_dataset.csv"

df = pd.read_csv(file)

df["date"] = pd.to_datetime(df["date"])


# =========================================================
# 2. DEFINE FEATURES
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
# 3. TRAIN / TEST SPLIT
# =========================================================

# 2023 + 2024 = TRAIN
train = df[df["date"].dt.year <= 2024].copy()

# 2025 = TEST
test = df[df["date"].dt.year == 2025].copy()


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training samples:", len(train))
print("Testing samples :", len(test))

print("\nTraining target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# =========================================================
# 4. MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    ),
}


# =========================================================
# 5. TRAIN + EVALUATE
# =========================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    # Train
    model.fit(X_train, y_train)

    # Probability
    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold
    y_pred = (y_prob >= 0.50).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, zero_division=0)

    recall = recall_score(y_test, y_pred, zero_division=0)

    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob)

    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }
    )


# =========================================================
# 6. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# =========================================================
# 7. SAVE RESULTS
# =========================================================

output_file = r"d:\PowerGrid_BD\model_comparison_2023_2025.csv"

results_df.to_csv(output_file, index=False)

print("\nResults saved to:")
print(output_file)
