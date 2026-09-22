import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(r"d:\PowerGrid_BD\sylhet_prediction_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "demand_mw",
    "previous_load_shed",
    "load_shed_3day_avg",
    "load_shed_7day_avg",
    "previous_demand",
    "demand_3day_avg",
    "demand_change",
    "is_weekend",
]

target = "next_day_risk"


X = df[features]
y = df[target]


# ==========================================
# 3. TIME-BASED TRAIN/TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== TRAIN / TEST ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", random_state=42)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
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


# ==========================================
# 5. TRAIN + EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability for positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, zero_division=0)

    recall = recall_score(y_test, y_pred, zero_division=0)

    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

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
            "ROC-AUC": auc,
        }
    )


# ==========================================
# 6. COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))
