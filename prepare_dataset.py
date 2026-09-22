import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(r"d:\PowerGrid_BD\bpdb_area_wise_2024.csv")

df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# Sort properly
df = df.sort_values(["zone", "date"]).reset_index(drop=True)


# ==========================================
# 2. SELECT SYLHET
# ==========================================

sylhet = df[df["zone"] == "Sylhet"].copy()

sylhet = sylhet.sort_values("date")


# ==========================================
# 3. CREATE TIME FEATURES
# ==========================================

sylhet["day_of_week"] = sylhet["date"].dt.dayofweek
sylhet["month"] = sylhet["date"].dt.month
sylhet["day_of_month"] = sylhet["date"].dt.day

sylhet["is_weekend"] = (sylhet["day_of_week"] >= 5).astype(int)


# ==========================================
# 4. PREVIOUS LOAD-SHEDDING FEATURES
# ==========================================

sylhet["previous_load_shed"] = sylhet["load_shed_mw"].shift(1)

sylhet["load_shed_3day_avg"] = sylhet["load_shed_mw"].shift(1).rolling(3).mean()

sylhet["load_shed_7day_avg"] = sylhet["load_shed_mw"].shift(1).rolling(7).mean()


# ==========================================
# 5. DEMAND HISTORY
# ==========================================

sylhet["previous_demand"] = sylhet["demand_mw"].shift(1)

sylhet["demand_3day_avg"] = sylhet["demand_mw"].shift(1).rolling(3).mean()

sylhet["demand_change"] = sylhet["demand_mw"] - sylhet["previous_demand"]


# ==========================================
# 6. NEXT-DAY TARGET
# ==========================================

# We want tomorrow's load shedding
sylhet["next_day_load_shed"] = sylhet["load_shed_mw"].shift(-1)

# Binary target
sylhet["next_day_risk"] = (sylhet["next_day_load_shed"] > 0).astype(int)


# ==========================================
# 7. REMOVE ROWS WHERE NEXT CALENDAR DAY
#    IS NOT ACTUALLY AVAILABLE
# ==========================================

sylhet["next_date"] = sylhet["date"].shift(-1)

sylhet["date_difference"] = (sylhet["next_date"] - sylhet["date"]).dt.days

# Keep only consecutive dates
sylhet = sylhet[sylhet["date_difference"] == 1].copy()


# ==========================================
# 8. REMOVE MISSING FEATURE ROWS
# ==========================================

sylhet = sylhet.dropna()


# ==========================================
# 9. SHOW RESULT
# ==========================================

print("\n========== PREPARED DATASET ==========")

print("Rows:", len(sylhet))

print("\nColumns:")
print(sylhet.columns.tolist())

print("\nTarget distribution:")
print(sylhet["next_day_risk"].value_counts())

print("\nTarget percentage:")
print(sylhet["next_day_risk"].value_counts(normalize=True) * 100)

print("\nFirst 10 rows:")
print(
    sylhet[
        [
            "date",
            "demand_mw",
            "previous_load_shed",
            "load_shed_3day_avg",
            "previous_demand",
            "demand_change",
            "month",
            "is_weekend",
            "next_day_load_shed",
            "next_day_risk",
        ]
    ].head(10)
)


# ==========================================
# 10. SAVE
# ==========================================

output_file = r"d:\PowerGrid_BD\sylhet_prediction_dataset.csv"

sylhet.to_csv(output_file, index=False)

print("\nSaved to:")
print(output_file)
