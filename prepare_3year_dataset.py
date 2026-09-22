import pandas as pd

# =========================================================
# 1. LOAD MERGED DATA
# =========================================================

file = r"d:\PowerGrid_BD\bpdb_area_wise_2023_2025.csv"

df = pd.read_csv(file)

df["date"] = pd.to_datetime(df["date"])


# =========================================================
# 2. SELECT SYLHET
# =========================================================

sylhet = df[df["zone"] == "Sylhet"].copy()

sylhet = sylhet.sort_values("date").reset_index(drop=True)


# =========================================================
# 3. CONVERT NUMERIC COLUMNS
# =========================================================

sylhet["demand_mw"] = pd.to_numeric(sylhet["demand_mw"], errors="coerce")

sylhet["load_shed_mw"] = pd.to_numeric(sylhet["load_shed_mw"], errors="coerce")


# =========================================================
# 4. CHECK BASIC INFORMATION
# =========================================================

print("\n========== SYLHET DATA ==========")

print("Total records:", len(sylhet))

print("Start date:", sylhet["date"].min())

print("End date:", sylhet["date"].max())


# =========================================================
# 5. CREATE COMPLETE DAILY CALENDAR
# =========================================================

full_dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")

sylhet = sylhet.set_index("date").reindex(full_dates)

sylhet.index.name = "date"

sylhet = sylhet.reset_index()


# Restore zone
sylhet["zone"] = "Sylhet"


# =========================================================
# 6. CHECK MISSING SYLHET DATES
# =========================================================

missing_dates = sylhet[sylhet["demand_mw"].isna()]["date"]

print("\n========== MISSING SYLHET DATES ==========")

print("Missing dates:", len(missing_dates))

for date in missing_dates:
    print(date.strftime("%d-%m-%Y"))


# =========================================================
# 7. CREATE TIME FEATURES
# =========================================================

sylhet["day_of_week"] = sylhet["date"].dt.dayofweek

sylhet["month"] = sylhet["date"].dt.month

sylhet["day_of_month"] = sylhet["date"].dt.day

sylhet["is_weekend"] = (sylhet["day_of_week"] >= 5).astype(int)


# =========================================================
# 8. CREATE VALID-DAY FLAG
# =========================================================

sylhet["valid_day"] = sylhet["demand_mw"].notna() & sylhet["load_shed_mw"].notna()


# =========================================================
# 9. PREVIOUS-DAY FEATURES
# =========================================================

sylhet["previous_load_shed"] = sylhet["load_shed_mw"].shift(1)

sylhet["previous_demand"] = sylhet["demand_mw"].shift(1)


# =========================================================
# 10. ROLLING FEATURES
# =========================================================

sylhet["load_shed_3day_avg"] = (
    sylhet["load_shed_mw"].shift(1).rolling(window=3, min_periods=3).mean()
)

sylhet["load_shed_7day_avg"] = (
    sylhet["load_shed_mw"].shift(1).rolling(window=7, min_periods=7).mean()
)

sylhet["demand_3day_avg"] = (
    sylhet["demand_mw"].shift(1).rolling(window=3, min_periods=3).mean()
)


# =========================================================
# 11. DEMAND CHANGE
# =========================================================

sylhet["demand_change"] = sylhet["demand_mw"] - sylhet["previous_demand"]


# =========================================================
# 12. NEXT-DAY TARGET
# =========================================================

sylhet["next_day_load_shed"] = sylhet["load_shed_mw"].shift(-1)

sylhet["next_date"] = sylhet["date"].shift(-1)


# =========================================================
# 13. MAKE SURE NEXT DAY IS ACTUALLY NEXT CALENDAR DAY
# =========================================================

sylhet["date_difference"] = (sylhet["next_date"] - sylhet["date"]).dt.days


# =========================================================
# 14. BINARY TARGET
# =========================================================

sylhet["next_day_risk"] = (sylhet["next_day_load_shed"] > 0).astype(int)


# =========================================================
# 15. REMOVE INVALID ROWS
# =========================================================

# Current day must exist
sylhet = sylhet[sylhet["valid_day"] == True].copy()


# Next calendar day must exist
sylhet = sylhet[sylhet["date_difference"] == 1].copy()


# Remove rows where required historical
# features are unavailable
required_features = [
    "previous_load_shed",
    "load_shed_3day_avg",
    "load_shed_7day_avg",
    "previous_demand",
    "demand_3day_avg",
    "demand_change",
    "next_day_load_shed",
]

sylhet = sylhet.dropna(subset=required_features)


# =========================================================
# 16. SORT
# =========================================================

sylhet = sylhet.sort_values("date").reset_index(drop=True)


# =========================================================
# 17. FINAL SUMMARY
# =========================================================

print("\n========================================")
print("FINAL PREDICTION DATASET")
print("========================================")

print("Total rows:", len(sylhet))

print("Start:", sylhet["date"].min())

print("End:", sylhet["date"].max())


print("\nTarget distribution:")

print(sylhet["next_day_risk"].value_counts())


print("\nTarget percentage:")

print(sylhet["next_day_risk"].value_counts(normalize=True) * 100)


# =========================================================
# 18. YEAR-WISE DISTRIBUTION
# =========================================================

sylhet["year"] = sylhet["date"].dt.year

print("\n========== YEAR-WISE ROWS ==========")

print(sylhet["year"].value_counts().sort_index())


print("\n========== YEAR-WISE TARGET ==========")

print(pd.crosstab(sylhet["year"], sylhet["next_day_risk"]))


# =========================================================
# 19. SHOW SAMPLE
# =========================================================

print("\n========== SAMPLE ==========")

print(
    sylhet[
        [
            "date",
            "demand_mw",
            "previous_load_shed",
            "load_shed_3day_avg",
            "load_shed_7day_avg",
            "previous_demand",
            "demand_3day_avg",
            "demand_change",
            "is_weekend",
            "next_day_load_shed",
            "next_day_risk",
        ]
    ].head(15)
)


# =========================================================
# 20. SAVE
# =========================================================

output_file = r"d:\PowerGrid_BD\sylhet_2023_2025_prediction_dataset.csv"

sylhet.to_csv(output_file, index=False)


print("\n========================================")
print("Dataset saved successfully!")
print(output_file)
print("========================================")
