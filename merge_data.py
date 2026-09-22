import pandas as pd

# =========================================================
# 1. LOAD THREE YEARS
# =========================================================

files = [
    r"d:\PowerGrid_BD\bpdb_area_wise_2023.csv",
    r"d:\PowerGrid_BD\bpdb_area_wise_2024.csv",
    r"d:\PowerGrid_BD\bpdb_area_wise_2025.csv",
]

dfs = []

for file in files:

    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # Convert numeric columns
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

    df["load_shed_mw"] = pd.to_numeric(df["load_shed_mw"], errors="coerce")

    dfs.append(df)


# =========================================================
# 2. MERGE
# =========================================================

df = pd.concat(dfs, ignore_index=True)

df = df.sort_values(["zone", "date"]).reset_index(drop=True)


# =========================================================
# 3. BASIC CHECK
# =========================================================

print("\n========== MERGED DATA ==========")

print("Total records:", len(df))

print("\nRecords by year:")

print(df.groupby(df["date"].dt.year).size())


print("\nRecords by zone:")

print(df["zone"].value_counts())


# =========================================================
# 4. CHECK DUPLICATES
# =========================================================

duplicates = df.duplicated(subset=["date", "zone"]).sum()

print("\nDuplicate date-zone records:", duplicates)


# =========================================================
# 5. CHECK MISSING DATES FOR EACH YEAR
# =========================================================

print("\n========== MISSING DATES ==========")

for year in [2023, 2024, 2025]:

    expected_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

    actual_dates = pd.DatetimeIndex(df[df["date"].dt.year == year]["date"].unique())

    missing_dates = expected_dates.difference(actual_dates)

    print(f"\n{year}:")
    print("Missing dates:", len(missing_dates))

    for date in missing_dates:
        print(" ", date.strftime("%d-%m-%Y"))


# =========================================================
# 6. SAVE MERGED DATA
# =========================================================

output_file = r"d:\PowerGrid_BD\bpdb_area_wise_2023_2025.csv"

df.to_csv(output_file, index=False)


print("\n================================")
print("Merge completed!")
print("Total records:", len(df))
print("Saved as:")
print(output_file)
print("================================")
