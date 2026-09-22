import pandas as pd
import os

print("\n========== BPDB DATA CHECK ==========\n")

files = [
    "bpdb_area_wise_2023.csv",
    "bpdb_area_wise_2024.csv",
    "bpdb_area_wise_2025.csv",
]

for file in files:

    print("======================================")
    print("File:", file)

    df = pd.read_csv(file)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("Date range:")
    print("  Start:", df["date"].min())
    print("  End  :", df["date"].max())

    print("Zones:")
    print(df["zone"].unique())

    print("Missing values:")
    print(df.isnull().sum().sum())

    print()
