import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"d:\PowerGrid_BD\bpdb_area_wise_2024.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# ==============================
# 1. LOAD SHEDDING BY ZONE
# ==============================

zone_loadshed = df.groupby("zone")["load_shed_mw"].sum().sort_values(ascending=False)

print("\n========== TOTAL LOAD SHEDDING BY ZONE ==========")
print(zone_loadshed)

plt.figure(figsize=(10, 6))

zone_loadshed.plot(kind="bar")

plt.title("Total Load Shedding by Zone - 2024")
plt.xlabel("Zone")
plt.ylabel("Total Load Shedding (MW)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==============================
# 2. NUMBER OF LOAD-SHEDDING DAYS
# ==============================

loadshed_days = (
    df[df["load_shed_mw"] > 0].groupby("zone").size().sort_values(ascending=False)
)

print("\n========== LOAD-SHEDDING DAYS BY ZONE ==========")
print(loadshed_days)

plt.figure(figsize=(10, 6))

loadshed_days.plot(kind="bar")

plt.title("Number of Load-Shedding Days by Zone - 2024")
plt.xlabel("Zone")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==============================
# 3. SYLHET LOAD SHEDDING OVER TIME
# ==============================

sylhet = df[df["zone"] == "Sylhet"].copy()

sylhet = sylhet.sort_values("date")

plt.figure(figsize=(14, 6))

plt.plot(sylhet["date"], sylhet["load_shed_mw"])

plt.title("Sylhet Load Shedding During 2024")
plt.xlabel("Date")
plt.ylabel("Load Shedding (MW)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ==============================
# 4. SYLHET DEMAND VS LOAD SHEDDING
# ==============================

plt.figure(figsize=(10, 6))

plt.scatter(sylhet["demand_mw"], sylhet["load_shed_mw"])

plt.title("Sylhet Demand vs Load Shedding - 2024")
plt.xlabel("Demand (MW)")
plt.ylabel("Load Shedding (MW)")

plt.tight_layout()

plt.show()
