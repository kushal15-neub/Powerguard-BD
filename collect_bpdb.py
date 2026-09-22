import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv
import urllib3

# =========================================================
# 1. SETTINGS
# =========================================================

YEAR = 2025

BASE_URL = "https://misc.bpdb.gov.bd/area-wise-demand"

# Disable SSL warning because BPDB certificate may cause issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =========================================================
# 2. DATE RANGE
# =========================================================

start_date = datetime(YEAR, 1, 1)
end_date = datetime(YEAR, 12, 31)


# =========================================================
# 3. DATA STORAGE
# =========================================================

all_data = []

current_date = start_date


# =========================================================
# 4. COLLECT DATA
# =========================================================

while current_date <= end_date:

    date_string = current_date.strftime("%d-%m-%Y")

    url = f"{BASE_URL}?date={date_string}"

    print(f"Collecting: {date_string}")

    try:

        response = requests.get(url, verify=False, timeout=20)

        # ---------------------------------------------
        # Check HTTP status
        # ---------------------------------------------

        if response.status_code != 200:

            print(f"  Failed: HTTP {response.status_code}")

            current_date += timedelta(days=1)
            continue

        # ---------------------------------------------
        # Parse HTML
        # ---------------------------------------------

        soup = BeautifulSoup(response.text, "html.parser")

        tables = soup.find_all("table")

        if len(tables) < 2:

            print("  No data table found")

            current_date += timedelta(days=1)
            continue

        # BPDB Area Wise Demand table
        table = tables[1]

        rows = table.find_all("tr")

        found_data = False

        # ---------------------------------------------
        # Read rows
        # ---------------------------------------------

        for row in rows[1:]:

            cells = row.find_all(["td", "th"])

            if len(cells) < 4:
                continue

            sl = cells[0].get_text(strip=True)

            zone = cells[1].get_text(strip=True)

            demand = cells[2].get_text(strip=True)

            load_shed = cells[3].get_text(strip=True)

            # -----------------------------------------
            # Ignore Total row
            # -----------------------------------------

            if zone.lower() == "total":
                continue

            # -----------------------------------------
            # Make sure row is valid
            # -----------------------------------------

            if not sl.isdigit():
                continue

            # -----------------------------------------
            # Store data
            # -----------------------------------------

            all_data.append(
                {
                    "date": date_string,
                    "zone": zone,
                    "demand_mw": demand,
                    "load_shed_mw": load_shed,
                }
            )

            found_data = True

        # ---------------------------------------------
        # Status
        # ---------------------------------------------

        if found_data:

            print("  Data collected")

        else:

            print("  No records found")

    except requests.exceptions.Timeout:

        print("  Error: Request timed out")

    except requests.exceptions.RequestException as e:

        print(f"  Request error: {e}")

    except Exception as e:

        print(f"  Error: {e}")

    # Next day
    current_date += timedelta(days=1)


# =========================================================
# 5. SAVE CSV
# =========================================================

output_file = f"bpdb_area_wise_{YEAR}.csv"


with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file, fieldnames=["date", "zone", "demand_mw", "load_shed_mw"]
    )

    writer.writeheader()

    writer.writerows(all_data)


# =========================================================
# 6. FINAL SUMMARY
# =========================================================

print("\n================================")
print("Data collection completed!")
print("Year:", YEAR)
print("Total records:", len(all_data))
print("Saved as:", output_file)
print("================================")
