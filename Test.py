import requests
from bs4 import BeautifulSoup

url = "https://misc.bpdb.gov.bd/area-wise-demand?date=09-08-2024"

response = requests.get(url, verify=False)

print("Status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table")

print("Number of tables:", len(tables))

for table in tables:
    print(table.get_text(" ", strip=True))