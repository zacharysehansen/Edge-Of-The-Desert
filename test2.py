import requests

# USDM provides county-level drought area percentages
# via their statistics API
BASE_URL = "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI"

# Query each county individually
county_fips = ["04019", "04021", "04023", "04003", "04013", "04011", "04027", "04007"]

frames = []
for fips in county_fips:
    params = {
        "aoi": fips,
        "startdate": "1/1/2000",
        "enddate": "12/31/2023",
        "statisticsType": "1",  # DSCI
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    print(f"{fips}: status={r.status_code}")
    if r.status_code == 200:  # noqa: PLR2004
        print(r.text[:200])
        frames.append(r.json())
