import requests
import pandas as pd

url = "https://data.lacity.org/resource/d5tf-ez2w.json"

for year in range(2010,2025):
    print(f"Pulling {year} data...")
    all_dfs = []
    offset = 0
    chunk_size = 50000

    while True:
        params = {
            "$limit": chunk_size,
            "$offset": offset,
            "$where": f"date_occ > '{year-1}-12-31' and date_occ <= '{year}-12-31'"
        }
        response = requests.get(url, params=params)
        chunk = pd.DataFrame(response.json())

        if len(chunk) == 0:
            break

        all_dfs.append(chunk)
        offset += chunk_size
        print(f"Pulled {offset} rows for {year} so far...")

    df = pd.concat(all_dfs)
    df.to_csv(f"la_collisions_{year}.csv", index=False)
    print(f"✅ {year} saved! ({len(df)} rows)")