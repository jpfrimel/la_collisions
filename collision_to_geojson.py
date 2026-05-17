import pandas as pd
import json
import ast
import os

def extract_lat(x):
    try:
        d = ast.literal_eval(x)
        return float(d['latitude'])
    except:
        return None

def extract_lon(x):
    try:
        d = ast.literal_eval(x)
        return float(d['longitude'])
    except:
        return None

def get_injury(mocode_string):
    if not mocode_string or pd.isna(mocode_string):
        return None
    if any(code in str(mocode_string) for code in ['3024', '3025', '3026', '3027']):
        return True
    if '3028' in str(mocode_string):
        return False
    return None

def get_collision_type(mocode_string):
    if not mocode_string or pd.isna(mocode_string):
        return 'Unknown'
    m = str(mocode_string)
    if '3003' in m: return 'Pedestrian'
    if '3008' in m or '3016' in m: return 'Bicycle'
    if '3009' in m or '3013' in m: return 'Motorcycle'
    if '3038' in m or '3039' in m: return 'DUI'
    if '3040' in m: return 'Street Racing'
    if '3029' in m or '3030' in m: return 'Hit & Run'
    if '3004' in m: return 'Vehicle vs Vehicle'
    return 'Other'

# Output directory — points to the React app's public/data folder.
# Update this path to match your local directory structure before running.
OUTPUT_DIR = '../la-collision-viz/public/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

years = range(2010, 2025)

for year in years:
    csv_path = f'la_collisions_{year}.csv'
    
    if not os.path.exists(csv_path):
        print(f"⚠ Skipping {year} — CSV not found")
        continue
    
    print(f"Processing {year}...")
    df = pd.read_csv(csv_path)
    
    # Extract coordinates
    df['lat'] = df['location_1'].apply(extract_lat)
    df['lon'] = df['location_1'].apply(extract_lon)
    df = df.dropna(subset=['lat', 'lon'])
    
    # Add derived fields
    df['injury']         = df['mocodes'].apply(get_injury)
    df['collision_type'] = df['mocodes'].apply(get_collision_type)
    df['date']           = df['date_occ'].str[:10]
    df['area']           = df['area_name'].fillna('Unknown')
    
    # Build GeoJSON
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['lon'], row['lat']]
            },
            "properties": {
                "id":    str(row['dr_no']),
                "date":  row['date'],
                "area":  row['area'],
                "type":  row['collision_type'],
                "injury": row['injury'],
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save
    output_path = f"{OUTPUT_DIR}/collisions_{year}.geojson"
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    print(f"✅ {year}: {len(features)} records → collisions_{year}.geojson")

print("\nAll done! GeoJSON files saved to la-collision-viz/public/data/")