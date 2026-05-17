import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import pickle
from collision_utils import *

# Load and combine all yearly CSVs from 2010 to 2024
all_dfs = []
for year in range(2010, 2025):  # 2010-2024 only, never 2025
    csv_path = f'la_collisions_{year}.csv'
    if os.path.exists(csv_path):
        all_dfs.append(pd.read_csv(csv_path))
        print(f"Loaded {year}")
    else:
        print(f"Warning: {csv_path} not found, skipping")
df = pd.concat(all_dfs)

df.to_csv('la_collisions_2010_2024.csv', index=False)
print(f"Combined shape (raw): {df.shape}")

df = engineer_features(df)

# Select final features for model
features = [col for col in df.columns if col.startswith('area_') or
            col.startswith('sex_') or
            col.startswith('descent_') or
            col.startswith('is_') or
            col.startswith('grid_') or
            col in ['vict_age', 'hour_of_day', 'latitude', 'longitude']]

# Save feature list for test alignment
with open('features.json', 'w') as f:
    json.dump(features, f)
print(f"Saved {len(features)} features!")

# Create X and y
X = df[features].values
y = df['injury'].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"X_scaled shape: {X_scaled.shape}")
print(f"X_scaled mean (should be ~0): {X_scaled.mean(axis=0)[:5]}")
print(f"X_scaled std (should be ~1): {X_scaled.std(axis=0)[:5]}")

# Save everything
np.save('X_scaled.npy', X_scaled)
np.save('y.npy', y)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Processed 2010-2024 data saved!")
