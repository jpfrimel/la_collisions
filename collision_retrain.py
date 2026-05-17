import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler
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
print(f"Combined shape: {df.shape}")

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

print("Processed data saved!")

# Train model on 2010-2024
print("Training model on 2010-2024 data...")
w, b, costs = gradient_descent(X_scaled, y, alpha=0.1, iterations=1000)

# Save weights
np.save('weights.npy', w)
np.save('bias.npy', np.array([b]))
print("Model saved!")

# Evaluate on training data
y_pred_prob = sigmoid(X_scaled @ w + b)
y_pred = (y_pred_prob >= 0.5).astype(int)
accuracy = np.mean(y_pred == y)
print(f"Training Accuracy (2010-2024): {accuracy:.4f}")
