import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from collision_utils import *

# Get user input
start_year = int(input("Enter training start year: "))
end_year = int(input("Enter training end year: "))
test_year = int(input("Enter test year: "))


# Load and combine training years
all_dfs = []
for year in range(start_year, end_year + 1):
    df_year = pd.read_csv(f"la_collisions_{year}.csv")
    all_dfs.append(df_year)
    print(f"Loaded {year} data!")

df_train = pd.concat(all_dfs)
print(f"Combined training data shape: {df_train.shape}")

# Engineer features on training data
df_train = engineer_features(df_train)

# Select features
features = [col for col in df_train.columns if col.startswith('area_') or
            col.startswith('sex_') or
            col.startswith('descent_') or
            col.startswith('is_') or
            col in ['vict_age', 'hour_of_day', 'latitude', 'longitude']]

# Load and engineer test data
df_test = pd.read_csv(f"la_collisions_{test_year}.csv")
df_test = engineer_features(df_test)

# Align test columns to training features
missing_cols = [col for col in features if col not in df_test.columns]
if missing_cols:
    missing_df = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
    df_test = pd.concat([df_test, missing_df], axis=1)

# Create training arrays
X_train = df_train[features].values
y_train = df_train['injury'].values

# Scale training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create test arrays
X_test = df_test[features].values
y_test = df_test['injury'].values

# Scale test data using TRAINING scaler
X_test_scaled = scaler.transform(X_test)

# Train the model
print("Training model...")
w, b, costs = gradient_descent(X_train_scaled, y_train, alpha=0.1, iterations=1000)

# Make predictions on TEST data
y_pred_prob = sigmoid(X_test_scaled @ w + b)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy ({test_year}): {accuracy:.4f}")

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature weights - TOP 10 most important
feature_weights = list(zip(features, w))
feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\nTop 10 most important features:")
for feat, weight in feature_weights[:10]:
    print(f"  {feat}: {weight:.4f}")