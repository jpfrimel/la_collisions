import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from collision_utils import *

results = []

for start_year in range(2010, 2024):
    for end_year in range(start_year + 1, 2024):
        test_year = end_year + 1
        
        try:
            print(f"\nTraining {start_year}-{end_year}, testing {test_year}...")
            
            # Load and combine training years
            all_dfs = []
            for year in range(start_year, end_year + 1):
                df_year = pd.read_csv(f"la_collisions_{year}.csv")
                all_dfs.append(df_year)
            
            df_train = pd.concat(all_dfs)
            df_train = engineer_features(df_train)
            
            # Select features
            features = [col for col in df_train.columns if col.startswith('area_') or
                        col.startswith('sex_') or
                        col.startswith('descent_') or
                        col.startswith('is_') or
                        col.startswith('grid_') or
                        col in ['vict_age', 'hour_of_day', 'latitude', 'longitude']]
            
            # Load and engineer test data
            df_test = pd.read_csv(f"la_collisions_{test_year}.csv")
            df_test = engineer_features(df_test)
            
            # Align test columns
            missing_cols = [col for col in features if col not in df_test.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
                df_test = pd.concat([df_test, missing_df], axis=1)
            
            # Create arrays
            X_train = df_train[features].values
            y_train = df_train['injury'].values
            X_test = df_test[features].values
            y_test = df_test['injury'].values
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            w, b, costs = gradient_descent(X_train_scaled, y_train, alpha=0.1, iterations=1000, verbose=False)
            
            # Predict
            y_pred_prob = sigmoid(X_test_scaled @ w + b)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            accuracy = float(np.mean(y_pred == y_test))
            
            # Feature weights
            feature_weights = list(zip(features, w.tolist()))
            feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result = {
                'train_start': start_year,
                'train_end': end_year,
                'test_year': test_year,
                'accuracy': accuracy,
                'final_cost': float(costs[-1]),
                'final_bias': float(b),
                'top_features': feature_weights
            }
            results.append(result)
            print(f"✅ Accuracy: {accuracy:.4f}")
            
            # Save after every window in case of crash
            with open('sliding_window_results_v3.json', 'w') as f:
                json.dump(results, f)
                
        except Exception as e:
            print(f"❌ Error on {start_year}-{end_year}: {e}")
            continue

print(f"\nDone! {len(results)} windows computed!")