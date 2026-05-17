# collision_utils.py
# Shared cleaning and feature engineering functions
import numpy as np
import pandas as pd
import ast



def get_injury_label(mocode_string):
    if pd.isna(mocode_string):
        return None
    if '3024' in mocode_string or '3025' in mocode_string or '3026' in mocode_string or '3027' in mocode_string:
        return 1
    if '3028' in mocode_string:
        return 0
    return None

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

def engineer_features(df):
    # Injury label
    df['injury'] = df['mocodes'].apply(get_injury_label)
    df = df[df['injury'].notna()].copy()

    # Collision type features
    df['is_pedestrian']       = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3003' in x else 0))
    df['is_bike']             = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3008' in x or '3016' in x else 0))
    df['is_motorcycle']       = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3009' in x or '3013' in x else 0))
    df['is_fixed_object']     = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3011' in x else 0))
    df['is_parked_vehicle']   = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3006' in x else 0))

    # Circumstances features
    df['is_hit_and_run']  = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3029' in x or '3030' in x else 0))
    df['is_dui_alcohol']  = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3038' in x else 0))
    df['is_dui_drugs']    = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3039' in x else 0))
    df['is_intersection'] = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3036' in x else 0))
    df['is_unlicensed']   = df['mocodes'].apply(lambda x: 0 if pd.isna(x) else (1 if '3602' in x else 0))

    # Location
    df['latitude']  = df['location_1'].apply(extract_lat)
    df['longitude'] = df['location_1'].apply(extract_lon)

    # Time of day
    df['hour_of_day'] = df['time_occ'] // 100

    # Victim sex cleanup
    df['vict_sex'] = df['vict_sex'].replace({'H': 'X', 'N': 'X'})

    # Time features
    df = get_time_features(df)

    # Age features
    df = get_age_features(df)

    # Location features
    df = get_location_features(df)

    # Corridor features
    df = get_corridor_features(df)

    # MO code features
    df = get_mocode_features(df)

    # One hot encoding
    df = pd.get_dummies(df, columns=['area_name', 'vict_sex', 'vict_descent', 'lat_lon_grid'], prefix=['area', 'sex', 'descent', 'grid'])

    # Drop bad descent column if exists
    if 'descent_-' in df.columns:
        df = df.drop(columns=['descent_-'])

    return df

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost
def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    cost = -1/m * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    return cost

#Gradient
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    dw = 1/m * np.dot(X.T, (y_hat - y))
    db = 1/m * np.sum(y_hat - y)
    return dw, db

#Gradient Descent
def gradient_descent(X, y, alpha, iterations, verbose=True):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    costs = []
    for i in range(iterations):
        dw, db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            costs.append(cost)
            if verbose:
                print(f"Iteration {i}: Cost = {cost:.4f}")
    return w, b, costs


def get_time_features(df):
    # Parse date and time
    df['date_occ'] = pd.to_datetime(df['date_occ'])
    df['hour'] = df['time_occ'].fillna(0).astype(int) // 100

    # Basic time flags
    df['is_rush_hour']  = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                           (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
    df['day_of_week']   = df['date_occ'].dt.dayofweek
    df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
    return df


def get_age_features(df):
    df['vict_age'] = df['vict_age'].fillna(df['vict_age'].median())
    df['age_16_25']    = ((df['vict_age'] >= 16) & (df['vict_age'] <= 25)).astype(int)
    df['age_26_40']    = ((df['vict_age'] >= 26) & (df['vict_age'] <= 40)).astype(int)
    df['age_41_64']    = ((df['vict_age'] >= 41) & (df['vict_age'] <= 64)).astype(int)
    df['age_65_plus']  = (df['vict_age'] >= 65).astype(int)
    return df


def get_location_features(df):
    lat = df['latitude']
    lon = df['longitude']

    # Geo-grid cells: lat/lon rounded to 0.05° buckets, one-hot encoded as grid_<lat>_<lon>
    lat_grid = ((lat / 0.05).round() * 0.05).round(2)
    lon_grid = ((lon / 0.05).round() * 0.05).round(2)
    df['lat_lon_grid'] = lat_grid.astype(str) + '_' + lon_grid.astype(str)

    return df


def get_corridor_features(df):
    df['is_figueroa_st']    = df['location'].apply(lambda x: 1 if pd.notna(x) and 'FIGUEROA'  in str(x).upper() else 0)
    df['is_western_ave']    = df['location'].apply(lambda x: 1 if pd.notna(x) and 'WESTERN'   in str(x).upper() else 0)
    df['is_vermont_ave']    = df['location'].apply(lambda x: 1 if pd.notna(x) and 'VERMONT'   in str(x).upper() else 0)
    df['is_sepulveda_blvd'] = df['location'].apply(lambda x: 1 if pd.notna(x) and 'SEPULVEDA' in str(x).upper() else 0)
    df['is_broadway']       = df['location'].apply(lambda x: 1 if pd.notna(x) and 'BROADWAY'  in str(x).upper() else 0)
    df['is_sunset_blvd']    = df['location'].apply(lambda x: 1 if pd.notna(x) and 'SUNSET'    in str(x).upper() else 0)
    df['is_olympic_blvd']   = df['location'].apply(lambda x: 1 if pd.notna(x) and 'OLYMPIC'   in str(x).upper() else 0)
    df['is_victory_blvd']   = df['location'].apply(lambda x: 1 if pd.notna(x) and 'VICTORY'   in str(x).upper() else 0)
    df['is_van_nuys_blvd']  = df['location'].apply(lambda x: 1 if pd.notna(x) and 'VAN NUYS'  in str(x).upper() else 0)
    df['is_venice_blvd']    = df['location'].apply(lambda x: 1 if pd.notna(x) and 'VENICE'    in str(x).upper() else 0)
    df['is_pico_blvd']      = df['location'].apply(lambda x: 1 if pd.notna(x) and 'PICO'      in str(x).upper() else 0)

    return df


def get_mocode_features(df):
    df['is_multi_vehicle'] = df['mocodes'].apply(
        lambda x: 0 if pd.isna(x) else (1 if '3004' in x or '3005' in x else 0)
    )
    return df


