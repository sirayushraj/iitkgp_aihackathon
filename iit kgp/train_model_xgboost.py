
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def add_features(df, rolling_win_size=10, sensor_cols=[]):
    df = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    rolling_means = df.groupby('unit_number')[sensor_cols].rolling(window=rolling_win_size, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_stds = df.groupby('unit_number')[sensor_cols].rolling(window=rolling_win_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    
    rolling_means.columns = [f'{c}_mean' for c in rolling_means.columns]
    rolling_stds.columns = [f'{c}_std' for c in rolling_stds.columns]
    
    df = df.reset_index(drop=True)
    df = pd.concat([df, rolling_means, rolling_stds], axis=1)
    
    return df

def train_xgboost():
    # Load
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Clean
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    sensor_cols = [c for c in train_df.columns if c.startswith('s')]

    # Features
    print("Generating rolling features...")
    train_df = add_features(train_df, rolling_win_size=15, sensor_cols=sensor_cols)
    test_df = add_features(test_df, rolling_win_size=15, sensor_cols=sensor_cols)

    # RUL
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    # Clip
    RUL_CLIP = 125
    train_df['RUL_clipped'] = train_df['RUL'].clip(upper=RUL_CLIP)

    # Train
    features = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL', 'RUL_clipped']]
    X_train = train_df[features]
    y_train = train_df['RUL_clipped']

    print("Training XGBoost Regressor...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    last_cycle_test = test_df.groupby('unit_number').last().reset_index()
    X_test = last_cycle_test[features]
    y_pred = model.predict(X_test)
    
    # Eval
    y_true = true_rul['RUL']
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("-" * 30)
    print(f"XGBoost R^2 Score: {r2:.4f}")
    print(f"XGBoost RMSE: {rmse:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    train_xgboost()
