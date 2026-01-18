
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def add_advanced_features(df, sensor_cols):
    df = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    # 1. Rolling Features (Multiple Windows)
    windows = [5, 10, 20]
    for w in windows:
        # Mean
        rolling_mean = df.groupby('unit_number')[sensor_cols].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f'{c}_mean_{w}' for c in sensor_cols]
        
        # Std
        rolling_std = df.groupby('unit_number')[sensor_cols].rolling(window=w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        rolling_std.columns = [f'{c}_std_{w}' for c in sensor_cols]
        
        df = df.reset_index(drop=True)
        df = pd.concat([df, rolling_mean, rolling_std], axis=1)
        
    # 2. Expanding Mean (Cumulative behavior)
    expanding_mean = df.groupby('unit_number')[sensor_cols].expanding().mean().reset_index(level=0, drop=True)
    expanding_mean.columns = [f'{c}_expanding_mean' for c in sensor_cols]
    df = pd.concat([df, expanding_mean], axis=1)

    return df

def train_stacking():
    print("Loading data...")
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Clean
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    sensor_cols = [c for c in train_df.columns if c.startswith('s')]

    # Feature Engineering
    print("Generating advanced features (Rolling 5, 10, 20 + Expanding)...")
    train_df = add_advanced_features(train_df, sensor_cols)
    test_df = add_advanced_features(test_df, sensor_cols)

    # RUL calculation
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    # Clip RUL
    RUL_CLIP = 125
    train_df['RUL_clipped'] = train_df['RUL'].clip(upper=RUL_CLIP)

    # Prepare Data
    features = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL', 'RUL_clipped']]
    X_train = train_df[features]
    y_train = train_df['RUL_clipped']

    # Predictor for Test
    last_cycle_test = test_df.groupby('unit_number').last().reset_index()
    X_test = last_cycle_test[features]
    y_true = true_rul['RUL']

    # Stacking Setup
    print("Initializing Base Models...")
    
    # 1. XGBoost (Optimized from before)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    
    # 2. Random Forest (Robust tree based)
    rf = RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=10, n_jobs=-1, random_state=42)
    
    # 3. SVR (Different inductive bias - smooth). Needs scaling.
    svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    
    estimators = [('xgb', xgb), ('rf', rf), ('svr', svr)]
    
    print("Training Stacking Regressor (Meta-Learner: RidgeCV)...")
    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        n_jobs=-1,
        passthrough=False 
    )
    
    stacker.fit(X_train, y_train)
    
    print("Predicting...")
    y_pred = stacker.predict(X_test)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("-" * 30)
    print(f"Stacking R^2 Score: {r2:.4f}")
    print(f"Stacking RMSE: {rmse:.4f}")
    print("-" * 30)

    # Check accuracy components
    # We can inspect which model contributed most if we looked at final_estimator coefficients (not easy in one line)
    
    if r2 >= 0.90:
        print("GOAL MET: R^2 >= 0.90")
    else:
        print("Goal Not Met. Feature limitations on Test Set likely.")

if __name__ == "__main__":
    train_stacking()
