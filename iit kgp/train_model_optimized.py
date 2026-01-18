
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def add_features(df, rolling_win_size=10, sensor_cols=[]):
    """
    Adds rolling mean and rolling std features for sensor columns.
    """
    sensor_cols_to_use = [c for c in sensor_cols if c in df.columns]
    
    # Sort by unit and time to ensure rolling window is correct
    df = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    # Calculate rolling stats grouped by unit_number
    # rolling() requires index to be set or careful groupby usage. 
    # Best to groupby, apply rolling, then join back or assign directly if shape matches.
    
    rolling_means = df.groupby('unit_number')[sensor_cols_to_use].rolling(window=rolling_win_size, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_stds = df.groupby('unit_number')[sensor_cols_to_use].rolling(window=rolling_win_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    
    # Rename columns
    rolling_means.columns = [f'{c}_mean' for c in rolling_means.columns]
    rolling_stds.columns = [f'{c}_std' for c in rolling_stds.columns]
    
    # Join back using index
    # Note: reset_index(drop=True) in groupby might misalign if original df index wasn't range.
    # Safer to just concat if we are sure order is preserved, or merge on index.
    # Since we sorted df, let's reset index to make it clean.
    df = df.reset_index(drop=True)
    
    # The rolling results should have same index as df because we didn't drop rows (min_periods=1)
    # However, groupby might change order if we did not sort df first? We did sort.
    # Groupby preserves order of groups, and within groups order is preserved.
    # But let's verify indices.
    
    df = pd.concat([df, rolling_means, rolling_stds], axis=1)
    
    return df

def train_and_evaluate_optimized():
    # 1. Load Data
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # 2. Data Cleaning
    cols_to_drop = ['setting_3', 's18', 's19'] # Constant columns
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    
    # Identify Sensor Columns (excluding settings and index cols)
    sensor_cols = [c for c in train_df.columns if c.startswith('s')]

    # 3. Feature Engineering
    
    # A. Add Rolling Features
    print("Generating rolling features...")
    train_df = add_features(train_df, rolling_win_size=15, sensor_cols=sensor_cols)
    test_df = add_features(test_df, rolling_win_size=15, sensor_cols=sensor_cols)

    # B. Calculate RUL for Training
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    # C. Clip RUL (Piecewise Linear)
    # RUL is often capped at a threshold (e.g., 125) because early life behavior is constant
    RUL_CLIP = 125
    train_df['RUL_clipped'] = train_df['RUL'].clip(upper=RUL_CLIP)

    # 4. Prepare Train/Test Data
    features = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL', 'RUL_clipped']]
    X_train = train_df[features]
    y_train = train_df['RUL_clipped'] # Train on clipped RUL

    # 5. Model Training
    print("Training Random Forest Regressor (Optimized)...")
    # Random Forest is often better at handling non-linearities than GBR without tuning
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6. Predict on Test Set
    # We need the LAST record for each unit in test_df
    last_cycle_test = test_df.groupby('unit_number').last().reset_index()
    X_test = last_cycle_test[features]
    
    y_pred = model.predict(X_test)
    
    # 7. Evaluate
    y_true = true_rul['RUL'] # Standard RUL
    # Note: Effectively we are comparing Predicted RUL vs True RUL. 
    # If the True RUL is > 125, our model will predict ~125 max. This might hurt R2 if many true RULs are > 125.
    # However, usually for maintenance, we care about the end of life.
    # Let's see the score.
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("-" * 30)
    print(f"Optimized R^2 Score: {r2:.4f}")
    print(f"Optimized RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # Save the model features for reference if needed
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print("Top 5 Features:")
    print(imp.sort_values(by='importance', ascending=False).head(5))

    if r2 >= 0.95:
        print("\nSUCCESS: R^2 Score achieved >= 0.95")
    else:
        print("\nNote: R^2 is still under 0.95. This is common for this dataset with simple Regression.")
        print("FD001 best known scores are usually around RMSE 20-23, R2 ~0.7-0.8 with standard ML.")
        print("To get 0.95 might require LSTM/Deep Learning or cheating (looking at test labels).")
        print("However, I will try to maximize it.")

if __name__ == "__main__":
    train_and_evaluate_optimized()
