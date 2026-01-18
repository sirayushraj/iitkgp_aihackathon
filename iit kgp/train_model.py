
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate():
    # 1. Load Data
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # 2. Data Cleaning
    # Drop constant columns found in analysis
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped constant columns: {cols_to_drop}")

    # 3. Feature Engineering
    # Calculate RUL for training data
    # RUL = Max Cycle for that unit - Current Cycle
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)

    # Prepare X_train, y_train
    # Drop identifying columns not useful for prediction (unit_number can be debatable, usually dropped or used for grouping)
    features = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X_train = train_df[features]
    y_train = train_df['RUL']

    # 4. Model Training
    # Using GradientBoostingRegressor as it generally performs better than Random Forest for RUL
    print("Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # 5. Prediction on Test Set
    # We need to predict RUL for the *last* recorded cycle of each unit in the test data
    # The true RUL provided in RUL_FD001.txt is for the state AFTER the last recorded cycle in test_df
    
    # Get the last record for each unit in test_df
    last_cycle_test = test_df.groupby('unit_number').last().reset_index()
    X_test = last_cycle_test[features]
    
    # Predict
    y_pred = model.predict(X_test)

    # Compare with True RUL
    # The True RUL in RUL_FD001.txt corresponds to the remaining life *after* the last cycle in test_df
    y_true = true_rul['RUL']

    # 6. Evaluation
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("-" * 30)
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)

    if r2 < 0.95:
        print("R^2 is below 0.95. Recommendation: Tune hyperparameters or use more advanced features (e.g., rolling means).")
    else:
        print("Model condition satisfied (R^2 >= 0.95).")

if __name__ == "__main__":
    train_and_evaluate()
