
import pandas as pd
import numpy as np
import joblib
import os
import torch

def preprocess_and_save():
    print("Loading raw test data...")
    # Load raw files
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    
    # Try multiple paths for robustness
    if os.path.exists('test_FD001.txt'):
        test_path = 'test_FD001.txt'
        rul_path = 'RUL_FD001.txt'
    elif os.path.exists('data/test_FD001.txt'):
        test_path = 'data/test_FD001.txt'
        rul_path = 'data/RUL_FD001.txt'
    else:
        # Fallback absolute path
        test_path = 'd:/iit kgp/test_FD001.txt'
        rul_path = 'd:/iit kgp/RUL_FD001.txt'

    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    # Drop unused columns (setting_3, s18, s19)
    cols_to_drop = ['setting_3', 's18', 's19']
    test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print("Loading scaler...")
    # Load Scaler
    if os.path.exists('ai_hackathon/models/scaler.pkl'):
        scaler = joblib.load('ai_hackathon/models/scaler.pkl')
    else:
        # Fallback if not yet created (should be created by train_lstm.py)
        # We will wait or this script is run *after* training.
        print("Scaler not found. Ensure training is complete.")
        return

    # Scale Features
    sensor_cols = [c for c in test_df.columns if c not in ['unit_number', 'time_in_cycles']]
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    print("Saving processed data...")
    processed_data = {
        'test_df': test_df,
        'true_rul': true_rul,
        'sensor_cols': sensor_cols
    }
    
    os.makedirs('ai_hackathon/models', exist_ok=True)
    joblib.dump(processed_data, 'ai_hackathon/models/processed_test_data.pkl')
    print("Success! Processed data saved to ai_hackathon/models/processed_test_data.pkl")

if __name__ == "__main__":
    preprocess_and_save()
