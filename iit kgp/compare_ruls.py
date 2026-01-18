
import os
import joblib
import torch
import pandas as pd
import numpy as np
import sys

# Add path for app import if needed, but we can just use the model directly
sys.path.append(os.path.join(os.getcwd(), 'ai_hackathon'))
from app import LSTMRegressor

def check_rul():
    print("Loading optimized data...")
    try:
        data = joblib.load('ai_hackathon/models/processed_test_data.pkl')
        test_df = data['test_df']
        true_rul = data['true_rul'] # DataFrame or Series
        sensor_cols = data.get('sensor_cols')
        
        # Standardize unit_nr
        if 'unit_number' in test_df.columns:
            test_df.rename(columns={'unit_number': 'unit_nr'}, inplace=True)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(sensor_cols)
    model = LSTMRegressor(input_size=input_size, hidden_size=150, num_layers=2)
    model.load_state_dict(torch.load('ai_hackathon/models/lstm_model.pth', map_location=device))
    model.to(device)
    model.eval()

    print("\n" + "="*50)
    print(f"{'Unit':<6} | {'True RUL':<10} | {'Pred RUL':<10} | {'Error':<10}")
    print("="*50)

    total_error = 0
    num_samples = 10
    
    # Check first 10 units
    for unit_id in range(1, num_samples + 1):
        # Prepare Data
        unit_data = test_df[test_df['unit_nr'] == unit_id]
        
        # Assuming data in pickle IS ALREADY SCALED (from create_pickle logic)
        # We just need to extract the features.
        # Check column names in pickle. preprocess_data.py didn't rename to sX, 
        # but the scaler was trained on sX. 
        # preprocess_data.py saved 'test_df' which had 'sX' names if I recall...
        # Let's check:
        # preprocess_data.py: cols_to_drop = setting_3... names=col_names (s1...s21)
        # So yes, they are s1...s21.
        
        features = unit_data[sensor_cols].values
        
        # Sequence
        seq_len = 30
        if len(features) >= seq_len:
            seq = features[-seq_len:]
        else:
            pad = seq_len - len(features)
            seq = np.pad(features, ((pad, 0), (0, 0)), 'constant', constant_values=0)
            
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(tensor).item()
            
        # True RUL (RUL_FD001 is ordered by unit ID, row 0 is unit 1)
        # true_rul usually has index matching unit-1
        actual = true_rul.iloc[unit_id - 1]['RUL']
        
        error = abs(pred - actual)
        total_error += error
        
        print(f"{unit_id:<6} | {actual:<10.1f} | {pred:<10.1f} | {error:<10.1f}")

    print("="*50)
    print(f"Average Absolute Error (First {num_samples}): {total_error/num_samples:.2f} cycles")
    print("="*50)

if __name__ == "__main__":
    check_rul()
