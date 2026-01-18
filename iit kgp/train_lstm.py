
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TurbofanDataset(Dataset):
    def __init__(self, X, y=None, mode='train'):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def prepare_data(seq_len=50):
    # Load
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Clean
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    
    # RUL for Train
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=125)

    # Scale Features
    sensor_cols = [c for c in train_df.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    # Generate Sequences for Training
    X_train_seq = []
    y_train_seq = []
    
    for unit in train_df['unit_number'].unique():
        unit_data = train_df[train_df['unit_number'] == unit]
        data_arr = unit_data[sensor_cols].values
        rul_arr = unit_data['RUL'].values
        
        # Create sliding windows
        for i in range(len(unit_data) - seq_len + 1):
            X_train_seq.append(data_arr[i : i+seq_len])
            y_train_seq.append(rul_arr[i+seq_len-1])
            
    X_train = np.array(X_train_seq)
    y_train = np.array(y_train_seq)
    
    # Generate Sequences for Test
    # For test, we only need the LAST sequence for each unit to predict the RUL at the end point
    X_test_seq = []
    
    for unit in test_df['unit_number'].unique():
        unit_data = test_df[test_df['unit_number'] == unit]
        if len(unit_data) >= seq_len:
            # Take last sequence
            data_arr = unit_data[sensor_cols].values
            X_test_seq.append(data_arr[-seq_len:])
        else:
            # Handle short sequences by padding (simple zero padding at start)
            # This is rare in FD001 if seq_len is small, but 30 is safe. 50 might be tight for some.
            # Minimum length in Test FD001 is 31. So 30 is OK. Let's use 30 to be safe.
            # If using 50, we need padding.
            pad_len = seq_len - len(unit_data)
            data_arr = unit_data[sensor_cols].values
            # Pad with zeros at the beginning
            padded = np.pad(data_arr, ((pad_len, 0), (0, 0)), 'constant', constant_values=0)
            X_test_seq.append(padded)

    X_test = np.array(X_test_seq)
    y_test = true_rul['RUL'].values
    
    return X_train, y_train, X_test, y_test, len(sensor_cols)


def train_lstm():
    SEQ_LEN = 30 # Safe for FD001
    BATCH_SIZE = 64
    EPOCHS = 40 # Reduced for faster results (User requested 'instant')
    LR = 0.001

    print("Preparing data...")
    X_train, y_train, X_test, y_test, input_dim = prepare_data(seq_len=SEQ_LEN)
    
    # Datasets
    train_dataset = TurbofanDataset(X_train, y_train)
    # No val split here for simplicity, using all for training to maximize info
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = LSTMRegressor(input_size=input_dim, hidden_size=150, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Training LSTM...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # Eval
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    # Comparison
    # Note: RUL_FD001 is the remaining life *after* the series. 
    # train_lstm predicts based on the series ending. 
    # Logic: The model predicts "How many cycles left from now".
    # For Test, we input the *last* window. The prediction IS the estimated RUL.
    # So direct comparison is correct.
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("-" * 30)
    print(f"LSTM R^2 Score: {r2:.4f}")
    print(f"LSTM RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # Save Model and Scaler
    import joblib
    import os
    os.makedirs('ai_hackathon/models', exist_ok=True)
    
    # Save State Dict
    torch.save(model.state_dict(), 'ai_hackathon/models/lstm_model.pth')
    
    # Save Scaler (we need to be careful, prepare_data creates it internally. 
    # Let's refactor slightly or just return it from prepare_data to save it here?
    # Actually, prepare_data is self-contained. 
    # I should modify prepare_data to return the scaler, or re-fit it here? 
    # Re-fitting is safer if deterministic. 
    # BUT, actually, let's just modify the end of this script to fit a fresh scaler on the same data and save it.
    # It's identical logic.)
    
    # Re-create scaler for saving
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    sensor_cols = [c for c in train_df.columns if c not in ['unit_number', 'time_in_cycles']]
    
    # RUL calculation not needed for scaler fit, only features
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    joblib.dump(scaler, 'ai_hackathon/models/scaler.pkl')
    
    # Save metadata
    joblib.dump({'sensor_cols': sensor_cols, 'seq_len': SEQ_LEN}, 'ai_hackathon/models/model_metadata.pkl')
    
    print("Model, Scaler, and Metadata saved to ai_hackathon/models/")

    if r2 >= 0.90:
        print("SUCCESS: Target Accuracy Met (>90%)")
    else:
        print("Result is good but maybe not >90% yet. More epochs or tuning needed.")

if __name__ == "__main__":
    train_lstm()
