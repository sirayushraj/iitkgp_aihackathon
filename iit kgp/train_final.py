
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TurbofanDataset(Dataset):
    def __init__(self, X, y=None, mode='train'):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.mode = mode
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.mode == 'train': return self.X[idx], self.y[idx]
        else: return self.X[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=150, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def add_expanding_features(df, sensor_cols):
    # Sort to ensure correct expanding calculation
    df = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    # Calculate expanding min and max for sensors
    # Groups by unit_number so expansion resets for each engine
    grouped = df.groupby('unit_number')[sensor_cols]
    
    exp_min = grouped.expanding().min().reset_index(level=0, drop=True)
    exp_min.columns = [f'{c}_expanding_min' for c in sensor_cols]
    
    exp_max = grouped.expanding().max().reset_index(level=0, drop=True)
    exp_max.columns = [f'{c}_expanding_max' for c in sensor_cols]
    
    # Standard rolling for short term context (keep window 15)
    roll_mean = grouped.rolling(window=15, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_mean.columns = [f'{c}_roll_mean' for c in sensor_cols]
    
    df = df.reset_index(drop=True)
    # Combine original + expanding + rolling
    df = pd.concat([df, exp_min, exp_max, roll_mean], axis=1)
    return df

def prepare_data(seq_len=30):
    col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Clean
    cols_to_drop = ['setting_3', 's18', 's19']
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)

    # RUL Calculation
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=125)

    # Feature Engineering
    sensor_cols = [c for c in train_df.columns if c.startswith('s')]
    print("Generating Expanding & Rolling features...")
    train_df = add_expanding_features(train_df, sensor_cols)
    test_df = add_expanding_features(test_df, sensor_cols)

    # Select Features (Exclude ID and Target)
    feat_cols = [c for c in train_df.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
    
    # Scale
    scaler = MinMaxScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    # Sequence Gen
    X_train_seq, y_train_seq = [], []
    for unit in train_df['unit_number'].unique():
        unit_data = train_df[train_df['unit_number'] == unit]
        data_arr = unit_data[feat_cols].values
        rul_arr = unit_data['RUL'].values
        for i in range(len(unit_data) - seq_len + 1):
            X_train_seq.append(data_arr[i : i+seq_len])
            y_train_seq.append(rul_arr[i+seq_len-1])
            
    X_train = np.array(X_train_seq)
    y_train = np.array(y_train_seq)

    # Test Sequences (Last sequence only)
    X_test_seq = []
    for unit in test_df['unit_number'].unique():
        unit_data = test_df[test_df['unit_number'] == unit]
        data_arr = unit_data[feat_cols].values
        if len(unit_data) >= seq_len:
            X_test_seq.append(data_arr[-seq_len:])
        else:
            pad_len = seq_len - len(unit_data)
            padded = np.pad(data_arr, ((pad_len, 0), (0, 0)), 'constant', constant_values=0)
            X_test_seq.append(padded)

    X_test = np.array(X_test_seq)
    y_test = true_rul['RUL'].values
    
    return X_train, y_train, X_test, y_test, len(feat_cols)

def train():
    SEQ_LEN = 30
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001

    X_train, y_train, X_test, y_test, input_dim = prepare_data(SEQ_LEN)
    train_loader = DataLoader(TurbofanDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMRegressor(input_dim, 150, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training on {input_dim} features...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%10==0: print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("-" * 30)
    print(f"Final R^2: {r2:.4f}")
    print(f"Final RMSE: {rmse:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    train()
