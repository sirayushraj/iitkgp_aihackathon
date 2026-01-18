
import pandas as pd
import numpy as np

def analyze_data():
    try:
        # Load data
        # Data does not have headers, so we define them based on standard C-MAPSS documentation or generic names
        # Columns: unit_number, time_in_cycles, setting_1, setting_2, setting_3, s1...s21
        col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
        
        print("Loading data...")
        train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
        test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
        rul_df = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
        
        print(f"Train Clean: {train_df.shape}")
        print(f"Test Shape: {test_df.shape}")
        print(f"RUL Shape: {rul_df.shape}")

        # Check for missing values
        print("\nChecking for missing values...")
        train_nan = train_df.isna().sum().sum()
        test_nan = test_df.isna().sum().sum()
        rul_nan = rul_df.isna().sum().sum()
        
        if train_nan == 0 and test_nan == 0 and rul_nan == 0:
             print("No missing values found.")
        else:
             print(f"Missing values found: Train={train_nan}, Test={test_nan}, RUL={rul_nan}")

        # Check for constant columns (std dev = 0)
        print("\nChecking for constant columns...")
        const_cols = [col for col in train_df.columns if train_df[col].std() == 0]
        if const_cols:
            print(f"Constant columns found in training data: {const_cols}")
        else:
            print("No constant columns found.")

        # Summary
        if train_nan == 0 and test_nan == 0 and rul_nan == 0 and not const_cols:
             print("\nDataset Status: CLEAN")
        else:
             print("\nDataset Status: NEEDS CLEANING")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_data()
