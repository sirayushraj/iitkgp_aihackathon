
import sys
import os
import torch
import pandas as pd

# Add ai_hackathon to path so we can import app
sys.path.append(os.path.join(os.getcwd(), 'ai_hackathon'))

try:
    from app import load_inference_components, load_data, preprocess_for_inference, predict_rul
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify():
    print("1. Loading components...")
    model, scaler, metadata = load_inference_components()
    
    if model is None:
        print("FAIL: Model not found. Training likely still in progress.")
        return

    print("PASS: Components loaded.")
    print(f"Metadata: {metadata.keys()}")

    print("2. Loading Data...")
    df = load_data() # No args, uses optimized loading logic
    if df is None:
        print("FAIL: Data not found.")
        return
    print(f"PASS: Data loaded ({len(df)} rows).")

    print("3. Running Inference on Unit 1...")
    unit1 = df[df['unit_nr'] == 1]
    tensor = preprocess_for_inference(unit1, scaler, metadata)
    
    if tensor is None:
        print("FAIL: Preprocessing failed.")
        return
    
    print(f"Tensor shape: {tensor.shape}")
    
    pred = predict_rul(model, tensor)
    print(f"Prediction: {pred:.4f}")
    
    if isinstance(pred, float):
        print("PASS: Prediction success.")
    else:
        print("FAIL: Prediction format error.")

if __name__ == "__main__":
    verify()
