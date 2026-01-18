
import pickle
import os

files = [
    r'd:\iit kgp\ai_hackathon\models\model_metadata.pkl',
    r'd:\iit kgp\ai_hackathon\models\scaler.pkl'
]

for f in files:
    print(f"--- Inspecting {os.path.basename(f)} ---")
    if not os.path.exists(f):
        print("File not found.")
        continue
    
    try:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            print(f"Type: {type(data)}")
            print(f"Content: {data}")
    except Exception as e:
        print(f"Error loading pickle: {e}")
    print("\n")
