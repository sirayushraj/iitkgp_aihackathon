
import pickle
import os

f = r'd:\iit kgp\ai_hackathon\models\model_metadata.pkl'
try:
    with open(f, 'rb') as file:
        data = pickle.load(file)
        print(f"Features: {data.get('feature_cols')}")
except Exception as e:
    print(e)
