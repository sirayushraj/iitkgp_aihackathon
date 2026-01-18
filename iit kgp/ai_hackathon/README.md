# 🔧 Turbofan Engine Predictive Maintenance System

A machine learning solution for predicting Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS FD001 dataset.

## 🎯 Project Overview

This project demonstrates predictive maintenance capabilities using:
- **XGBoost** for RUL prediction (RMSE < 20 cycles)
- **Rolling mean features** to capture sensor degradation trends
- **Streamlit dashboard** for real-time health monitoring

## 📁 Project Structure

```
ai_hackathon/
├── app.py              # Main application (all-in-one)
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Place datasets here
│   ├── train_FD001.txt
│   └── test_FD001.txt
└── models/            # Saved models (auto-generated)
    ├── xgb_model.json
    ├── scaler.pkl
    └── model_metadata.pkl
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the NASA C-MAPSS dataset and place the following files in the `data/` folder:
- `train_FD001.txt`
- `test_FD001.txt`

Dataset source: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

### 3. Train the Model

```bash
python app.py
```

This will:
- Load and preprocess the data
- Create rolling mean features
- Train an XGBoost model
- Save the model to `models/xgb_model.json`
- Print RMSE and feature importances

Expected output:
```
VALIDATION RMSE: ~18.5 cycles
Target: < 20 cycles | Status: PASS ✓
```

### 4. Launch Dashboard

**Option 1: Using Streamlit command**
```bash
streamlit run app.py
```

**Option 2: Using batch file (Windows)**
```bash
run_dashboard.bat
```

**Option 3: Using Python directly**
```bash
python -m streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 🎯 For Hackathon Live Demo

1. **Start the dashboard**: Run `streamlit run app.py`
2. **Select Random Engine**: Click the "🎲 PICK RANDOM ENGINE" button in the sidebar
3. **View Health Graph**: The Health Graph shows the RUL trajectory for submission
4. **Check RMSE**: See the test set RMSE in the sidebar
5. **Expand Model Explanation**: Click the expandable section to explain your model to judges

## 📊 Dashboard Features

- **Engine Selector**: Choose any engine unit from the dropdown
- **Health Metrics**: 
  - Remaining Useful Life (cycles)
  - Health Score (percentage)
  - Status (Healthy/Warning/Critical)
- **Trend Visualization**: Interactive health degradation chart
- **Feature Importance**: Top factors affecting predictions
- **Model Explanation**: Technical details for judges

## 🔬 Technical Details

| Aspect | Details |
|--------|---------|
| Algorithm | XGBoost Regressor |
| Features | 52 (39 raw + 13 rolling means) |
| Training Time | < 10 minutes (CPU) |
| Prediction Time | < 1 second |
| Validation RMSE | < 20 cycles |

### Why These Choices?

1. **XGBoost over LSTM**: 10x faster training, better interpretability
2. **Rolling Means (window=10)**: Captures degradation trends
3. **RUL Clipping at 130**: Standard piece-wise linear approach
4. **Health Thresholds**: Industry standard (70% healthy, 30% warning)

## 📈 Performance

- Training time: < 10 minutes on CPU
- Prediction time: < 1 second per engine
- Dashboard load time: < 5 seconds
- RMSE target: < 20 cycles ✓

## 🏆 Hackathon Judges

Expand the "Model Explanation" section in the dashboard for a complete methodology overview including:
- Algorithm selection rationale
- Feature engineering approach
- Data pipeline visualization
- Key sensor identification

## 📝 License

Built for Hackathon 2024 - Predictive Maintenance Challenge
