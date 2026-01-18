# Turbofan Engine Predictive Maintenance System

[cite_start]This repository hosts an end-to-end **Predictive Maintenance System** [cite: 16] [cite_start]engineered to estimate the **Remaining Useful Life (RUL)** of turbofan engines[cite: 1, 2]. [cite_start]By leveraging multivariate time-series sensor data, the system predicts how many operational cycles an engine can sustain before failure[cite: 17]. 

[cite_start]The project adheres to industry-grade design principles, ensuring a clear separation of concerns between data ingestion, processing, modeling, and deployment[cite: 18]. [cite_start]It is designed for scalability, reproducibility, and integration into enterprise asset management workflows[cite: 19].

---

## 1. Business Objective
[cite_start]Traditional maintenance strategies often rely on fixed schedules (preventive) or waiting for failure (reactive)[cite: 26]. [cite_start]This platform enables **Condition-Based Maintenance (CBM)**[cite: 27].

[cite_start]**The Goal:** Predict the precise RUL to optimize the trade-off between maximizing asset utilization and preventing catastrophic failure[cite: 28].

| Impact Area | Benefit |
| :--- | :--- |
| **Cost Reduction** | [cite_start]Avoids unnecessary maintenance on healthy engines and expensive repairs on failed ones[cite: 29]. |
| **Uptime** | [cite_start]Reduces unplanned downtime by forecasting failure windows[cite: 29]. |
| **Safety** | [cite_start]Identifies degradation patterns before they become critical safety hazards[cite: 29]. |
| **Logistics** | [cite_start]Optimizes spare parts inventory and maintenance crew scheduling[cite: 29]. |

---

## 2. High-Level Architecture
[cite_start]The solution follows a linear, modular pipeline where each stage is an independent component[cite: 31]:


* [cite_start]**Data Preprocessing:** Cleaning and RUL labeling[cite: 33].
* [cite_start]**Feature Engineering:** Rolling statistics, trend extraction, and scaling[cite: 34, 35].
* [cite_start]**Model Training Layer:** Supports Random Forest, XGBoost, LSTM, and Stacking Ensembles[cite: 36, 37].
* [cite_start]**Model Serialization:** Artifacts saved as `.pkl`, `.json`, or `.h5`[cite: 38].
* [cite_start]**Inference Service:** A user-facing Streamlit dashboard for real-time decision support[cite: 23, 39].

---

## 3. Repository Structure
[cite_start]The project is organized to separate the core ML pipeline from the deployment layer[cite: 42].

```text
iit kgp/
├── ai_hackathon/                  # INFERENCE & VISUALIZATION LAYER [cite: 43]
│   ├── app.py                     # Streamlit operational dashboard [cite: 43]
│   ├── run_dashboard.bat          # Windows 1-click launcher [cite: 43]
│   ├── requirements.txt           # Dashboard dependencies [cite: 43]
│   ├── data/                      # Sample input data for demonstration [cite: 43, 44]
│   └── models/                    # Deployment artifacts (Models & Scalers) [cite: 43, 44]
├── DATA FILES
│   ├── train_FD001.txt            # Historical training data [cite: 44]
│   ├── test_FD001.txt             # Test data for validation [cite: 44]
│   └── RUL_FD001.txt              # Ground truth RUL for test set [cite: 44, 45]
├── PIPELINE SCRIPTS
│   ├── preprocess_data.py         # Cleaning, RUL labeling, outlier removal [cite: 45]
│   ├── get_features.py            # Feature engineering & scaling [cite: 45]
│   ├── analyze_data.py            # EDA & Data visualization [cite: 45]
│   └── compare_ruls.py            # Post-training analysis (Pred vs Actual) [cite: 45]
├── TRAINING MODULES
│   ├── train_model.py             # Random Forest (Baseline) [cite: 45, 46]
│   ├── train_model_optimized.py   # Hyperparameter tuning logic [cite: 46]
│   ├── train_model_xgboost.py     # Gradient Boosting implementation [cite: 46]
│   ├── train_lstm.py              # Long Short-Term Memory (Deep Learning) [cite: 46]
│   ├── train_stacking.py          # Ensemble Stacking strategy [cite: 46]
│   └── train_final.py             # Final Production Model consolidation [cite: 46]
└── QUALITY ASSURANCE
    ├── inspect_pkl.py             # Artifact structural validation [cite: 47]
    └── verify_frontend.py         # Inference pipeline integration test [cite: 47]
