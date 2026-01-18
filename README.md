Turbofan Engine Predictive Maintenance SystemThis repository hosts an end-to-end Predictive Maintenance System 1111engineered to estimate the Remaining Useful Life (RUL) of turbofan engines2222. By leveraging multivariate time-series sensor data, the system predicts how many operational cycles an engine can sustain before failure3.The project adheres to industry-grade design principles, ensuring a clear separation of concerns between data ingestion, processing, modeling, and deployment4. It is designed for scalability, reproducibility, and integration into enterprise asset management workflows5.1. Business ObjectiveTraditional maintenance strategies often rely on fixed schedules (preventive) or waiting for failure (reactive)6. This platform enables Condition-Based Maintenance (CBM)7.The Goal: Predict the precise RUL to optimize the trade-off between maximizing asset utilization and preventing catastrophic failure8.Impact AreaBenefitCost ReductionAvoids unnecessary maintenance on healthy engines and expensive repairs on failed ones9.UptimeReduces unplanned downtime by forecasting failure windows10.SafetyIdentifies degradation patterns before they become critical safety hazards11.LogisticsOptimizes spare parts inventory and maintenance crew scheduling12.2. High-Level ArchitectureThe solution follows a linear, modular pipeline where each stage is an independent component13:Raw Sensor Data Ingestion 14Data Preprocessing: Cleaning and RUL labeling15.Feature Engineering: Rolling statistics, trend extraction, and scaling16.Model Training Layer: Supports Random Forest, XGBoost, LSTM, and Stacking Ensembles17.Model Serialization: Artifacts saved as .pkl, .json, or .h518.Inference Service: A Streamlit-based operational dashboard19.3. Repository StructureThe project is organized to separate the core ML pipeline from the deployment layer20.Plaintextiit kgp/
├── ai_hackathon/                  # INFERENCE & VISUALIZATION LAYER [cite: 43]
│   ├── app.py                     # Streamlit operational dashboard [cite: 43]
│   ├── run_dashboard.bat          # Windows 1-click launcher [cite: 43]
│   ├── requirements.txt           # Dashboard dependencies [cite: 43]
│   ├── data/                      # Sample input data for demonstration [cite: 43, 44]
│   └── models/                    # Deployment artifacts (Models & Scalers) [cite: 44]
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
4. Data DescriptionThe dataset consists of time-series data from multiple aircraft engines21. Each engine starts with different degrees of initial wear and manufacturing variation22.Engine ID: Unique identifier for the asset23.Cycle: The current operational time index (counter)24.Operational Settings (3): Environmental parameters like Altitude, Mach Number, and Throttle Resolver Angle25.Sensor Measurements (21): Telemetry from subsystems including Fan speed, Core speed, Pressure, and Temperature26.Target Variable (RUL): Defined as $RUL = \text{Max Observed Cycle} - \text{Current Cycle}$27.5. Feature Engineering StrategyThe get_features.py script applies robust transformations to capture health dynamics28:Noise Smoothing: Applies rolling window averages to dampen high-frequency sensor noise29.Trend Extraction: Calculates rolling standard deviations and degradation rates30.Scaling: Normalizes features (StandardScaler/MinMaxScaler) for model convergence31.Temporal Behavior: Captures the “velocity” of degradation over time32.6. Model PortfolioModel TypeScriptDescriptionRandom Foresttrain_model.pyBaseline model. High interpretability, handles non-linearities well33.XGBoosttrain_model_xgboost.pyHigh Performance. Gradient boosting framework known for speed and accuracy34.LSTMtrain_lstm.pyDeep Learning. RNN designed to capture long-term temporal dependencies35.Stackingtrain_stacking.pyEnsemble. Combines predictions from multiple weak learners36.Final Modeltrain_final.pyThe chosen best-configuration model ready for deployment37.7. Installation & SetupPrerequisitesPython 3.8+ 38pip package manager 39Environment SetupBash# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
40Install DependenciesBashpip install numpy pandas scikit-learn xgboost tensorflow matplotlib seaborn streamlit
418. Execution WorkflowA. Training PipelineRun the scripts in the following order to process data and generate model artifacts42:Preprocess Data: python preprocess_data.py 43Generate Features: python get_features.py 44Train Models: 45python train_model.py (Baseline)python train_model_xgboost.py (Boosting)python train_lstm.py (Deep Learning)python train_stacking.py (Ensemble)python train_final.py (Finalize)Evaluate: python compare_ruls.py 46B. Inference & DashboardThe ai_hackathon folder contains the operational interface47.Method 1: Double-click run_dashboard.bat inside the ai_hackathon folder48.Method 2 (Command Line):Bashcd ai_hackathon
pip install -r requirements.txt
streamlit run app.py
49Operational Views:Fleet Monitor: Displays healthy engines with green status arcs and stable readings50.Critical Alerts: Highlights engines below 30% RUL with red indicators and anomaly warnings51.Model Transparency: Details the LSTM architecture and sensor importance rankings52.9. Validation & Quality AssuranceStrict validation scripts are included for production readiness53:inspect_pkl.py: Audits serialized model files to ensure they contain correct metadata and feature lists54.verify_frontend.py: Runs a “dry run” of inference logic to guarantee backend/frontend alignment55.compare_ruls.py: Analyzes RMSE to detect systematic under or over-prediction56.10. Production RoadmapRecommended enhancements for enterprise integration57:API Migration: Refactor app.py into a RESTful API using FastAPI58.Data Drift Detection: Monitor incoming sensor data for statistical deviations59.CI/CD: Automate retraining using GitHub Actions or Jenkins60.Model Registry: Store artifacts in versioned object stores (e.g., AWS S3, MLflow)61.Developed for IIT KGP AI Hackathon 62Would you like me to create the content for one of the specific Python scripts mentioned in the repository structure?
