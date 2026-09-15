# Predictive Maintenance Agent

LSTM-based predictive maintenance prototype using NASA C-MAPSS FD001 turbofan sensor time-series data.

## Minimum requirements covered

- Sensor data processing
- Time-series analysis
- Explicit abnormal/degradation pattern detection
- Failure prediction through RUL
- Failure probability score
- Maintenance alert generation
- Interactive sensor trend visualization

## Model outputs

- **RUL:** predicted Remaining Useful Life in cycles
- **Risk:** failure-risk probability

Training pipeline:
C-MAPSS FD001 -> preprocessing -> sensor difference/rolling features -> StandardScaler -> 30-cycle sequences -> multi-output LSTM.

## Validation results

- RUL MAE: **20.88 cycles**
- RUL RMSE: **30.26 cycles**
- Risk Accuracy: **96.42%**
- Risk Precision: **91.04%**
- Risk Recall: **88.55%**
- Risk F1: **89.78%**
- Risk AUC: **99.27%**

These are validation results from the prototype, not official FD001 test-set results.

## Files

```text
predictive-maintenance-agent/
├── app.py
├── predictive_maintenance_lstm.keras
├── scaler.pkl
├── requirements.txt
└── README.md
```

## Input

The app accepts C-MAPSS `.txt` files separated by whitespace or `.csv` files with these 26 raw columns:

`unit, cycle, setting_1, setting_2, setting_3, sensor_1 ... sensor_21`

At least 30 cycles are required for the selected engine.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Upload the files to GitHub and create a Streamlit app using `app.py` as the main file.

**Important:** Put the trained `predictive_maintenance_lstm.keras` and `scaler.pkl` files in the repository root beside `app.py`.

## Prototype note

The abnormal-pattern thresholds are application-level prototype thresholds, not NASA-defined failure thresholds.
