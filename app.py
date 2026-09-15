import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Predictive Maintenance Agent",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = "predictive_maintenance_lstm.keras"
SCALER_PATH = "scaler.pkl"
WINDOW_SIZE = 30

RAW_COLUMNS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


@st.cache_resource
def load_assets():
    return tf.keras.models.load_model(MODEL_PATH), joblib.load(SCALER_PATH)


def prepare_raw_dataframe(df):
    df = df.copy()

    if all(c in df.columns for c in RAW_COLUMNS):
        return df[RAW_COLUMNS].copy()

    if df.shape[1] == 26:
        df.columns = RAW_COLUMNS
        return df

    raise ValueError(
        "Input must contain 26 raw C-MAPSS columns: "
        "unit, cycle, 3 settings and 21 sensors."
    )


def feature_engineering(df):
    df = df.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    for sensor in SENSOR_COLS:

        # Difference between current and previous cycle
        df[f"{sensor}_diff"] = (
            df.groupby("unit")[sensor].diff()
        )

        # Rolling mean
        df[f"{sensor}_roll_mean"] = (
            df.groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    10,
                    min_periods=1
                ).mean()
            )
        )

        # Rolling standard deviation
        df[f"{sensor}_roll_std"] = (
            df.groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    10,
                    min_periods=2
                ).std()
            )
        )

    # Fill engineered feature missing values
    eng = [
        c for c in df.columns
        if "_diff" in c
        or "_roll_mean" in c
        or "_roll_std" in c
    ]

    df[eng] = (
        df.groupby("unit")[eng]
        .transform(
            lambda x: x.ffill().bfill()
        )
        .fillna(0)
    )

    return df


def abnormal_pattern_score(engine_df):

    recent = (
        engine_df
        .sort_values("cycle")
        .tail(WINDOW_SIZE)
    )

    scores = []

    for sensor in SENSOR_COLS:

        values = (
            recent[sensor]
            .astype(float)
            .values
        )

        if len(values) < 4:
            continue

        split = len(values) // 2

        early = values[:split]
        late = values[split:]

        std = np.std(early)

        if std < 1e-8:
            scores.append(0.0)
        else:
            scores.append(
                abs(
                    np.mean(late)
                    - np.mean(early)
                ) / std
            )

    score = (
        float(np.mean(scores))
        if scores
        else 0.0
    )

    if score >= 2.0:
        status = "ABNORMAL / HIGH DEGRADATION"

    elif score >= 1.0:
        status = "WARNING / DEGRADATION DETECTED"

    else:
        status = "NORMAL"

    return score, status


def maintenance_alert(
    rul,
    risk,
    abnormal
):

    pct = risk * 100

    if (
        rul <= 10
        or pct >= 80
        or abnormal >= 2
    ):
        return (
            "CRITICAL",
            "🚨 IMMEDIATE MAINTENANCE REQUIRED"
        )

    if (
        rul <= 30
        or pct >= 50
        or abnormal >= 1
    ):
        return (
            "WARNING",
            "⚠️ SCHEDULE MAINTENANCE SOON"
        )

    return (
        "HEALTHY",
        "✅ NO IMMEDIATE MAINTENANCE REQUIRED"
    )


def predict_engine(
    engine_df,
    model,
    scaler
):

    # Check minimum sequence length
    if len(engine_df) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} cycles "
            "are required for prediction."
        )

    # Feature engineering
    processed = feature_engineering(
        engine_df
    )

    # ==========================================================
    # IMPORTANT FIX:
    # Use EXACT feature names and order stored in scaler.pkl
    # ==========================================================

    if not hasattr(
        scaler,
        "feature_names_in_"
    ):
        raise ValueError(
            "Scaler does not contain feature_names_in_. "
            "Please use the original scaler.pkl from training."
        )

    features = list(
        scaler.feature_names_in_
    )

    # Check whether all required features exist
    missing = [
        col
        for col in features
        if col not in processed.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing)
        )

    # Latest 30 cycles
    latest = (
        processed
        .sort_values("cycle")
        .tail(WINDOW_SIZE)
        .copy()
    )

    # ==========================================================
    # Scale using EXACT same feature order as training
    # ==========================================================

    scaled = scaler.transform(
        latest[features]
    )

    # Convert to LSTM input shape
    X = np.asarray(
        scaled,
        dtype=np.float32
    ).reshape(
        1,
        WINDOW_SIZE,
        len(features)
    )

    # ==========================================================
    # LSTM Prediction
    # ==========================================================

    rul_out, risk_out = model.predict(
        X,
        verbose=0
    )

    # Predicted RUL
    rul = max(
        0.0,
        float(
            np.asarray(
                rul_out
            ).reshape(-1)[0]
        )
    )

    # Failure probability
    risk = float(
        np.clip(
            np.asarray(
                risk_out
            ).reshape(-1)[0],
            0,
            1
        )
    )

    # ==========================================================
    # Abnormal Pattern Detection
    # ==========================================================

    abnormal, abnormal_status = (
        abnormal_pattern_score(
            latest
        )
    )

    # ==========================================================
    # Maintenance Decision
    # ==========================================================

    status, alert = maintenance_alert(
        rul,
        risk,
        abnormal
    )

    return (
        rul,
        risk,
        abnormal,
        abnormal_status,
        status,
        alert,
        latest
    )


# ==============================================================
# STREAMLIT UI
# ==============================================================

st.title(
    "⚙️ Predictive Maintenance Agent"
)

st.caption(
    "LSTM-based multivariate sensor "
    "time-series analysis"
)


# ==============================================================
# Check model and scaler
# ==============================================================

if not (
    Path(MODEL_PATH).exists()
    and Path(SCALER_PATH).exists()
):

    st.error(
        "Missing predictive_maintenance_lstm.keras "
        "or scaler.pkl. Put both files beside app.py "
        "in GitHub."
    )

    st.stop()


# ==============================================================
# Load model and scaler
# ==============================================================

try:

    model, scaler = load_assets()

except Exception as e:

    st.error(
        f"Model/scaler loading failed: {e}"
    )

    st.stop()


# ==============================================================
# File Upload
# ==============================================================

uploaded = st.sidebar.file_uploader(
    "Upload C-MAPSS sensor data",
    type=["csv", "txt"]
)


if uploaded is None:

    st.info(
        "Upload a C-MAPSS FD001 .txt/.csv file "
        "containing at least 30 cycles per selected engine."
    )

    st.stop()


# ==============================================================
# Read Dataset
# ==============================================================

try:

    if uploaded.name.lower().endswith(
        ".txt"
    ):

        raw = pd.read_csv(
            uploaded,
            sep=r"\s+",
            header=None
        )

    else:

        raw = pd.read_csv(
            uploaded
        )

    raw = prepare_raw_dataframe(
        raw
    )

except Exception as e:

    st.error(
        f"Could not read input: {e}"
    )

    st.stop()


# ==============================================================
# Engine Selection
# ==============================================================

raw = raw.sort_values(
    ["unit", "cycle"]
)

units = raw["unit"].unique()

unit = st.sidebar.selectbox(
    "Select engine",
    units
)

engine = raw[
    raw["unit"] == unit
].copy()


# ==============================================================
# Prediction
# ==============================================================

try:

    (
        rul,
        risk,
        abnormal,
        abnormal_status,
        status,
        alert,
        latest
    ) = predict_engine(
        engine,
        model,
        scaler
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ==============================================================
# Results
# ==============================================================

st.subheader(
    f"Engine {unit} — Predictive Maintenance Analysis"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Predicted RUL",
    f"{rul:.1f} cycles"
)

c2.metric(
    "Failure Probability",
    f"{risk * 100:.1f}%"
)

c3.metric(
    "Abnormal Score",
    f"{abnormal:.2f}"
)

c4.metric(
    "Time-Series Window",
    f"{len(latest)} cycles"
)


st.divider()


# ==============================================================
# Abnormal Pattern + Maintenance Alert
# ==============================================================

a, b = st.columns(2)


with a:

    st.subheader(
        "🔍 Abnormal Pattern Detection"
    )

    st.write(
        f"**{abnormal_status}**"
    )

    st.progress(
        min(
            abnormal / 3.0,
            1.0
        )
    )


with b:

    st.subheader(
        "🚨 Maintenance Alert"
    )

    st.write(
        f"**Status: {status}**"
    )

    if status == "CRITICAL":

        st.error(alert)

    elif status == "WARNING":

        st.warning(alert)

    else:

        st.success(alert)


st.divider()


# ==============================================================
# Sensor Trend Visualization
# ==============================================================

st.subheader(
    "📈 Sensor Trend Visualization"
)


plot_sensors = st.multiselect(
    "Select sensors",
    SENSOR_COLS,
    default=[
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7"
    ]
)


if plot_sensors:

    fig = go.Figure()

    for sensor in plot_sensors:

        fig.add_trace(
            go.Scatter(
                x=latest["cycle"],
                y=latest[sensor],
                mode="lines+markers",
                name=sensor
            )
        )

    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ==============================================================
# Processed Sensor Data
# ==============================================================

with st.expander(
    "📊 Processed Sensor Data"
):

    st.dataframe(
        latest[
            ["cycle"] + SENSOR_COLS
        ],
        use_container_width=True
    )


# ==============================================================
# Footer
# ==============================================================

st.caption(
    "Prototype abnormal-pattern thresholds: "
    "score ≥1.0 = warning; ≥2.0 = high degradation. "
    "Maintenance decisions also use RUL and failure probability."
)
