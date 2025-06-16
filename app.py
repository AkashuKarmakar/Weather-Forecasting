import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import warnings
import os

warnings.filterwarnings("ignore")


# Define model directory path
model_dir = "F:/projects/personal project/weather prediction_v1/models/"

# --- Function to load data and apply feature engineering ---
# This function replicates the feature engineering from your notebook
# to ensure the input data for prediction has the same structure as training data.
@st.cache_data # Cache the data loading and processing for performance
def load_and_process_data(file_path="F:/projects/personal project/weather prediction_v1/data/weather_atmospheric_processed.csv"):
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

    # Create lag features
    for col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Rolling means
    for col in df.columns:
        df[f"{col}_roll3"] = df[col].rolling(window=3).mean()

    # Time-based features
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear

    # Drop rows with NaNs due to shifting/rolling
    df.dropna(inplace=True)

    # For the app, we only need the features part after engineering
    # We will use the last row of this 'features_df' for prediction
    features_df = df.drop(columns=[col for col in df.columns if 'TMP_t+' in col], errors='ignore')
    return features_df

# --- Load all necessary assets (models, scaler, feature names) ---
@st.cache_resource # Cache resource loading (models are heavy)
def load_assets():
    try:
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        feature_names = json.load(open("feature_names.json", "r"))

        rf_model_t1 = joblib.load(os.path.join(model_dir, "rf_model_TMP_t+1.pkl"))
        rf_model_t2 = joblib.load(os.path.join(model_dir, "rf_model_TMP_t+2.pkl"))
        rf_model_t3 = joblib.load(os.path.join(model_dir, "rf_model_TMP_t+3.pkl"))

        # Custom objects need to be handled when loading LSTM models
        # If your LSTM model had custom layers/functions, you'd pass custom_objects here
        lstm_model_t1 = load_model(os.path.join(model_dir, "lstm_model_TMP_t+1.h5"), compile=False)
        lstm_model_t2 = load_model(os.path.join(model_dir, "lstm_model_TMP_t+2.h5"), compile=False)
        lstm_model_t3 = load_model(os.path.join(model_dir, "lstm_model_TMP_t+3.h5"), compile=False)

        return scaler, feature_names, rf_model_t1, rf_model_t2, rf_model_t3, \
               lstm_model_t1, lstm_model_t2, lstm_model_t3
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all model files (scaler.pkl, .pkl models, .h5 models, feature_names.json) are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Weather Prediction App", layout="centered")

st.title("☀️ Multi-Step Weather Prediction")
st.markdown("This application predicts temperature for the next 3 time steps (e.g., days) using Random Forest and LSTM models.")

# Load data and assets
features_df = load_and_process_data()
scaler, feature_names, rf_model_t1, rf_model_t2, rf_model_t3, \
lstm_model_t1, lstm_model_t2, lstm_model_t3 = load_assets()

st.subheader("Current Weather Data (from last historical record)")
# Display the last known historical weather data (which is used as input for prediction)
last_record = features_df.iloc[-1]
st.write(last_record)

if st.button("Predict Next 3 Days Temperature"):
    # Prepare the input for prediction (last row of engineered features)
    input_features = last_record.values.reshape(1, -1)

    # Scale the input features
    scaled_input_features = scaler.transform(input_features)

    # Reshape for LSTM (samples, timesteps=1, features)
    scaled_input_features_lstm = scaled_input_features.reshape((scaled_input_features.shape[0], 1, scaled_input_features.shape[1]))

    st.subheader("Predictions:")

    # --- Random Forest Predictions ---
    st.markdown("#### Random Forest Model")
    rf_pred_t1 = rf_model_t1.predict(scaled_input_features)[0]
    rf_pred_t2 = rf_model_t2.predict(scaled_input_features)[0]
    rf_pred_t3 = rf_model_t3.predict(scaled_input_features)[0]

    st.info(f"**Temperature (t+1):** {rf_pred_t1:.2f} k")
    st.info(f"**Temperature (t+2):** {rf_pred_t2:.2f} k")
    st.info(f"**Temperature (t+3):** {rf_pred_t3:.2f} k")

    # --- LSTM Predictions ---
    st.markdown("#### LSTM Model")
    lstm_pred_t1 = lstm_model_t1.predict(scaled_input_features_lstm)[0][0]
    lstm_pred_t2 = lstm_model_t2.predict(scaled_input_features_lstm)[0][0]
    lstm_pred_t3 = lstm_model_t3.predict(scaled_input_features_lstm)[0][0]

    st.success(f"**Temperature (t+1):** {lstm_pred_t1:.2f} k")
    st.success(f"**Temperature (t+2):** {lstm_pred_t2:.2f} k")
    st.success(f"**Temperature (t+3):** {lstm_pred_t3:.2f} k")

st.markdown("---")
st.markdown("Note: Predictions are based on the last available historical weather data point.")