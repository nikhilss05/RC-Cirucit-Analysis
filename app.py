
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🔬 RC Frequency Predictor using ML")

st.header("Enter RC values and tolerances")

R_actual = st.number_input("Resistance (Ohms)", min_value=1.0, value=1000.0, step=1.0)
C_actual = st.number_input("Capacitance (F)", min_value=1e-12, value=1e-9, format="%e")
R_tol = st.selectbox("Resistance Tolerance", [0.01, 0.05, 0.10])
C_tol = st.selectbox("Capacitance Tolerance", [0.01, 0.05, 0.10])

if st.button("Predict Frequency"):
    input_scaled = scaler.transform([[R_actual, C_actual, R_tol, C_tol]])
    log_pred = model.predict(input_scaled)
    freq_pred = np.expm1(log_pred[0])
    st.success(f"🔊 Predicted Frequency: {freq_pred:.2f} Hz")
