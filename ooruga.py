import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()


st.title("RC Circuit Frequency Predictor")
st.write("""
This application predicts the frequency of an RC circuit based on user input for resistance, capacitance, and tolerances.
""")


st.sidebar.header("Input Parameters")

def user_input_features():
    resistance = st.sidebar.number_input("Resistance Actual (Ω)", min_value=10.0, max_value=1e6, value=1000.0, step=10.0)
    capacitance = st.sidebar.number_input("Capacitance Actual (F)", min_value=1e-12, max_value=1e-3, value=1e-6, format="%.10f")
    r_tolerance = st.sidebar.slider("R Tolerance (%)", 0.0, 20.0, 5.0)
    c_tolerance = st.sidebar.slider("C Tolerance (%)", 0.0, 20.0, 10.0)

    data = {
        "Resistance Actual (Ω)": resistance,
        "Capacitance Actual (F)": capacitance,
        "R Tolerance": r_tolerance / 100,
        "C Tolerance": c_tolerance / 100
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()


st.subheader("User Input Parameters")
st.write(input_df)

X_scaled = scaler.transform(input_df)

if st.button("Predict Frequency"):
    y_log_pred = model.predict(X_scaled)
    y_pred_actual = np.expm1(y_log_pred)  
    st.subheader("Predicted Frequency (Hz)")
    st.write(f"{y_pred_actual[0]:,.2f} Hz")
