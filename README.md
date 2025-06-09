# 🔬 RC Frequency Detection using Machine Learning

This project predicts the **actual oscillation frequency** of an RC (Resistor-Capacitor) circuit using a machine learning model trained on synthetic data. The model takes into account **tolerances** in resistor and capacitor values, which often affect real-world performance.

---

## 📌 Problem Statement

In real-world electronics, resistors and capacitors rarely hold their **nominal values** due to manufacturing tolerances (e.g., ±1%, ±5%). These small variations can significantly affect the **cutoff or oscillation frequency** of RC-based systems like filters or timers.

This project aims to:
- Simulate these variations in R and C values.
- Predict the **actual frequency** using an ML model trained on noisy, tolerance-based data.

---

## 🧠 What this Project Does

- **Generates synthetic data** for various combinations of R and C values with tolerances.
- Calculates **nominal and actual frequencies** based on formula:
  
  \[
  f = \frac{1}{2\\pi RC}
  \]
  
- Trains a **Random Forest Regressor** on the data.
- Deploys a **Streamlit web app** to input values and get predictions interactively.
