import streamlit as st
import pandas as pd
import joblib

# Set page title
st.set_page_config(page_title="PCOS Prediction", layout="centered")
st.title("💊 PCOS Prediction App")
st.markdown("This app predicts the likelihood of PCOS based on input medical features.")

# Load the full model pipeline (preprocessing + model)
model = joblib.load("pcos_model_rf.pkl")  # NOTE: This should be the full pipeline, not just the model

# Collect user inputs
age = st.selectbox("Age (years)", ["15-20", "20-25", "25-30", "30-35", "35-44", "44 and above"])
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=150.0, value=55.0, step=0.5)
hormonal_imbalance = st.selectbox("Hormonal Imbalance", ["Yes", "No"])
hyperandrogenism = st.selectbox("Hyperandrogenism", ["Yes", "No"])
hirsutism = st.selectbox("Hirsutism (Excess Hair Growth)", ["Yes", "No"])
conception_diff = st.selectbox("Difficulty in Conception", ["Yes", "No"])
insulin_resistance = st.selectbox("Insulin Resistance", ["Yes", "No"])
exercise_freq = st.selectbox("Exercise Frequency", ["Rarely", "1-2 Times a Week", "3-4 Times a Week", "6-8 Times a Week", "Never"])
exercise_type = st.selectbox("Exercise Type", [
    "No Exercise",
    "Cardio (e.g., running, cycling, swimming)",
    "Strength training (e.g., weightlifting, resistance exercises)",
    "Flexibility and balance (e.g., yoga, pilates)",
    "High-intensity interval training (HIIT)",
    "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)",
    "Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)"
])
exercise_duration = st.selectbox("Duration of Exercise", ["Not Applicable", "Less than 30 minutes", "30 minutes", "45 minutes", "More than 30 minutes"])
sleep_hours = st.selectbox("Average Sleep Hours", ["3-4 hours", "Less than 6 hours", "6-8 hours", "9-12 hours", "More than 12 hours"])
exercise_benefit = st.selectbox("Do you feel benefits from exercise?", ["Somewhat", "Not at All", "Yes Significantly", "Not Much"])

# Wrap input into a DataFrame
def collect_inputs():
    return pd.DataFrame([{
        "Age": age,
        "Weight_kg": weight,
        "Hormonal_Imbalance": hormonal_imbalance,
        "Hyperandrogenism": hyperandrogenism,
        "Hirsutism": hirsutism,
        "Conception_Difficulty": conception_diff,
        "Insulin_Resistance": insulin_resistance,
        "Exercise_Frequency": exercise_freq,
        "Exercise_Type": exercise_type,
        "Exercise_Duration": exercise_duration,
        "Sleep_Hours": sleep_hours,
        "Exercise_Benefit": exercise_benefit
    }])

# Predict on button click
if st.button("Predict PCOS"):
    input_df = collect_inputs()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Likely PCOS (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Unlikely PCOS (Confidence: {prob:.2%})")

    st.caption("Disclaimer: This tool is for educational use only and not a medical diagnosis.")
