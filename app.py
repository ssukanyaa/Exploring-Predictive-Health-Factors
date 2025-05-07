import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="PCOS Prediction", layout="centered")
st.title("💊 PCOS Prediction App")
st.markdown("This app predicts the likelihood of PCOS based on input medical features.")

model = joblib.load("pcos_model_rf.pkl") 

age = st.selectbox("Age (years)", ["15-20","20-25","25-30","30-35","35-44","44 and above"])
hormonal_imbalance = st.selectbox("Hormonal Imbalance", ["Yes", "No"])
hyperandrogenism = st.selectbox("Hyperandrogenism", ["Yes", "No"])
hirsutism = st.selectbox("Hirsutism (Excess Hair Growth)", ["Yes", "No"])
conception_diff = st.selectbox("Difficulty in Conception", ["Yes", "No"])
insulin_resistance = st.selectbox("Insulin Resistance", ["Yes", "No"])
exercise_freq = st.selectbox("Exercise Frequency", ["Rarely", "1-2 Times a Week","3-4 Times a Week","6-8 Times a Week", "Never"])
exercise_type = st.selectbox("Exercise Type", ["No Exercise", "Cardio (e.g., running, cycling, swimming)", "Strength training (e.g., weightlifting, resistance exercises)", "Flexibility and balance (e.g., yoga, pilates)","High-intensity interval training (HIIT)",
                                                 "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)","Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)"])
exercise_duration = st.selectbox("Duration of Exercise", ["Not Applicable", "Less than 30 minutes", "30 minutes","45 minutes", "More than 30 minutes"])
sleep_hours = st.selectbox("Average Sleep Hours", ["3-4 hours", "Less than 6 hours", "6-8 hours", "9-12 hours", "More than 12 hours"])
exercise_benefit = st.selectbox("Do you feel benefits from exercise?", ["Somewhat","Not at All","Yes Significantly","Not Much"])

mappings = {
    "Age": {
        "15-20": 0, "20-25": 1, "25-30": 2,
        "30-35": 3, "35-44": 4, "44 and above": 5
    },
    "Binary": {"Yes": 1, "No": 0},
    "Exercise Frequency": {
        "Rarely": 0, "1-2 Times a Week": 1,
        "3-4 Times a Week": 2, "6-8 Times a Week": 3, "Never": 4
    },
    "Exercise Type": {
        "No Exercise": 0,
        "Cardio (e.g., running, cycling, swimming)": 1,
        "Strength training (e.g., weightlifting, resistance exercises)": 2,
        "Flexibility and balance (e.g., yoga, pilates)": 3,
        "High-intensity interval training (HIIT)": 4,
        "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)": 5,
        "Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)": 6
    },
    "Exercise Duration": {
        "Not Applicable": 0, "Less than 30 minutes": 1,
        "30 minutes": 2, "45 minutes": 3, "More than 30 minutes": 4
    },
    "Sleep Hours": {
        "3-4 hours": 0, "Less than 6 hours": 1,
        "6-8 hours": 2, "9-12 hours": 3, "More than 12 hours": 4
    },
    "Exercise Benefit": {
        "Not at All": 0, "Not Much": 1,
        "Somewhat": 2, "Yes Significantly": 3
    }
}

# Encode the inputs into a DataFrame
def encode_inputs():
    return pd.DataFrame([{
        "Age": mappings["Age"][age],
        "Hormonal_Imbalance": mappings["Binary"][hormonal_imbalance],
        "Hyperandrogenism": mappings["Binary"][hyperandrogenism],
        "Hirsutism": mappings["Binary"][hirsutism],
        "Conception_Difficulty": mappings["Binary"][conception_diff],
        "Insulin_Resistance": mappings["Binary"][insulin_resistance],
        "Exercise_Frequency": mappings["Exercise Frequency"][exercise_freq],
        "Exercise_Type": mappings["Exercise Type"][exercise_type],
        "Exercise_Duration": mappings["Exercise Duration"][exercise_duration],
        "Sleep_Hours": mappings["Sleep Hours"][sleep_hours],
        "Exercise_Benefit": mappings["Exercise Benefit"][exercise_benefit]
    }])

# Predict on button click
if st.button("Predict PCOS"):
    input_df = encode_inputs()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Likely PCOS (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Unlikely PCOS (Confidence: {prob:.2%})")

    st.caption("Disclaimer: This tool is for educational use only and not a medical diagnosis.")
