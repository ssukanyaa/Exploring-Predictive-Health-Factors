# 🧬 PCOS Prediction – Exploring Predictive Health Factors

This project aims to predict the presence of **Polycystic Ovary Syndrome (PCOS)** based on patient health data using classification algorithms.

## 📊 Dataset
- Source: (https://www.kaggle.com/competitions/exploring-predictive-health-factors/data)
- Features include: Age, Weight, Hyperandrogenism, Insulin_Resistance, Hormonal_Imbalance, Hirsutism, Exercise_Frequency, Sleep Hours and Perceived Exercise Benefit.

## 🧠 Objective
To build a model to classify whether a patient has PCOS based on  on a variety of clinical and lifestyle indicators.

## 🔍 Workflow
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data cleaning & missing value treatment
- ✅ Feature engineering
- ✅ Class imbalance handling with SMOTE
- Model training:
  - XG Boost
  - Gradient Boost
  - CatBoost
  - Voting Classifier
  - Random Forest (final model)
- ✅ End-to-end preprocessing pipeline using ColumnTransformer
- ✅ Final model deployed via a Streamlit web app

## 📈 Evaluation
- Primary Metric: ROC-AUC
- Best Score: ~80% ROC-AUC on private test set

## 💻 App Features (Streamlit)
- Input fields for user health and lifestyle data
- Instant prediction of PCOS likelihood
- Probability-based confidence score
- Lightweight, responsive interface
🔗 [Launch the App] (https://exploring-predictive-health-factors-pcos.streamlit.app/)

## 🧪 Model Deployment
- Preprocessing and classification steps saved as a pipeline using joblib
- Input data is passed as raw text; the model handles all encoding internally
- Classifier: RandomForestClassifier trained on SMOTE-balanced data

## ⚠️ Disclaimer
This tool is for educational purposes only and not a substitute for medical advice. For actual diagnosis or treatment, consult a certified healthcare provider.
