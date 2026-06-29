import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
encoders = joblib.load("label_encoders.pkl")

df = pd.read_csv("data_C.csv")

st.title("Credit Score Prediction System")
st.write("Enter the required features to predict the credit score")
st.write("By: 2602113546 - M.Kaisar.Novrenza")

inputs = {}

for col in feature_columns:

    if df[col].dtype == "object":

        options = df[col].dropna().unique().tolist()
        selected = st.selectbox(col, options)

        inputs[col] = encoders[col].transform([selected])[0]

    else:
        inputs[col] = st.number_input(col, value=float(df[col].mean()))


if st.button("Predict Credit Score"):

    input_df = pd.DataFrame([inputs])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    label_map = {0: "Good", 1: "Poor", 2: "Standard"}

    result = label_map[int(prediction[0])]
    confidence = np.max(proba)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")