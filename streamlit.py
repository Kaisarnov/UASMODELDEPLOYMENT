import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

MODEL_PATH = "model.pkl"
MODEL_ID = "1yp7BuvXdnf5-Grflsj1isWO9C9bTvJf2"

FEATURE_PATH = "feature_columns.pkl"
ENCODER_PATH = "label_encoders.pkl"
DATA_PATH = "data_C.csv"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURE_PATH)

@st.cache_resource
def load_encoders():
    return joblib.load(ENCODER_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
feature_columns = load_features()
encoders = load_encoders()
df = load_data()


st.title("Credit Score Prediction System")
st.write("Enter the required features to predict the credit score")
st.write("By: 2602113546 - M.Kaisar.Novrenza")

inputs = {}

for col in feature_columns:

    if col in encoders:

        options = df[col].dropna().astype(str).unique().tolist()
        selected = st.selectbox(col, options)

        inputs[col] = encoders[col].transform([selected])[0]

    else:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        default_val = float(numeric_series.mean())

        inputs[col] = st.number_input(col, value=default_val)


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
