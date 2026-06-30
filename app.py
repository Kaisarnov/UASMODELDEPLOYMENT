import streamlit as st
import joblib
import pandas as pd
import gdown
import os

MODEL_PATH = "model.pkl"
FEATURE_PATH = "feature_columns.pkl"

# download model
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1yp7BuvXdnf5-Grflsj1isWO9C9bTvJf2"
    gdown.download(url, MODEL_PATH, quiet=False)

# download feature columns (YOU MUST ADD THIS FILE)
if not os.path.exists(FEATURE_PATH):
    st.error("feature_columns.pkl missing in deployment")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

st.title("Credit Score Prediction")

user_input = st.text_input("Enter JSON features")

if st.button("Predict"):
    try:
        data = eval(user_input)
        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(df)
        st.success(f"Prediction: {pred[0]}")

    except Exception as e:
        st.error(str(e))
