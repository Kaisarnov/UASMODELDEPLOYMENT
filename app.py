from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import gdown
import os

app = FastAPI(title="Credit Score Prediction API")

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1yp7BuvXdnf5-Grflsj1isWO9C9bTvJf2"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load("feature_columns.pkl")

@app.get("/")
def home():
    return {
        "message": "Credit Score Model API is running",
        "n_features": len(feature_columns)
    }


class InputData(BaseModel):
    features: dict 

@app.post("/predict")
def predict(data: InputData):

    input_df = pd.DataFrame([data.features])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_df)

    return {
        "prediction": int(prediction[0])
    }
