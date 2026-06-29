from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Credit Score Prediction API")

model = joblib.load("model.pkl")
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