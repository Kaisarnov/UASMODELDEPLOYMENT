import joblib
import numpy as np

def model_fn(model_dir):
    return joblib.load(model_dir + "/model.pkl")

def predict_fn(input_data, model):
    return model.predict(np.array(input_data))