import pandas as pd

def load_data(path="data_C.csv"):
    df = pd.read_csv(path)
    return df