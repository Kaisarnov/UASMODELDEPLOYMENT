import mlflow
import mlflow.sklearn
import joblib

from imp_data import load_data
from preprocess import preprocess
from model import build_model
from train import split_data, train_model, evaluate, save_model


class CreditScorePipeline:

    def __init__(self):
        self.model = build_model()

    def run(self, path):

        mlflow.set_experiment("Credit_Score_Project")

        with mlflow.start_run():

            df = load_data(path)

            X, y, encoder = preprocess(df)

            feature_columns = X.columns.tolist()
            joblib.dump(feature_columns, "feature_columns.pkl")
        
            X_train, X_test, y_train, y_test = split_data(X, y)

            self.model = train_model(self.model, X_train, y_train)

            acc = evaluate(self.model, X_test, y_test)

            save_model(self.model)

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(self.model, "model")

        return acc