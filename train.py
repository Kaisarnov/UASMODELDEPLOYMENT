from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\n=== RESULTS ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    return acc


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print("Model saved:", path)