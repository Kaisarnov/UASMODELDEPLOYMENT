from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

def preprocess(df):

    df = df.drop(columns=[c for c in df.columns if "ID" in c or "Unnamed" in c], errors="ignore")

    X = df.drop("Credit_Score", axis=1)
    y = df["Credit_Score"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    joblib.dump(encoders, "label_encoders.pkl")

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    joblib.dump(target_encoder, "target_encoder.pkl")

    return X, y, target_encoder