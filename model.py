from sklearn.ensemble import RandomForestClassifier

def build_model():

    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced" 
    )