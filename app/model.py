import joblib
import pandas as pd

# Load artifacts
model = joblib.load("artifacts/model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")
numerical_cols = joblib.load("artifacts/numerical_cols.pkl")
categorical_cols = joblib.load("artifacts/categorical_cols.pkl")


def predict_price(data: dict) -> float:
    df = pd.DataFrame([data])

    # Split features
    X_num = df[numerical_cols]
    X_cat = df[categorical_cols]

    # One-hot encode
    X_cat_encoded = pd.DataFrame(encoder.transform(X_cat),
                                 columns=encoder.get_feature_names_out(categorical_cols),
                                 index=df.index)

    # Combine features
    X_final = pd.concat([X_num, X_cat_encoded], axis=1)

    # Ensure all column names are strings (just in case)
    X_final.columns = X_final.columns.astype(str)

    # Predict
    prediction = model.predict(X_final)
    return float(prediction[0])