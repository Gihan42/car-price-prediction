import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'app', 'dataset', 'train.csv')

try:
    train_df = pd.read_csv(csv_path)
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ File not found: {csv_path}")
    exit(1)


def clean_running(df):
    df['running'] = pd.to_numeric(df['running'].astype(str).str.replace(' km', '').str.replace(',', ''), errors='coerce')
    df['running'] = df['running'].fillna(df['running'].median())
    return df

train_df = clean_running(train_df)


X = train_df.drop('price', axis=1)
y = train_df['price']

categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col != 'Id']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_X = pd.DataFrame(
    encoder.fit_transform(X[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)
OH_X.index = X.index

X_num = X[numerical_cols]
X_final = pd.concat([X_num, OH_X], axis=1)

# ✅ Ensure all column names are strings
X_final.columns = X_final.columns.astype(str)


X_train, X_valid, y_train, y_valid = train_test_split(X_final, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=2)
model.fit(X_train, y_train)


artifacts_path = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(artifacts_path, exist_ok=True)

joblib.dump(model, os.path.join(artifacts_path, "model.pkl"))
joblib.dump(encoder, os.path.join(artifacts_path, "encoder.pkl"))
joblib.dump(numerical_cols, os.path.join(artifacts_path, "numerical_cols.pkl"))
joblib.dump(categorical_cols, os.path.join(artifacts_path, "categorical_cols.pkl"))

print("✅ Model and encoder saved to 'artifacts/' folder.")
