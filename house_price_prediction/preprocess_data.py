import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

os.makedirs("models", exist_ok=True)

def preprocess_data():
    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Log-transform target
    y = np.log1p(train["SalePrice"])

    # Drop ID, target, and Neighborhood
    train = train.drop(["Id", "SalePrice"], axis=1)
    test = test.drop(["Id"], axis=1)

    # Combine
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

    # ✅ KEEP ONLY NUMERIC COLUMNS
    num_cols = all_data.select_dtypes(include=np.number).columns.tolist()
    all_data = all_data[num_cols]

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    all_data[num_cols] = imputer.fit_transform(all_data[num_cols])

    # Scale
    scaler = StandardScaler()
    all_data[num_cols] = scaler.fit_transform(all_data[num_cols])

    # Split training data
    X = all_data.iloc[:len(y)]

    # Save artifacts
    joblib.dump(X, "models/X_train.pkl")
    joblib.dump(y, "models/y_train.pkl")
    joblib.dump(num_cols, "models/numeric_features.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Preprocessing completed successfully (numeric features only)")

if __name__ == "__main__":
    preprocess_data()
