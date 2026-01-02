import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

X = joblib.load("models/X_train.pkl")
y = joblib.load("models/y_train.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_val)

print("R2 Score:", r2_score(y_val, preds))

joblib.dump(model, "models/best_model.pkl")
print("Model trained successfully")
