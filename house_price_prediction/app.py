import joblib
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/feature_names.pkl")
num_cols = joblib.load("models/numeric_features.pkl")

st.title("🏠 House Price Prediction")

# Numeric inputs
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sqft)", 500, 10000, 1500)
bedrooms = st.number_input("Bedrooms", 0, 10, 3)
bath = st.number_input("Bathrooms", 0, 5, 2)
garage = st.number_input("Garage Cars", 0, 5, 2)

# UI-only neighborhood
neighborhood_quality = {
    "Premium Area": [5, 5, 4, 5, 5],
    "Good Residential Area": [4, 4, 4, 4, 4],
    "Average Area": [3, 3, 3, 3, 3],
    "Developing Area": [2, 2, 3, 2, 2],
    "Poor Area": [1, 2, 2, 1, 1],
}

selected_neighborhood = st.selectbox(
    "Neighborhood Type",
    list(neighborhood_quality.keys())
)

if st.button("Predict Price"):
    data = pd.DataFrame(0, index=[0], columns=features)

    data["OverallQual"] = overall_qual
    data["GrLivArea"] = gr_liv_area
    data["BedroomAbvGr"] = bedrooms
    data["FullBath"] = bath
    data["GarageCars"] = garage

    # Neighborhood → influence multiplier (UI logic)
    school, hospital, transport, safety, market = neighborhood_quality[selected_neighborhood]

    influence = (school + hospital + transport + safety + market) / 25

    data[num_cols] = scaler.transform(data[num_cols])

    base_price = np.expm1(model.predict(data)[0])

    final_price = base_price * (0.7 + influence)*40

    st.success(f"Estimated Price: ₹{final_price:,.0f}")
