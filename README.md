# House_price_prediction
Machine learning project for predicting house prices using regression models, with an interactive Streamlit web app for real-time price estimation.

# 🏠 House Price Prediction using Machine Learning

## 📌 Overview

This project is an end-to-end **House Price Prediction system** built using **Machine Learning** techniques. It predicts house prices based on key property features by training regression models on historical housing data. The best-performing model is deployed using a **Streamlit web application** that allows users to interactively estimate house prices.

The project demonstrates the complete ML workflow including data preprocessing, feature scaling, model training, evaluation, and deployment.

---

## 🚀 Features

* Data preprocessing with missing value handling and feature scaling
* Training and evaluation of multiple regression models
* Model performance evaluation using **R² Score** and **MAE**
* Deployment using an interactive **Streamlit UI**
* Real-time house price prediction based on user inputs

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * Pandas
  * NumPy
  * Scikit-learn
  * XGBoost
  * Streamlit
  * Joblib
  * Matplotlib

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── app.py                 # Streamlit web application
├── preprocess_data.py     # Data preprocessing and feature engineering
├── train_model.py         # Model training and evaluation
├── requirements.txt       # Project dependencies
│
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Testing dataset
│
├── models/
│   ├── best_model.pkl     # Trained ML model
│   ├── scaler.pkl         # Feature scaler
│   ├── feature_names.pkl  # Feature names
│   └── numeric_features.pkl
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Preprocess the Data

```bash
python preprocess_data.py
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Model Details

* Algorithms Used:

  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting / XGBoost Regressor
* Evaluation Metrics:

  * R² Score
  * Mean Absolute Error (MAE)

The best-performing model is saved and used for prediction in the web app.

---

## 🖥️ Web Application

The Streamlit interface allows users to:

* Select property attributes such as area, quality, number of rooms, etc.
* Choose neighborhood type
* Get an estimated house price instantly

---

## 📈 Future Enhancements

* Add more advanced feature engineering
* Include categorical feature handling
* Improve UI with visualizations
* Deploy the app on cloud platforms (Streamlit Cloud / AWS / Heroku)
