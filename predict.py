# 📁 predict.py
import joblib
import pandas as pd
from src.models.predict_model import predict_new

# Cargar modelo desde archivo
model_path = "outputs/models/modelo_gradient_boosting.pkl"
model = joblib.load(model_path)

# Datos simulados de ejemplo (con las mismas columnas que X_label)
data = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 5,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 2,
    'OnlineSecurity': 0,
    'OnlineBackup': 1,
    'DeviceProtection': 1,
    'TechSupport': 0,
    'StreamingTV': 1,
    'StreamingMovies': 1,
    'Contract': 0,
    'PaperlessBilling': 1,
    'PaymentMethod': 3,
    'MonthlyCharges': 70.35,
    'TotalCharges': 351.75
}

# Ejecutar predicción
pred, prob = predict_new(model, data)
print(f"Predicción: {'Churn' if pred == 1 else 'No Churn'}")
print(f"Probabilidad de churn: {prob:.2f}")
