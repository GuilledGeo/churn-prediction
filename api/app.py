# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Cargar el modelo entrenado
model = joblib.load("outputs/models/modelo_gradient_boosting.pkl")

# Esquema del cliente esperado
class CustomerInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "🧠 API de predicción de churn lista."}

@app.post("/predict")
def predict_churn(data: CustomerInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "probability": round(probability, 4),
        "message": "Churn" if prediction == 1 else "No Churn"
    }
