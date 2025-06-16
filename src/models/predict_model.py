# 📁 src/models/predict_model.py
import pandas as pd

def predict_new(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

