# 📁 predict_batch.py

import pandas as pd
import joblib
import os

# 1. Cargar el modelo entrenado
model = joblib.load("outputs/models/modelo_gradient_boosting.pkl")

# 2. Cargar los nuevos datos desde un CSV
input_path = "data/to_predict/nuevos_clientes.csv"
df = pd.read_csv(input_path)

# 3. Generar predicciones
preds = model.predict(df)
probs = model.predict_proba(df)[:, 1]

# 4. Añadir resultados al DataFrame
df["Churn_pred"] = preds
df["Churn_prob"] = probs

# 5. Guardar los resultados en CSV
os.makedirs("outputs/predictions", exist_ok=True)
output_path = "outputs/predictions/predicciones.csv"
df.to_csv(output_path, index=False)

print(f"✅ Predicciones guardadas en: {output_path}")
