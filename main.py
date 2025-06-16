# 📁 main.py
from src.data_prep.load_data import load_csv
from src.features.build_features import split_data
from src.models.train_model import train_gb, train_lr
from src.models.evaluate_model import evaluate_model
import joblib
import os

# 1. Cargar dataset
print("🔹 Cargando datos...")
df = load_csv("data/processed/df_churn_limpio_model1.csv")

# 2. Dividir en train/test
print("🔹 Dividiendo dataset...")
X_train, X_test, y_train, y_test = split_data(df)

# 3. Entrenar modelos
print("🔹 Entrenando modelo Gradient Boosting...")
gb_model = train_gb(X_train, y_train)

print("🔹 Entrenando modelo Logistic Regression...")
lr_model = train_lr(X_train, y_train)

# 4. Evaluar modelos
evaluate_model(gb_model, X_test, y_test, model_name="Gradient Boosting")
evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")

# 5. Exportar modelos
print("🔹 Exportando modelos...")
joblib.dump(gb_model, "outputs/models/modelo_gradient_boosting.pkl")
joblib.dump(lr_model, "outputs/models/modelo_logistic_regression.pkl")
print("✅ Modelos exportados a outputs/models/")
