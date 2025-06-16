# 📁 src/data/load_data.py
import pandas as pd

def load_csv(path):
    """Carga un dataset CSV desde la ruta dada."""
    df = pd.read_csv(path)
    return df.drop(columns=["Churn_bin"], errors="ignore")