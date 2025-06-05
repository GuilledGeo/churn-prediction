# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    """Carga un archivo CSV como DataFrame."""
    return pd.read_csv(path)

def explore_data(df):
    """Muestra información general del DataFrame."""
    print("Shape:", df.shape)
    print("\nColumnas:", df.columns.tolist())
    print("\nTipos de datos:\n", df.dtypes)
    print("\nValores nulos por columna:\n", df.isnull().sum())
    print("\nEstadísticas descriptivas:\n", df.describe(include='all'))

def plot_churn_distribution(df, target_col='Churn'):
    """Grafica la distribución de la variable objetivo."""
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target_col, palette='Set2')
    plt.title('Distribución de la variable objetivo')
    plt.xlabel('Churn')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df, exclude_cols=None):
    """Grafica histogramas para variables numéricas."""
    if exclude_cols:
        df = df.drop(columns=exclude_cols)
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols].hist(bins=30, figsize=(15,10), layout=(int(len(num_cols)/3)+1, 3))
    plt.suptitle('Distribuciones de variables numéricas', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, target_col=None):
    """Muestra la matriz de correlación de variables numéricas."""
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()
