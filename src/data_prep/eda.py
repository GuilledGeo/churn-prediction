import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carga y descripción inicial
def load_data(path):
    return pd.read_csv(path)

def explore_data(df):
    print("▶️ Dimensiones del dataset:", df.shape)
    print("\n▶️ Tipos de variables:")
    print(df.dtypes)
    print("\n▶️ Porcentaje de valores nulos:")
    print(df.isnull().mean() * 100)

# 2. Análisis univariado
def plot_churn_distribution(df, target_col='Churn', palette='Set2'):
    """Grafica mejorada de la distribución de la variable objetivo (Churn)."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=target_col, palette=palette)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black')
    
    plt.title("Distribución de la variable objetivo (Churn)", fontsize=14)
    plt.xlabel("¿Cliente dado de baja?", fontsize=12)
    plt.ylabel("Número de clientes", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_numerical_distributions(df, exclude_cols=[]):
    """Muestra los histogramas de las variables numéricas, excluyendo las columnas indicadas."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(exclude_cols, errors='ignore')
    
    df[num_cols].hist(bins=20, figsize=(14, 10), color="#69b3a2", edgecolor='black')
    plt.suptitle("Distribuciones de variables numéricas", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, exclude_cols=[]):
    cat_cols = df.select_dtypes(include='object').columns.drop(exclude_cols, errors='ignore')
    for col in cat_cols:
        plt.figure(figsize=(6, 3))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        plt.show()

# 3. Análisis bivariado
def plot_numerical_vs_target(df, target_col='Churn'):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} vs {target_col}')
        plt.show()

def plot_categorical_vs_target(df, target_col='Churn'):
    cat_cols = df.select_dtypes(include='object').columns.drop(target_col, errors='ignore')
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target_col], normalize='index')
        contingency.plot(kind='bar', stacked=True)
        plt.title(f'{col} vs {target_col}')
        plt.ylabel("Proporción")
        plt.show()

def plot_correlation_matrix(df):
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación")
    plt.show()

# 4. Valores atípicos y limpieza
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return outliers

def clean_total_charges(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

# 5. Distribucion categorica
def plot_categorical_distributions(df, exclude_cols=[]):
    cat_cols = df.select_dtypes(include='object').columns.drop(exclude_cols, errors='ignore')
    for col in cat_cols:
        plt.figure(figsize=(6, 3))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        plt.show()

# 5. Preparación para modelado (a completar más adelante)
# Puedes crear un script aparte para esto (e.g. `features/prepare.py`)
