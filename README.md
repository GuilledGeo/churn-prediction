
# 🔄 Churn Prediction System – End-to-End ML Project

Este proyecto desarrolla un sistema completo de predicción de abandono de clientes (churn), desde la exploración de datos hasta la creación de una API desplegable con `FastAPI` y `Docker`, pasando por modelado, testing y CI/CD. El objetivo es detectar qué clientes tienen mayor probabilidad de causar baja, y anticiparse con acciones proactivas.

---

## 📦 Dataset

- **Fuente**: Dataset público simulado de churn de clientes de telecomunicaciones.
- **Contenido**: Información de clientes (servicios, cargos, historial, contratos).
- **Tamaño**: ~7.000 registros.
- **Objetivo (`Churn`)**: Variable binaria (1 = churn, 0 = no churn).

---

## 🎯 Objetivos del proyecto

- Explorar los datos y entender patrones relacionados con el abandono.
- Construir modelos predictivos y evaluar su rendimiento.
- Crear una estructura modular de producción con `src/`.
- Exportar modelos en `.pkl` y servir predicciones a través de una API.
- Dockerizar la aplicación para despliegue eficiente.
- Añadir tests, pre-commits y CI/CD con GitHub Actions.

---

## 🧪 Proceso de desarrollo

### 1. Exploración de datos (EDA)

Se realizó un análisis con gráficos y estadísticas:

- Distribuciones de `Churn`, cargos y duración del cliente (`tenure`).
- Variables categóricas como `Contract`, `PaymentMethod`, `OnlineSecurity`...
- Limpieza de `TotalCharges` y detección de valores atípicos.

📁 Ubicación: `notebooks/EDA.ipynb`
📁 Funciones reutilizadas: `src/data_prep/eda.py`

---

### 2. Preparación y limpieza

- Conversión de tipos (`TotalCharges` numérico).
- Codificación con Label Encoding.
- División en `train/test` con estratificación.

📁 Funciones clave:
- `load_csv()` en `src/data_prep/load_data.py`
- `split_data()` en `src/features/build_features.py`

---

### 3. Modelado predictivo

Se entrenaron varios modelos usando `scikit-learn`:

| Modelo               | Accuracy | F1 (churn) | ROC AUC |
|----------------------|----------|------------|---------|
| Gradient Boosting    | 0.80     | 0.57       | 0.84 ✅ |
| Logistic Regression  | 0.74     | 0.62       | 0.83 ✅ |
| Random Forest        | 0.79     | 0.53       | 0.82    |
| SVM                  | 0.75     | 0.00       | 0.77 ❌ |
| KNN                  | 0.74     | 0.44       | 0.72 ❌ |

📁 Entrenamiento: `src/models/train_model.py`
📁 Evaluación: `src/models/evaluate_model.py`
📁 Exportación a `.pkl`: `outputs/models/`

🔮 Se eligió como modelo final `GradientBoostingClassifier`.

---

### 4. Estructura modular `src/`

Se creó un entorno modular para producción:

```
src/
├── data_prep/
│   ├── load_data.py
│   └── eda.py
├── features/
│   └── build_features.py
├── models/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_model.py
├── scripts/
│   └── retrain_and_export.py
```

---

### 5. API de predicción con FastAPI

Se construyó una API con dos endpoints:

- `GET /` → mensaje de estado
- `POST /predict` → recibe JSON de cliente, devuelve predicción y probabilidad

📁 Código: `api/app.py`
📁 Cliente ejemplo: `predict.py`

---

### 6. Dockerización

La API se empaquetó con Docker:

- Imagen ligera basada en `python:3.10-slim`
- Se define `WORKDIR`, `COPY`, `RUN pip install`, `CMD`

📁 Dockerfile base
📁 Comandos en PowerShell (`make.ps1`):

```ps1
.\make.ps1 build     # Construye imagen
.\make.ps1 retrain   # Entrena modelo en contenedor
.\make.ps1 run       # Lanza API
.\make.ps1 clean     # Limpia modelos/cache
```

---

### 7. Testing

Se añadieron tests unitarios con `pytest`:

📁 Tests: `tests/test_train_model.py`, `tests/test_evaluate_model.py`

📁 Configuración: `pytest.ini`
```ini
[pytest]
pythonpath = .
```

---

### 8. CI/CD con GitHub Actions

Se configuró integración continua para:

- Ejecutar `pre-commit`
- Lanzar tests con `pytest`

📁 `.github/workflows/ci.yml`

```yaml
jobs:
  linter-and-tests:
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
      - name: Install deps + pytest
      - name: Run pre-commit
      - name: Run pytest
```

---

### 9. Pre-commit hooks

Se usan para mantener calidad de código (formato, estilo):

📁 `.pre-commit-config.yaml`
- `black`, `isort`, `trailing-whitespace`, `end-of-file-fixer`

---

## 📈 Resultados del modelo final

- **Modelo**: Gradient Boosting
- **Accuracy**: 0.80
- **F1-score (Churn)**: 0.57
- **ROC AUC**: 0.84
- **Top variables**:
  - `MonthlyCharges`
  - `tenure`
  - `TotalCharges`
  - `Contract`
  - `TechSupport`, `OnlineSecurity`

---

## 🧠 Lecciones aprendidas

- Cómo preparar un proyecto ML profesional de extremo a extremo
- Separar lógica por módulos (`src/`)
- Dockerizar una API de ML
- Añadir testing y CI/CD para producción robusta

---

## 🚀 Para lanzar la API localmente con Docker

```bash
.\make.ps1 build
.\make.ps1 retrain
.\make.ps1 run
```

Luego visita: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📫 Contacto

- 📧 Email: guillermodurantez@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/guillermodurantez/)

> ⭐ Este proyecto forma parte de mi portfolio de ciencia de datos.
> Repositorio: [github.com/GuilledGeo/churn-prediction](https://github.com/GuilledGeo/churn-prediction)
