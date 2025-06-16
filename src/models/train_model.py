# 📁 src/models/train_model.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def train_gb(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lr(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

