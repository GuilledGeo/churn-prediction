# 📁 src/features/build_features.py
from sklearn.model_selection import train_test_split

def split_data(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
