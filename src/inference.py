import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("artifacts/model.pkl")

class FraudModel:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.pipe = joblib.load(model_path)

    def predict_proba(self, df: pd.DataFrame):
        return self.pipe.predict_proba(df)[:, 1]

    def predict_flag(self, df: pd.DataFrame, threshold: float = 0.5):
        p = self.predict_proba(df)
        return (p >= threshold).astype(int), p
