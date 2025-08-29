import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib

from .features import build_preprocessor
from .data import load_data

ART_DIR = Path("artifacts"); ART_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = ART_DIR / "model.pkl"

def train_cv(df: pd.DataFrame, target: str = "Class"):
    y = df[target].values
    X = df.drop(columns=[target])

    pre = build_preprocessor(df, target=target)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_aucs, rocs, f1s = [], [], []

    for tr, va in skf.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        pr_aucs.append(average_precision_score(y[va], p))
        rocs.append(roc_auc_score(y[va], p))
        f1s.append(f1_score(y[va], (p >= 0.5).astype(int)))

    print(f"PR-AUC: {np.mean(pr_aucs):.3f}")
    print(f"ROC-AUC: {np.mean(rocs):.3f}")
    print(f"F1: {np.mean(f1s):.3f}")

    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH.resolve()}")

def evaluate(df: pd.DataFrame, target: str = "Class"):
    pipe = joblib.load(MODEL_PATH)
    y = df[target].values
    X = df.drop(columns=[target])
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("ROC-AUC:", roc_auc_score(y, proba))
    print("PR-AUC:", average_precision_score(y, proba))
    print(classification_report(y, preds))

if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    train_cv(df, target="Class")
    evaluate(df, target="Class")
