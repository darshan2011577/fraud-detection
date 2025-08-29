import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    return df

def split_data(df: pd.DataFrame, target: str = "Class", test_size: float = 0.2):
    """Random stratified split"""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=42
    )
    return train_df, test_df

