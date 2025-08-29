import pandas as pd
import numpy as np

# -------------------------------
# Config
# -------------------------------
np.random.seed(42)
rows = 5000   # more realistic dataset size
fraud_ratio = 0.02  # ~2% fraud

# -------------------------------
# Generate Base Data
# -------------------------------
df = pd.DataFrame({
    "Time": np.random.randint(0, 86400, rows),  # seconds in a day
    "Amount": np.random.exponential(scale=100, size=rows).round(2),  # skewed distribution
})

# Add PCA-like anonymized features V1..V28
for i in range(1, 29):
    df[f"V{i}"] = np.random.randn(rows)

# Fraud labels (imbalanced)
df["Class"] = np.random.choice(
    [0, 1], size=rows, p=[1 - fraud_ratio, fraud_ratio]
)

# -------------------------------
# Add Engineered Features
# -------------------------------
# Convert Time → Hour
df["Hour"] = (df["Time"] // 3600) % 24

# Weekend flag (Sat=5, Sun=6)
df["DayOfWeek"] = np.random.randint(0, 7, rows)
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# Amount category
df["AmountCategory"] = pd.cut(
    df["Amount"],
    bins=[-1, 50, 200, 500, np.inf],
    labels=["Low", "Medium", "High", "VeryHigh"]
)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# Save CSV
# -------------------------------
df.to_csv("data/creditcard.csv", index=False)

print("✅ Advanced dataset saved at data/creditcard.csv")
print("Shape:", df.shape)
print("Fraud cases:", df['Class'].sum())
print("Columns:", list(df.columns))
