import sys, os
# ✅ Add project root (fraud-detection) to sys.path so "src" is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from pathlib import Path
from src.inference import FraudModel

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# -------------------------------
# Load Model
# -------------------------------
model_path = Path("artifacts/model.pkl")
if not model_path.exists():
    st.error("⚠️ Model not found. Please train first using: python -m src.models")
    st.stop()

fm = FraudModel(model_path)

# -------------------------------
# Threshold Selection
# -------------------------------
thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

# -------------------------------
# Batch Scoring (CSV Upload)
# -------------------------------
st.subheader("📂 Upload Your CSV File for Fraud Prediction")
file = st.file_uploader("Upload transactions CSV", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)  # try default (comma-separated)
    except:
        df = pd.read_csv(file, sep="\s+")  # fallback (space-separated)

    flags, prob = fm.predict_flag(df, threshold=thr)
    out = df.copy()
    out["fraud_proba"] = prob
    out["fraud_flag"] = flags

    st.success("✅ Predictions completed (uploaded file)")
    st.dataframe(out.head(20))  # preview
    st.download_button("⬇ Download predictions", out.to_csv(index=False), "predictions.csv")

# -------------------------------
# Load Default Dataset (Button)
# -------------------------------
st.subheader("⚡ Quick Test with Default Dataset")
if st.button("Load default dataset (data/creditcard.csv)"):
    try:
        df = pd.read_csv("data/creditcard.csv")
        flags, prob = fm.predict_flag(df, threshold=thr)
        out = df.copy()
        out["fraud_proba"] = prob
        out["fraud_flag"] = flags

        st.success("✅ Predictions completed (default dataset)")
        st.dataframe(out.head(20))
        st.download_button("⬇ Download predictions", out.to_csv(index=False), "predictions_default.csv")
    except Exception as e:
        st.error(f"⚠️ Could not load default dataset: {e}")

# -------------------------------
# Footer Section (Credits + Photo)
# -------------------------------
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    # 👇 Make sure your file is in: fraud-detection/app/darshan_photo.jpeg
    st.image("app/darshan_photo.jpeg", caption="Darshan JR", width=150)

with col2:
    st.markdown("""
    ### 👨‍💻 Developed by  
    **DARSHAN JR (B.Tech AI&DS)**  

    ### 🙏 Constant Motivator  
    **Dr. Vijayraj (Asst Prof, AI&DS Dept)**  
    """)
