import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Indian Stock Market Predictor", layout="wide")

st.title(" Indian Stock Market Analysis & Prediction")

@st.cache_data
def load_data():
    df = pd.read_parquet("outputs/processed_stock.parquet")

    # Returns
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change()

    # Volatility
    if "Volatility" not in df.columns:
        df["Volatility"] = df["Return"].rolling(window=21).std()

    # Moving averages
    if "MA10" not in df.columns:
        df["MA10"] = df["Close"].rolling(10).mean()

    if "MA50" not in df.columns:
        df["MA50"] = df["Close"].rolling(50).mean()

    if "MA200" not in df.columns:
        df["MA200"] = df["Close"].rolling(200).mean()

    # EMA
    if "EMA10" not in df.columns:
        df["EMA10"] = df["Close"].ewm(span=10).mean()

    # RSI
    if "RSI" not in df.columns:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

df = load_data()


st.sidebar.header("Controls")

companies = sorted(df["Source_File"].unique())
selected_company = st.sidebar.selectbox("Select Company", companies)

company_df = df[df["Source_File"] == selected_company]


st.subheader(f" Price Trend — {selected_company}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(company_df.index, company_df["Close"])
ax.set_title("Closing Price Over Time")
st.pyplot(fig)


model = pickle.load(open("outputs/best_model.pkl", "rb"))
scaler = pickle.load(open("outputs/scaler.pkl", "rb"))


st.subheader(" Next-Day Return Prediction")

# MUST MATCH TRAINING FEATURES EXACTLY
features = [
    "Close",
    "MA10",
    "MA50",
    "MA200",
    "EMA10",
    "RSI",
    "Volatility",
]

latest = company_df.iloc[-1:]
X_latest = latest[features]

# scale
X_scaled = scaler.transform(X_latest)

# predict
pred = model.predict(X_scaled)[0]

st.metric("Predicted Next-Day Return", f"{pred:.6f}")