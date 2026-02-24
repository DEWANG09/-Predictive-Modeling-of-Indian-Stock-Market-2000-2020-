#1.Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

#2.Load Dataset
import os
import glob

# path to all csv files
path = "data/*.csv"

# read and combine all files
files = glob.glob(path)
df_list = []
for file in files:
    temp = pd.read_csv(file)
    temp["Source_File"] = os.path.basename(file) 
    df_list.append(temp)
df = pd.concat(df_list, ignore_index=True)

print("Total rows:", len(df))
df.head()

#3.Data Preprocessing
# Convert Date Column
df = pd.concat(df_list, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

print("Total rows:", len(df))
df.head()

# 4.Basic Inspection
df.info()
df.describe()
df.isnull().sum()

# Handle Missing Values
df.ffill(inplace=True)

#5.Exploratory Data Analysis (EDA)
# Price Trend
plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title("Stock Closing Price Over Time")

plt.savefig("outputs/closing_price_trend.png") 
plt.close()  

# Daily Returns
df['Return'] = df['Close'].pct_change()

# Volatility
df['Volatility'] = df['Return'].rolling(window=21).std()


# 6. Feature Engineering
# Moving averages
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()

# Exponential Moving Average
df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Target
df['Future_Return'] = df['Close'].pct_change(periods=5).shift(-5)


# 7. Prepare Training Data
# Drop NA
df = df.dropna().copy()

print("Training on ALL companies")

# ===== Feature Selection =====
features = ['Close', 'MA10', 'MA50', 'MA200', 'EMA10', 'RSI', 'Volatility']

print("Available columns:", df.columns.tolist())

X = df[features]
y = df['Future_Return']

# Train-Test Split (Time Series)
# features and target already prepared as X, y
split = int(len(X) * 0.8)

# safety check
if split >= len(X):
    split = len(X) - 1

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# 8. Model Building
# Model 1: XGBoost
print("Starting model training...")

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)
print(" XGBoost trained")

xgb_pred = xgb.predict(X_test)

# Model 2: Random Forest
print(" Preparing subset for Random Forest...")

rf_sample_size = min(300000, len(X_train))  # adjust if needed

X_train_rf = X_train[:rf_sample_size]
y_train_rf = y_train[:rf_sample_size]

print(f"RF training on {len(X_train_rf)} samples")

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_rf, y_train_rf)
print(" Random Forest trained")

rf_pred = rf.predict(X_test)


# 9. Model Evaluation
# calculate metrics once
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== XGBoost Results =====")
print("XGB RMSE:", xgb_rmse)
print("XGB R2:", xgb_r2)

print("\n===== Random Forest Results =====")
print("RF RMSE:", rf_rmse)
print("RF R2:", rf_r2)

print("\n===== Model Comparison =====")
print("XGB RMSE:", xgb_rmse)
print("RF  RMSE:", rf_rmse)

# 10. Select and Save Best Model
import pickle
import os

# calculate RMSEs
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\n===== Final Model Selection =====")
print("XGB RMSE:", xgb_rmse)
print("RF  RMSE:", rf_rmse)

# choose best model
if xgb_rmse <= rf_rmse:
    best_model = xgb
    print(" XGBoost selected as best model")
else:
    best_model = rf
    print(" Random Forest selected as best model")

# create outputs folder
os.makedirs("outputs", exist_ok=True)

# save BEST model with fixed name
pickle.dump(best_model, open("outputs/best_model.pkl", "wb"))

print(" Best model saved as outputs/best_model.pkl")

# 11. Visualization
sample_size = min(2000, len(y_test)) 

y_sample = y_test.values[:sample_size]

if xgb_rmse <= rf_rmse:
    pred_sample = xgb_pred[:sample_size]
else:
    pred_sample = rf_pred[:sample_size]

plt.figure(figsize=(12,5))
plt.plot(y_sample, label="Actual", alpha=0.7)
plt.plot(pred_sample, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Returns (Sample)")
plt.savefig("outputs/actual_vs_pred.png")
plt.close()  

# 12. Detect Market Ups & Downs
if "Return" not in df.columns:
    df["Return"] = df["Close"].pct_change()

# Bull vs Bear Days
df["Market_Direction"] = np.where(df["Return"] > 0, "Up", "Down")

print("\n===== Market Direction Counts =====")
print(df["Market_Direction"].value_counts())

# Monthly Trend (pick one company safely)
one_company = df[df["Source_File"] == df["Source_File"].iloc[0]].copy()
one_company = one_company.sort_index()

monthly = one_company["Return"].resample("ME").mean()

# plot
plt.figure(figsize=(12,5))
monthly.plot(title="Monthly Average Returns")
plt.savefig("outputs/monthly_returns.png")
plt.close()

# save processed data for fast Streamlit loading
os.makedirs("outputs", exist_ok=True)
df.to_parquet("outputs/processed_stock.parquet")

print("Processed parquet saved")