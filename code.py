#1.Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    temp["Source_File"] = os.path.basename(file)  # optional but useful
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
plt.show()

# Daily Returns
df['Return'] = df['Close'].pct_change()

# Volatility
df['Volatility'] = df['Return'].rolling(window=21).std()

# 7.Feature Engineering
# Moving Averages
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()

# Exponential Moving Average
df['EMA10'] = df['Close'].ewm(span=10).mean()

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss

df['RSI'] = 100 - (100 / (1 + rs))

# Target Variable (Future Return)
df['Future_Return'] = df['Close'].pct_change().shift(-1)

# 8. Prepare Training Data
# Drop NA
df = df.dropna()

# Feature Selection
features = ['Close','MA10','MA50','MA200','EMA10','RSI','Volatility']
X = df[features]
y = df['Future_Return']

# Train-Test Split (Time Series Safe)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Model Building
# Model: XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)
xgb.fit(X_train_scaled, y_train)    

xgb_pred = xgb.predict(X_test_scaled)

# 10. Model Evaluation
print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))
print("XGB R2:", r2_score(y_test, xgb_pred))

# 11. Visualization of Predictions
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label='Actual')
plt.plot(xgb_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Returns")
plt.show()

# 12. Detect Market Ups & Downs
# Bull vs Bear Days
df['Market_Direction'] = np.where(df['Return'] > 0, 'Up', 'Down')
df['Market_Direction'].value_counts()

# Monthly Trend
monthly = df['Return'].resample('M').mean()
monthly.plot(figsize=(12,5), title='Monthly Average Returns')
plt.show()