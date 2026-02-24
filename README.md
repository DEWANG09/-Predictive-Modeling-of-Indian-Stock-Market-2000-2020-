# -Predictive Modeling of Indian Stock Market 2000-2020

##  Overview

This project builds an end-to-end machine learning pipeline to analyze and predict next-day returns in the Indian stock market using historical data (2000–2020). The workflow includes data preprocessing, technical indicator engineering, model training, and an interactive Streamlit dashboard.

---
##  Project Highlights

- End-to-end stock market prediction pipeline  
- Technical indicator feature engineering  
- Time-series aware train/test split  
- Automated best model selection  
- Interactive Streamlit dashboard  

---
## Dataset

The dataset used in this project is available on Kaggle:

🔗 https://www.kaggle.com/datasets/sagara9595/stock-data

Download it and place it in the `data/` folder before running the code.

---
##  Methodology

- Data cleaning and preprocessing  
- Technical indicator engineering (EMA, RSI, etc.)  
- Time-series safe train-test split  
- Model training using XGBoost and Random Forest  
- Performance evaluation using RMSE and R²  
- Best model selection and persistence  

---
## Model Performance
  - XGBoost showed superior scalability and training efficiency on the full dataset.
  - Random Forest was additionally evaluated on a sampled subset to enable faster experimentation and model comparison.
  - The pipeline automatically selects and saves the best-performing model based on RMSE.

---
##  How to Run

### 1️. Clone the repository

git clone https://github.com/<your-username>/Predictive-Modeling-of-Indian-Stock-Market.git
cd Predictive-Modeling-of-Indian-Stock-Market


### 2️. Create a virtual environment (recommended)

conda create -n stock_env python=3.10
conda activate stock_env


### 3️. Install dependencies

pip install -r requirements.txt


### 4️. Add dataset

Download the dataset from Kaggle and place all CSV files inside:
data/


### 5️. Run training pipeline

python code.py

This will:
* preprocess data
* train XGBoost & Random Forest
* save the best model
* generate visualizations


### 6️. Launch the Streamlit app

streamlit run app.py

---
##  Tech Stack
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** XGBoost, Random Forest (Scikit-learn)  
- **Visualization:** Matplotlib  
- **Model Persistence:** Pickle  
- **Web App:** Streamlit  
- **Dataset Source:** Kaggle (Indian Stock Market Data)

---
