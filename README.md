# -Predictive Modeling of Indian Stock Market 2000-2020
## Dataset

The dataset used in this project is available on Kaggle:

🔗 https://www.kaggle.com/datasets/sagara9595/stock-data

Download it and place it in the `data/` folder before running the code.

## Model Performance Note
  - XGBoost showed superior scalability and training efficiency on the full dataset.
  - Random Forest was additionally evaluated on a sampled subset to enable faster experimentation and model comparison.
  - The pipeline automatically selects and saves the best-performing model based on RMSE.
