# %%
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helpers import load_and_join_for_modeling, train_val_test_split
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


results = []

with open(f"./benchmarks/naive_baseline.log", 'w') as f:
    f.write("")

for ticker in tickers:
    print(f"==== {ticker} ====")
    df: pd.DataFrame = load_and_join_for_modeling(ticker).dropna(subset=['label'])  # drop last 2
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    # mean only
    preds = np.zeros_like(ytest) + pd.concat([ytrain, yval]).mean()
    
    mae = mean_absolute_error(ytest, preds)
    print(f"MAE = {mae:.5f}")

    z_mae = (mae - df['label'].mean())/(df['label'].std())
    print(f"z std. MAE = {z_mae:.3f}")


    if (pd.concat([ytrain, yval]) > 0).mean() > (pd.concat([ytrain, yval]) < 0).mean():
        print("Majority class = 1")
        preds = np.ones_like(ytest)
    else:
        print("Majority class = 0")
        preds = np.zeros_like(ytest)

    acc = accuracy_score((ytest > 0), preds)
    print(f"Accuracy = {acc:.4f}")

    with open(f"./benchmarks/naive_baseline.log", 'a') as f:
        f.write(f"==== {ticker} ====\n")
        f.write(f"MAE = {mae:.5f}\n")
        f.write(f"z std. MAE = {z_mae:.3f}\n")
        f.write(f"Accuracy = {acc:.4f}\n\n")

    results.append(
        {
            'ticker': ticker,
            'mae': mae,
            'zmae': z_mae,
            'accuracy': acc
        }
    )

#%%
res_df = pd.DataFrame.from_dict(results)
res_df = res_df.set_index('ticker', drop=True)

(
    res_df
    .round({'mae': 5, 'zmae': 3, 'accuracy': 3})
    .to_csv(root_path + "30_results/naive_baseline_benchmarks.csv", sep=";")
)
#%%
print(res_df.round({'mae': 5, 'zmae': 3, 'accuracy': 3}).to_markdown())
