# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.preprocessing import load_and_join_for_modeling
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# ticker = "INTC"

# %%
if True:
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))

    for idx, ticker in enumerate(tickers):
        df: pd.DataFrame = load_and_join_for_modeling(ticker).dropna()
        plot_acf(df.vader, ax=axes[idx//2][idx % 2], lags=15, color='green', vlines_kwargs=dict(color='green'), zero=False, title=f"Autocorrelation of {ticker} Sentiment")

    sns.despine()
    plt.tight_layout()
    plt.savefig(root_path + f"30_results/plots/ACFplots_vader.png", dpi=200, facecolor='white')

#%%

with open(root_path + "30_results/ADFtests.log", 'w') as f:
    f.write("")

ticker = "INTC"
results = {}
for ticker in tickers:
    with open(root_path + "30_results/ADFtests.log", 'a') as f:

        df: pd.DataFrame = load_and_join_for_modeling(ticker).dropna()
        print(f"=== ADF Test for {ticker} ===")
        f.write(f"\n=== ADF Test for {ticker} ===\n")
        
        pvals = []
        for col in df.columns:
            pval = round(adfuller(df[col])[1], 4)
            print(f"{col + ': ':<13}p = {pval}")
            f.write(f"{col + ': ':<13}p = {pval}\n")
            pvals.append(pval)
        results.update({ticker: pvals})

#%%
pd.DataFrame(results, index=df.columns).round(3).to_csv(root_path + "30_results/ADF_pvals.csv", sep=";")
