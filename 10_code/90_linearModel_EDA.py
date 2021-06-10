# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import load_and_join_for_modeling
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

ticker = "INTC"

# %%
if False:
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))

    for idx, ticker in enumerate(tickers):
        df: pd.DataFrame = load_and_join_for_modeling(ticker).dropna()
        plot_acf(df.label, ax=axes[idx//2][idx % 2], lags=15, color='blue', vlines_kwargs=dict(color='blue'), zero=False, title=f"Autocorrelation of {ticker} Returns")

    sns.despine()
    plt.tight_layout()
    plt.savefig(root_path + f"30_results/plots/ACFplots.png", dpi=200, facecolor='white')

