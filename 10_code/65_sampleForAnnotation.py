# %%
import statsmodels.api as sm
import pandas as pd
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
# ticker = "INTC"
samples = []

for ticker in tickers:
    df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    samples.append(df.sample(300, random_state=42)[['tweet', 'id', 'created_at', 'link']])

#%%
total = pd.concat(samples)
total
#%%
total.to_csv(root_path + "20_outputs/SRS_sentiment.csv")