#%%
import pandas as pd
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]
SENTI = "ml_sentiment"

df = pd.read_csv(f"../30_results/VARX_paramtuning_{SENTI}.csv", sep=";", index_col="order")
df[tickers]
#%%

df = df.round(3).astype("string")[tickers]
# df[df == "-1.0"] = "NC"
df.replace("-1.0", "-")[tickers].to_csv(f"../30_results/VARX_paramtuning_word_{SENTI}.csv", sep=";")
