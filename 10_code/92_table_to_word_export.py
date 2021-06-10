#%%
import pandas as pd
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

df = pd.read_csv("../30_results/VARX_paramtuning.csv", sep=";", index_col="order")
df
#%%

df = df.round(3).astype("string")[tickers]
# df[df == "-1.0"] = "NC"
df.replace("-1.0", "-").to_csv("../30_results/VARX_paramtuning_word.csv", sep=";")
