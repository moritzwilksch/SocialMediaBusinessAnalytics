#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
fig, ax = plt.subplots(figsize=(15,8))

for ticker in tickers:
    df = pd.read_csv(f"../00_source_data/{ticker}_tweets.csv")
    df.head()
    break