# %%
import pandas as pd
import yfinance as yf
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
yfticker = yf.Ticker("TSLA")

#%%
df = (
    yfticker
    .history(start="2019-01-01", end="2021-04-30")
    .reindex(
        pd.DatetimeIndex(pd.date_range(start="2019-01-01", end="2021-04-30"))
        )
)
