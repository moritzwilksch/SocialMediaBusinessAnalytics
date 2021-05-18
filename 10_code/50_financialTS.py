# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import seaborn as sns
import yfinance as yf
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
yfticker = yf.Ticker("TSLA")

#%%
(
    yfticker
    .history(start="2019-01-01", end="2021-04-30")
    .reindex(
        pd.DatetimeIndex(pd.date_range(start="2019-01-01", end="2021-04-30"))
        )
)
