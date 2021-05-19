# %%
import pandas as pd
import yfinance as yf
import time
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


for ticker in tickers:
    print(f"[{ticker}]...")
    yfticker = yf.Ticker(ticker)

    df: pd.DataFrame = (
        yfticker
        .history(start="2019-01-01", end="2021-04-30")
        .reindex(
            pd.DatetimeIndex(pd.date_range(start="2019-01-01", end="2021-04-30"))
            )
    )

    df.to_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet")
    time.sleep(2)