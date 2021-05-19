import pandas as pd
import pytest
root_path = "./"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

@pytest.mark.parametrize('ticker', tickers)
def test_index_date_range(ticker):
    df = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet")
    assert pd.date_range(start="2019-01-01", end="2021-04-30").difference(df.index.unique()).empty
