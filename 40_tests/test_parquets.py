
import pandas as pd
import pytest
root_path = "./"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


@pytest.mark.parametrize('ticker', tickers)
def test_index_date_range(ticker):
    df = pd.read_parquet(root_path + f"20_outputs/parquets/{ticker}_tweets.parquet")
    assert pd.date_range(start="2019-01-01", end="2021-04-30").difference(df['created_at'].dt.date.unique()).empty


@pytest.mark.parametrize('ticker', tickers)
def test_nonzero_ntweets_perday(ticker):
    df = pd.read_parquet(root_path + f"20_outputs/parquets/{ticker}_tweets.parquet")
    assert all(df.groupby(df.created_at.dt.date)['id'].count() > 0)
