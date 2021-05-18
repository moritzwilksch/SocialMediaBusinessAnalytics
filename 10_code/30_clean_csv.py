# %%
import pandas as pd
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]
# %%
keep_cols = ['id',
             'created_at',
             'date',
             'time',
             'user_id',
             'username',
             'name',
             'tweet',
             'language',
             'mentions',
             'urls',
             'replies_count',
             'retweets_count',
             'likes_count',
             'hashtags',
             'cashtags',
             'link']


# %%
ticker = tickers[0]
df = pd.read_csv(
    f"../00_source_data/{ticker}_tweets.csv",
    usecols=keep_cols
)

# %%


def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(created_at=data.created_at.astype('datetime64'))


(
    df
    .pipe(fix_dtypes)
).info()
