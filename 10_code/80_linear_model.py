# %%
from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]



#%%
ticker = 'INTC'
df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
senti_df = pd.read_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")
prices = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet")
result = pd.merge(df, senti_df, how='left', on='id', validate="1:1")

#%%
daily_senti_ts = senti_df.groupby(result.created_at.dt.date)['vader'].mean()
returns = prices['Close'].pct_change()

# %%
df = (
    pd.merge(returns, daily_senti_ts, left_index=True, right_index=True)
    .rename({'Close': 'perf_1'}, axis=1)
)
    # # .ffill()

    # )

df = (
    df
    .assign(label=df['perf_1'].shift(-1))
    .dropna()
)

df
#%%
df.loc[df['perf_1']==0, 'perf_1'] = pd.NA

#%%
df = df.assign(perf_1=df['perf_1'].ffill())[~(df.label == 0)]

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression(n_jobs=-1)

X, y = df.drop('label', axis=1), df.label
lr.fit(X, y)
y = y.copy()
res = []

def runsim():
    y = np.random.choice(df.label.copy(), replace=False, size=len(df))
    lr.fit(X, y)
    preds = lr.predict(X)
    return (((preds > 0) & (y > 0)) | ((preds < 0) & (y < 0))).mean()

res = Parallel(n_jobs=-1, prefer='processes')(delayed(runsim)() for _ in range(1000))

#%%
sns.histplot(res)
(np.array(res) > 0.5549828178694158).mean()