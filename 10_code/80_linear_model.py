# %%
from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from helpers import load_and_join_for_modeling
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]



#%%
ticker = 'INTC'
df: pd.DataFrame = load_and_join_for_modeling(ticker)
df

#%%
from statsmodels.tsa.arima.model import ARIMA

df = df.fillna(0)
exog = "vader pct_pos pct_neg volume num_tweets".split()
endog = "return"

model = ARIMA(endog=df[endog], exog=df[exog], order=(1,0,0))
res = model.fit(method='hannan_rissanen')
print(res.summary())

#%%
model.predict(df[endog], df[exog])




#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression(n_jobs=-1)

X, y = df.drop('return', axis=1).fillna(0), df['return'].fillna(0)
lr.fit(X, y)
preds = lr.predict(X)
print((((preds > 0) & (y > 0)) | ((preds < 0) & (y < 0))).mean())

#%%
y = y.copy()
res = []

def runsim():
    y = np.random.choice(df["return"].fillna(0).copy(), replace=False, size=len(df))
    lr.fit(X, y)
    preds = lr.predict(X)
    return (((preds > 0) & (y > 0)) | ((preds < 0) & (y < 0))).mean()

res = Parallel(n_jobs=-1, prefer='processes')(delayed(runsim)() for _ in range(1000))

#%%
sns.histplot(res)
(np.array(res) > 0.3967).mean()
