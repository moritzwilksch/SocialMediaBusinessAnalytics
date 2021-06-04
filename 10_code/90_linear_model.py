# %%
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from helpers import load_and_join_for_modeling
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


# %%
ticker = 'INTC'
df: pd.DataFrame = load_and_join_for_modeling(ticker)
df

# %%

# df = df.fillna(0)
exog = "vader pct_pos pct_neg volume num_tweets".split()
endog = "label"

model = ARIMA(endog=df.loc[:"2021-02-28", endog], exog=df.loc[:"2021-02-28", exog], order=(1, 0, 1))
res = model.fit(method='hannan_rissanen')
print(res.summary())

# %%
preds = res.forecast(exog=df.loc["2021-03-01":"2021-04-30", exog], steps=61)
real = df.loc["2021-03-01":"2021-04-30", 'return']

correct = (((preds > 0) & (real > 0)) | ((preds < 0) & (real < 0)))
plt.scatter(real, preds, color=correct.map({True: 'green', False: 'red'}))

c.print(f"Accuracy = {correct.mean():.3f}", style="white on blue")


#%%
