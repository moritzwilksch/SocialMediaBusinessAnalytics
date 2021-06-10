# %%
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import itertools
from helpers import load_and_join_for_modeling, train_val_test_split, eval_regression
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


# %%
# df = df.fillna(0)
# exog = "vader pct_pos pct_neg volume num_tweets".split()
# endog = "label"

# model = ARIMA(endog=df.loc[:"2021-02-28", endog], exog=df.loc[:"2021-02-28", exog], order=(1, 0, 1))
# res = model.fit(method='hannan_rissanen')
# print(res.summary())


#%%
exog = "num_tweets".split()
endog = "label vader".split()
orders_til_5 = list(itertools.product(range(1, 6), range(1, 6)))
results = []

for ticker in tickers:
    df: pd.DataFrame = load_and_join_for_modeling(ticker)
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    train = pd.concat((xtrain, ytrain), axis=1)
    val = pd.concat((xval, yval), axis=1)
    # test = pd.concat((xtest, ytest), axis=1)

    for order in orders_til_5:
        c.print(f"[INFO] Fitting order = {order}...")
        model = sm.tsa.VARMAX(train[endog], exog=train[exog], order=order)
        res = model.fit()
        if not res.mle_retvals.get('converged'):
            c.print("[ERROR] MLE not converged!", style="white on red")
            acc = mae = -1
        else:
            print(res.summary())
            preds = res.forecast(exog=val[exog].values, steps=128)
            mae, acc = eval_regression(yval, preds.label)
        results.append({'order': order, 'mae': mae, 'acc': acc, 'ticker': ticker})

    c.print(f"[INFO] Done with {ticker}!", style="white on green")

#%%
# pd.DataFrame(results).pivot(index='order', values='acc', columns='ticker').to_csv(root_path + "30_results/VARX_paramtuning.csv", sep=";")
