# %%
import statsmodels.api as sm
import pandas as pd
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_regression
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


#%%
exog = "num_tweets".split()
endog = "label vader".split()
# orders_til_5 = list(itertools.product(range(1, 6), range(1, 6)))
orders_to_test = list(zip(range(1, 9), [0] * 8))
results = []

for ticker in tickers:
    df: pd.DataFrame = load_and_join_for_modeling(ticker)
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    train = pd.concat((xtrain, ytrain), axis=1)
    val = pd.concat((xval, yval), axis=1)
    # test = pd.concat((xtest, ytest), axis=1)
    for order in orders_to_test:
        c.print(f"[INFO] Fitting order = {order}...")
        model = sm.tsa.VARMAX(train[endog], exog=train[exog], order=order)
        res = model.fit(maxiter=250)
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
pd.DataFrame(results).pivot(index='order', values='acc', columns='ticker').to_csv(root_path + "30_results/VARX_paramtuning.csv", sep=";")