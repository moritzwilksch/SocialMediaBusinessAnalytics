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
SENTI = "vader"

if SENTI == 'vader':
    orders = {
        'TSLA':(5, 0),
        'AMZN':(2, 0),
        'FB':(3, 0),
        'MSFT':(5, 0),
        'AMD':(7, 0),
        'NVDA':(2, 0),
        'INTC':(3, 0),
    }
elif SENTI == 'ml_sentiment':
    orders = {
        'TSLA': (3, 0),
        'MSFT': (5, 0),
        'AMD': (7, 0),
        'NFLX': (7, 0),
        'NVDA': (4, 0),
        'INTC': (4, 0),
    }

exog = "num_tweets".split()
endog = f"label {SENTI}".split()

results = []

#%%
for ticker, order in orders.items():
    c.print(f"Fitting {ticker} model: order = {order}", style='white on blue')
    df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    _train = pd.concat((xtrain, ytrain), axis=1)
    _val = pd.concat((xval, yval), axis=1)
    trainval = pd.concat((_train, _val))

    test = pd.concat((xtest, ytest), axis=1)[:"2021-04-30"].fillna(0)


    model = sm.tsa.VARMAX(trainval[endog], exog=trainval[exog], order=order)
    res = model.fit(maxiter=250)
    if not res.mle_retvals.get('converged'):
        c.print("[ERROR] MLE not converged!", style="white on red")
        acc = mae = -1
    else:
        print(res.summary())
        preds = res.forecast(exog=test[exog].values, steps=127)
        mae, acc = eval_regression(ytest[:"2021-04-30"].fillna(0), preds.label)
    results.append((ticker, acc))
