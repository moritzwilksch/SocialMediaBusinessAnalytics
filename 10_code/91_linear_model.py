# %%
import statsmodels.api as sm
import parse
import pandas as pd
from itertools import product
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_regression
from statsmodels.tsa.vector_ar.var_model import VAR
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


#%%
# Broken VARMAX shit
"""SENTI = "vader"


exog = "num_tweets".split()
endog = f"label {SENTI}".split()
# orders_til_5 = list(itertools.product(range(1, 6), range(1, 6)))
orders_to_test = list(zip(range(1, 9), [0] * 8))
results = []

for ticker in tickers:
    df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    train = pd.concat((xtrain, ytrain), axis=1)
    val = pd.concat((xval, yval), axis=1)
    # test = pd.concat((xtest, ytest), axis=1)
    for order in orders_to_test:
        c.print(f"[INFO] Fitting order = {order}...")


        # model = sm.tsa.VARMAX(train[endog], exog=train[exog], order=order)  # ORIGINAL
        model = VAR(train[endog], exog=train[exog])  # real VAR


        res = model.fit(maxiter=250)
        if not res.mle_retvals.get('converged'):
            c.print("[ERROR] MLE not converged!", style="white on red")
            acc = mae = -1
        else:
            print(res.summary())
            preds = res.forecast(exog=val[exog].values, steps=92)
            mae, acc = eval_regression(yval, preds.label)
        results.append({'order': order, 'mae': mae, 'acc': acc, 'ticker': ticker})
    c.print(f"[INFO] Done with {ticker}!", style="white on green")
    break

pd.DataFrame(results).pivot(index='order', values='acc', columns='ticker').to_csv(root_path + f"30_results/VARX_paramtuning_{SENTI}_NEW.csv", sep=";")
"""







#%%
ticker = 'INTC'
SENTI = 'vader'


exog = "num_tweets pct_pos pct_neg volume".split()

results = []
for ticker, SENTI in product(tickers, ('ml_sentiment', 'vader')):
    print(f"[{ticker}]: {SENTI}")
    endog = f"label {SENTI}".split()

    df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)
    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

    train = pd.concat((xtrain, ytrain), axis=1)
    val = pd.concat((xval, yval), axis=1)
    test = pd.concat((xtest, ytest), axis=1).dropna()

    model = VAR(pd.concat((train, val))[endog], exog=pd.concat((train, val))[exog])  # real VAR

    print(parse.search("AIC -> {n},", model.select_order(maxlags=8).__str__()).named.get('n', 'NA'))

    res = model.fit(maxlags=8, verbose=True, ic='aic')
    print(res.summary())
    preds = res.forecast(test[endog].values, exog_future=test[exog].values, steps=87)
    mae, acc = eval_regression(ytest.dropna(), preds[:, 0])

    results.append(
        (ticker, SENTI, acc)
    )

    with open(root_path + f"30_results/CorrectVARX.log", 'a') as f:
        f.write(f"[{ticker}]: {SENTI}\n")
        f.write(f"Order = {parse.search('AIC -> {n},', model.select_order(maxlags=8).__str__()).named.get('n', 'NA')}\n")
        f.write(f"Accuracy = {acc}")
        f.write("\n")

#%%
pd.DataFrame(results, columns=('ticker', 'sentiment', 'acc')).pivot(index='ticker', columns='sentiment', values='acc').loc[tickers].round(3).to_csv(root_path + f"30_results/CorrectVARX.csv", sep=';')


#%%
# grab order for each model for export
with open(root_path + "30_results/CorrectVARX.log", 'r') as f:
    s = "".join(f.readlines())

parsed_list = list(parse.findall("[{ticker}]: {senti}\nOrder = {order}", s))[-20:]

pd.DataFrame([x.named for x in parsed_list]).pivot(index='ticker', columns='senti', values='order').loc[tickers].to_csv(root_path + f"30_results/CorrectVARX_orders.csv", sep=';')



