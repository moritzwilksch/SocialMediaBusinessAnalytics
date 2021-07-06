# %%
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from lightgbm import LGBMClassifier, plot_importance
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_classification
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
ticker = "NFLX"

SENTI = 'ml_sentiment'
# SENTI = 'vader'


for ticker, SENTI in product(tickers, ('vader', 'ml_sentiment')):
    print(f"[{ticker}]: {SENTI}")
    # close_price = pd.read_parquet(root_path + "20_outputs/financial_ts/INTC_stock.parquet")['Close'].ffill()
    df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)

    df.label = df.label > 0

    for i in range(2, 6):
        df[f'return_lag{i}'] = df['return'].shift(i)    # TODO: Feature Engineering?
        df[f'senti_lag{i}'] = df[SENTI].shift(i)    # TODO: Feature Engineering?
        df[f'ma_{i}'] = df['return'].rolling(i).mean()  # TODO: Feature Engineering?
        df[f'ms_{i}'] = df[SENTI].rolling(i).mean()  # TODO: Feature Engineering?

    df = df.fillna(0)

    def days_in_trend(returns):
        res = [0]
        for idx, currval in enumerate(returns):
            sign_t = np.sign(currval)
            sign_t_1 = np.sign(returns[idx-1 if idx > 0 else 0])
            if sign_t == sign_t_1:  # or sign_t == 0 or sign_t_1 == 0:
                res.append(res[-1]+1)
            else:
                res.append(0)
        return res[1:]

    # df = df.assign(diff=df.pct_pos - df.pct_neg)
    # df = df.assign(ratio=df.pct_pos/(df.pct_neg+1e-6))
    df = df.assign(days_in_trend=days_in_trend(df['return']))
    df = df.assign(dow=df.index.weekday)
    df = df.assign(moy=df.index.month)
    cat_fts = ['dow', 'moy']
    df[cat_fts] = df[cat_fts].astype('category')

    xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)


    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-5, 10, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
        }
        
        lr = LogisticRegression(n_jobs=1 if params['penalty'] == 'l1' else -1, solver='liblinear' if params['penalty'] == 'l1' else 'lbfgs', **params)

        lr.fit(xtrain, ytrain)
        preds = lr.predict(xval)
        return accuracy_score(yval, preds)


    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100)

    c.print(f"Refit, TEST performance:", style="bold underline")
    lr_model = LogisticRegression(solver='liblinear' if study.best_params['penalty'] == 'l1' else 'lbfgs', **study.best_params, n_jobs=-1)
    lr_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)))

    preds = lr_model.predict(xtest)
    print(f"{study.best_params} -> {study.best_value:.4f}")
    eval_classification(ytest, preds)
    print(classification_report(ytest, preds))

    with open(root_path + f"20_outputs/benchmarks/{ticker}/{SENTI}/LogisticRegStats.log", 'a') as f:
        f.write(f"Logistic Regression\n")
        f.write(f"{study.best_params}\nVALIDATION Accuracy = {study.best_value:.4f}\n")
        f.write(f"TEST Accuracy = {eval_classification(ytest, preds):.4f}" + "\n")
        f.write(classification_report(ytest, preds) + "\n")
