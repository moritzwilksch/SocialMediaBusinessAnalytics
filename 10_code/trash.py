#%%
from sklearn.svm import SVC
# %%
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from lightgbm import LGBMClassifier, plot_importance
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_classification
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
ticker = "INTC"

SENTI = 'ml_sentiment'
# SENTI = 'vader'

# close_price = pd.read_parquet(root_path + "20_outputs/financial_ts/INTC_stock.parquet")['Close'].ffill()
df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)

#%%
df.label = df.label > 0


for i in range(2, 6):
    df[f'return_lag{i}'] = df['return'].shift(i)    # TODO: Feature Engineering?
    df[f'senti_lag{i}'] = df[SENTI].shift(i)    # TODO: Feature Engineering?
    df[f'ma_{i}'] = df['return'].rolling(i).mean()  # TODO: Feature Engineering?
    df[f'ms_{i}'] = df[SENTI].rolling(i).mean()  # TODO: Feature Engineering?

df = df.fillna(0)

# %%


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


df = df.assign(days_in_trend=days_in_trend(df['return']))
df = df.assign(dow=df.index.weekday)
df = df.assign(moy=df.index.month)
cat_fts = ['dow', 'moy']
df[cat_fts] = df[cat_fts].astype('category')

#%%
# price = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet")
# price_ma = price['Close'].ffill().rolling(9).mean().bfill()
# price_over_ma = price['Close'] > price_ma

# df = df.join(price_over_ma.rename('price_over_ma'))
# df['price_over_ma'] = df['price_over_ma'].astype('bool')

# %%
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)


#%%
import optuna

def objective_svc(trial):
    params = {
        'C': trial.suggest_float('C', 1e-6, 10, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    }
    svc = KNeighborsClassifier(n_jobs=-1, **params)

    svc.fit(xtrain, ytrain)
    preds = svc.predict(xval)
    return accuracy_score(yval, preds)

    # return cross_val_score(SVC(random_state=42, **params), pd.concat((xtrain, xval)), pd.concat((ytrain, yval)), n_jobs=-1, cv=TimeSeriesSplit(), scoring='accuracy').mean()


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective_svc, n_trials=200)


#%%
c.print(f"Refit, TEST performance:", style="bold underline")
svc_model = SVC(random_state=42, **study.best_params)
svc_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)))
# rf_model.fit(xtrain, ytrain)  # re-create original model (optuna does not return...)
# rf_model.n_estimators = 200
# rf_model.fit(xval, yval)  # warm start adds new trees => aka. refits
preds = svc_model.predict(xtest)
print(f"{study.best_params} -> {study.best_value:.4f}")
eval_classification(ytest, preds)
print(classification_report(ytest, preds))
