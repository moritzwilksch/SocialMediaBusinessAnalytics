#%%
# %%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from rich.console import Console
c = Console(highlight=False)

ticker = "INTC"

# SENTI = 'ml_sentiment'
SENTI = 'vader'

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


df = df.assign(days_in_trend=days_in_trend(df['return']))
df = df.assign(dow=df.index.weekday)
df = df.assign(moy=df.index.month)
cat_fts = ['dow', 'moy']
df[cat_fts] = df[cat_fts].astype('category')
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

#%%


def one_run(ytrain, yval, ytest):
    # ytrain, yval, ytest = ytrain.copy(), yval.copy(), ytest.copy()
    ytrain = ytrain.sample(frac=1).reset_index(drop=True)
    yval = yval.sample(frac=1).reset_index(drop=True)
    yval = yval.sample(frac=1).reset_index(drop=True)


    rf_model = RandomForestClassifier(
        max_depth=study.best_params['max_depth'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        ccp_alpha=study.best_params['ccp_alpha'],
        random_state=42,
        n_jobs=-1,
        # warm_start=True
    )
    rf_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)))
    # rf_model.fit(xtrain, ytrain)  # re-create original model (optuna does not return...)
    # rf_model.n_estimators = 200
    # rf_model.fit(xval, yval)  # warm start adds new trees => aka. refits
    preds = rf_model.predict(xtest)
    return accuracy_score(ytest, preds)

#%%
Parallel(n_jobs=-1)(delayed(one_run)(ytrain.copy(), yval.copy(), ytest.copy()) for i in range(100))
