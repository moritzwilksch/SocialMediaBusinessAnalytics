# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, plot_partial_dependence
import optuna
import shap
from lightgbm import LGBMClassifier, plot_importance
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
import pandas as pd
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_classification
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
ticker = "INTC"

SENTI = 'vader'
# SENTI = 'vader'

# close_price = pd.read_parquet(root_path + "20_outputs/financial_ts/INTC_stock.parquet")['Close'].ffill()
df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)


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

# df.dow = df.dow.map(df.groupby('dow')['label'].mean())
# df.moy = df.moy.map(df.groupby('moy')['label'].mean())


# %%
pd.crosstab(df.moy, df.label)

# %%
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

# ytrain = ytrain > 0
# yval = yval > 0
# ytest = ytest > 0


def objective_gbm(trial):
    alpha = trial.suggest_float('alpha', 0, 25)
    lambda_ = trial.suggest_float('lambda_', 0, 25)
    subsample = trial.suggest_float('subsample', 0, 1)
    learning_rate = trial.suggest_float('learning_rate', 1e-1, 0.5, log=True)
    num_leaves = trial.suggest_int('num_leaves', 2, 32, 3)
    n_estimators = trial.suggest_int('n_estimators', 2, 300, 1)
    lgbm_model = LGBMClassifier(
        subsample=subsample,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        reg_alpha=alpha,
        reg_lambda=lambda_,
        random_state=42
    )
    lgbm_model.fit(xtrain, ytrain, eval_set=(xval, yval), verbose=False)
    preds = lgbm_model.predict(xval)

    return accuracy_score(yval, preds)


# %%
def objective_rf(trial):
    max_depth = trial.suggest_int('max_depth', 2, 32, 3)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20, 1)
    max_features = trial.suggest_float('max_features', 0, 1)
    rf = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(xtrain, ytrain)
    preds = rf.predict(xval)

    return accuracy_score(yval, preds)


# %%
MODEL = 'rf'
if MODEL == 'lgbm':
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_gbm, n_trials=250)

    c.print(f"Refit, TEST performance:", style="bold underline")
    lgbm_model = LGBMClassifier(
        subsample=study.best_params['subsample'],
        learning_rate=study.best_params['learning_rate'],
        n_estimators=study.best_params['n_estimators'],
        num_leaves=study.best_params['num_leaves'],
        reg_alpha=study.best_params['alpha'],
        reg_lambda=study.best_params['lambda_'],
        random_state=42
    )
    # lgbm_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)))
    # preds = lgbm_model.predict(xtest)
    lgbm_model.fit(xtrain, ytrain)
    refit = lgbm_model.booster_.refit(xval, yval)
    preds = refit.predict(xtest) > 0.5
    print(f"{study.best_params} -> {study.best_value:.4f}")
    eval_classification(ytest, preds)
    print(classification_report(ytest, preds))
elif MODEL == 'rf':
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_rf, n_trials=250)

    c.print(f"Refit, TEST performance:", style="bold underline")
    rf_model = RandomForestClassifier(
        max_depth=study.best_params['max_depth'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )
    # rf_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)))
    rf_model.fit(xtrain, ytrain)  # re-create original model (optuna does not return...)
    rf_model.fit(xval, yval)  # warm start adds new trees => aka. refits
    preds = rf_model.predict(xtest)
    print(f"{study.best_params} -> {study.best_value:.4f}")
    eval_classification(ytest, preds)
    print(classification_report(ytest, preds))


# %%
with open(root_path + f"20_outputs/benchmarks/{ticker}/{SENTI}/stats.log", 'w') as f:
    f.write(f"{MODEL}\n")
    f.write(f"{study.best_params}\nVALIDATION Accuracy = {study.best_value:.4f}\n")
    f.write(f"TEST Accuracy = {eval_classification(ytest, preds):.4f}" + "\n")
    f.write(classification_report(ytest, preds) + "\n")

# %%
if MODEL == 'rf':
    model = rf_model
elif MODEL == 'lgbm':
    model = lgbm_model

imp_df = pd.DataFrame([model.feature_importances_, xtrain.columns]).T.set_index(1).rename({0: 'importance'}, axis=1).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
palette = ['0.8'] * len(xtrain.columns)
palette[np.where(imp_df.index == SENTI)[0].item()] = '#00305e'
sns.barplot(y=imp_df.index, x=imp_df.importance/imp_df.importance.sum(), orient='h', palette=palette, ec='k')
ax.set_ylabel('Feature')
ax.set_xlabel('Feature Importance')
ax.set_title(f"{ticker} Feature Importance ({MODEL})", size=16, weight='bold')
sns.despine()
plt.tight_layout()
plt.savefig(root_path + f"20_outputs/benchmarks/{ticker}/{SENTI}/fi.png", dpi=200, facecolor='white')


# %%
plot_partial_dependence(model, xtest, [SENTI, 'pct_neg'])

# %%

# plot_precision_recall_curve(lgbm_model, xtest, ytest)
plot_roc_curve(model, xtest, ytest)


# %%

# %%
plot_partial_dependence(model, xval, ['moy'])
