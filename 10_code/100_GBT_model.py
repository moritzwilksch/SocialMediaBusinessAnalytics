# %%
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

SENTI = 'ml_sentiment'

df: pd.DataFrame = load_and_join_for_modeling(ticker, SENTI)


df.label = df.label > 0


for i in range(1, 11):
    df[f'return_lag{i}'] = df['return'].shift(i)    # TODO: Feature Engineering?
    df[f'ma_{i}'] = df['return'].rolling(i).mean()  # TODO: Feature Engineering?

df = df.fillna(0)

#%%

# %%
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

# ytrain = ytrain > 0
# yval = yval > 0
# ytest = ytest > 0


def objective(trial):
    alpha = trial.suggest_float('alpha', 0, 10)
    lambda_ = trial.suggest_float('lambda_', 0, 10)
    num_leaves = trial.suggest_int('num_leaves', 3, 33, 3)
    n_estimators = trial.suggest_int('n_estimators', 10, 300, 10)
    lgbm_model = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, reg_alpha=alpha, reg_lambda=lambda_, random_state=42)
    lgbm_model.fit(xtrain, ytrain, eval_set=(xval, yval), eval_metric="auc", early_stopping_rounds=50)
    preds = lgbm_model.predict(xval)

    return accuracy_score(yval, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

#%%
print(f"{study.best_params} -> {study.best_value:.4f}")
#%%
c.print(f"Refit, TEST performance:", style="bold underline")
lgbm_model = LGBMClassifier(n_estimators=study.best_params['n_estimators'], num_leaves=study.best_params['num_leaves'], reg_alpha=study.best_params['alpha'], reg_lambda=study.best_params['lambda_'], random_state=42)
lgbm_model.fit(pd.concat((xtrain, xval)), pd.concat((ytrain, yval)), eval_metric="auc")
preds = lgbm_model.predict(xtest)
eval_classification(ytest, preds)
print(classification_report(ytest, preds))

# %%
plot_importance(lgbm_model)
# %%
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(xtest)
shap.summary_plot(shap_values, xtest)

# %%
imp = permutation_importance(lgbm_model, xtest, ytest)
pd.DataFrame({'ft': xtrain.columns, 'imp': imp['importances_mean']}).sort_values(by='imp')
# TODO: why negative imp?
# %%
plot_partial_dependence(lgbm_model, xtest, ['ml_sentiment'])