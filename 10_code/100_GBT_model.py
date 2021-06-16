# %%
import numpy as np
import statsmodels.api as sm
import pandas as pd
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_classification
from rich.console import Console
c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
ticker = "INTC"

df: pd.DataFrame = load_and_join_for_modeling(ticker)

for i in range(1, 11):
    df[f'return_lag{i}'] = df['return'].shift(i)

df = df.fillna(0)

#%%
# ATTENTION
# df = df.assign(vader=df.vader.sample(frac=1, replace=False).values)  # use .values, else: unalignable axis = nans
######################################
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

ytrain = ytrain > 0
yval = yval > 0
ytest = ytest > 0


from lightgbm import LGBMClassifier, plot_importance

lgbm_model = LGBMClassifier(n_estimators=250)
lgbm_model.fit(xtrain, ytrain, eval_set=(xval, yval), eval_metric="auc", early_stopping_rounds=50)
preds = lgbm_model.predict(xval)
eval_classification(yval, preds)

#%%
plot_importance(lgbm_model)
#%%
import shap
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(xval)
shap.summary_plot(shap_values, xval)
