# %%
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from lightgbm import LGBMClassifier, plot_importance
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
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
# SENTI = 'vader'

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

num_fts = [col for col in df.columns.drop('label') if col not in cat_fts]
# %%
xtrain, ytrain, xval, yval, xtest, ytest = train_val_test_split(df)

# %%
ss = StandardScaler()
ss.fit(xtrain[num_fts])

#%%
xtrain_ss = xtrain.copy()
xtrain_ss[num_fts] = ss.transform(xtrain[num_fts])
xval_ss = xval.copy()
xval_ss[num_fts] = ss.transform(xval[num_fts])
xtest_ss = xtest.copy()
xtest_ss[num_fts] = ss.transform(xtest[num_fts])

#%%
tf.random.set_seed(42)
DROPOUT = 0.2
net = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


net.compile(tf.keras.optimizers.Adam(3e-4), 'binary_crossentropy', metrics=['accuracy'],)

hist = net.fit(
    x=xtrain.values,
    y=ytrain.values.astype('int'),
    batch_size=32,
    validation_data=(xval.values, yval.values.astype('int')),
    epochs=100 
)

# pd.DataFrame({'train_loss': hist.history['loss'], 'val_loss': hist.history['val_loss']}).plot()
pd.DataFrame({'acc': hist.history['accuracy'], 'validation': hist.history['val_accuracy']}).plot()

#%%

hist = net.fit(
    x=np.vstack((xtrain.values, xval.values)),
    y=np.hstack((ytrain.values, yval.values)),
    batch_size=32,
    epochs=100
)

pd.DataFrame({'train_loss': hist.history['loss'],}).plot()


print(classification_report(ytest, net.predict_classes(xtest.values)))
