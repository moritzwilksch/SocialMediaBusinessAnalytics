# %%
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers.preprocessing import prepare_tweet_for_sentimodel, replace_with_generics
import tensorflow_text as text

vader = SentimentIntensityAnalyzer()

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
df = pd.read_csv(
    root_path + "00_source_data/SRS_sentiment_labeled.csv", index_col="Unnamed: 0"
)
df


######################################################################
# %%
df = replace_with_generics(df)  # replace numbers, hashtags, tickers
df.tweet = df.tweet.apply(prepare_tweet_for_sentimodel)


# %%
xtrain, xval, ytrain, yval = train_test_split(
    df.tweet, df.sentiment, shuffle=True, random_state=42
)

ytrain, yval = np.eye(3)[ytrain.values], np.eye(3)[yval.values]


#%%
df_all = pd.concat(
    [
        pd.DataFrame({"tweet": xtrain, "set": "train"}),
        pd.DataFrame({"tweet": xval, "set": "val"}),
    ]
)

_xtrain, _xval, _ytrain, _yval = train_test_split(df_all.tweet, df_all.set)

vectorizer = TfidfVectorizer()
vectorizer.fit(_xtrain)

feature_names = [
    t[1]
    for t in sorted(
        {v: k for k, v in vectorizer.vocabulary_.items()}.items(), key=lambda t: t[0]
    )
]
_xtrain = vectorizer.transform(_xtrain).astype(np.float32)
_xval = vectorizer.transform(_xval).astype(np.float32)
#%%
from lightgbm import LGBMClassifier
from lightgbm.plotting import plot_importance, plot_metric

validator = LGBMClassifier()
validator.fit(_xtrain, _ytrain, eval_set=(_xval, _yval), eval_metric=["auc"], feature_name=feature_names)
plot_metric(validator, "auc")
plot_importance(validator, max_num_features=20)
