# %%
import tensorflow as tf
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from os import replace
from numpy.lib.function_base import average
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import statsmodels.api as sm
import pandas as pd
from tensorflow.python.keras.layers.core import Dropout
from helpers.preprocessing import load_and_join_for_modeling, train_val_test_split
from helpers.modeleval import eval_classification
from rich.console import Console
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
df = pd.read_csv(root_path + "00_source_data/SRS_sentiment_labeled.csv", index_col='Unnamed: 0')
df

# %%


def get_vader_senti(tweet):
    return vader.polarity_scores(tweet).get('compound')


df = df.assign(vader=df.tweet.apply(get_vader_senti))


# %%
df = df.assign(vader_bin=np.select(
    [
        df.vader <= -0.05,
        df.vader >= 0.05,
    ],
    [
        -1,
        1,
    ],
    default=0
))

# %%
pd.crosstab(df.sentiment, df.vader_bin)

# %%
tweets = df.tweet.copy()
tweets = tweets.str.replace("\d+", "NUM", regex=True)

# cv = CountVectorizer()
cv = TfidfVectorizer()
bow = cv.fit_transform(tweets)

# %%


net = tf.keras.Sequential([
    # tf.keras.layers.Dense(units=bow.shape[1]),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
    # tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
    # tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(units=3, activation='softmax'),
])

net.compile('adam', 'categorical_crossentropy')
ytrain = np.eye(3)[df.sentiment.iloc[:2400].values + 1]
yval = np.eye(3)[df.sentiment.iloc[2400:].values + 1]
net.fit(
    bow[:2400].toarray(),
    ytrain,
    validation_data=(bow[2400:].toarray(),
                     yval),
    epochs=25,
    batch_size=64,
    class_weight={0: 2, 1: 1, 2: 2}
)

preds = net.predict(bow[2400:].toarray()).argmax(axis=1) - 1

print(classification_report(df.sentiment.iloc[2400:], preds))
