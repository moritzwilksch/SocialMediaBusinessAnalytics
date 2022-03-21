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
from transformers import TF2_WEIGHTS_NAME
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
    df.tweet, df.sentiment + 1, shuffle=True, random_state=42
)

# ytrain, yval = np.eye(3)[ytrain.values], np.eye(3)[yval.values]


#%%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(xtrain)

trainvec = vectorizer.transform(xtrain).astype(np.float32)
valvec = vectorizer.transform(xval).astype(np.float32)


trainset = tf.data.Dataset.from_tensor_slices((trainvec.toarray(), ytrain))
valset = tf.data.Dataset.from_tensor_slices((valvec.toarray(), yval))
#%%
DROPOUT_RATE = 0.5
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=64, activation="swish", kernel_regularizer="l2"),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(units=32, activation="swish", kernel_regularizer="l2"),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(units=3, activation="softmax"),
    ]
)

model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.001),
    "sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    trainset.shuffle(1024).batch(64).prefetch(10),
    validation_data=valset.batch(512),
    batch_size=64,
    epochs=30,
)
