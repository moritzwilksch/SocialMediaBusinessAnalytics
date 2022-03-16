# %%
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf

# import optuna
import joblib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers.preprocessing import prepare_tweet_for_sentimodel, replace_with_generics

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

# %%

VOCAB_SIZE = 7000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode="count"
)
encoder.adapt(xtrain.values)
# encoder(xtrain.iloc[0])


# %%


# %%
model = tf.keras.Sequential(
    [
        encoder,
        # tf.keras.layers.Embedding(input_dim=encoder.vocabulary_size()+1, output_dim=5, input_length=100, mask_zero=True),
        # tf.keras.layers.Dropout(0.25),
        # tf.keras.layers.Conv1D(filters=32, kernel_size=2),
        # tf.keras.layers.MaxPooling1D(),
        # tf.keras.layers.AveragePooling1D(),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.03)
        ),
        tf.keras.layers.Dense(
            units=32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.03)
        ),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

model.compile(
    tf.keras.optimizers.Adam(), "categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    xtrain,
    ytrain,
    validation_data=(xval, yval),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            "./kerastrash/checkpoint",
            "accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    ],
)


#%%
model.load_weights("./kerastrash/checkpoint")
preds = model.predict(xval).argmax(axis=1) - 1
print(classification_report(preds, yval.argmax(axis=1) - 1))


# %%

df_train = df.loc[xtrain.index]
df_val = df.loc[xval.index]

# %%
cv = TfidfVectorizer(token_pattern=r"[^\s]+")
bow_train = cv.fit_transform(df_train.tweet)
bow_val = cv.transform(df_val.tweet)

# %%
lr = LogisticRegression(C=3, max_iter=150)
lr.fit(bow_train, df_train.sentiment)

# %%
preds = lr.predict(bow_val)
print(classification_report(preds, df_val.sentiment))
