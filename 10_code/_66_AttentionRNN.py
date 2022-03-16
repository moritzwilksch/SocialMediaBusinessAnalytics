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
# tokenizer = tf.keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(xtrain.to_numpy())
# tokenizer = text.WhitespaceTokenizer()
# tokenizer.tokenize(xtrain)

vectorizer = tf.keras.layers.TextVectorization(output_sequence_length=128)
vectorizer.adapt(xtrain)

#%%

trainset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
valset = tf.data.Dataset.from_tensor_slices((xval, yval))

#%%
class AttentionRNN(tf.keras.Model):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vectorizer.vocabulary_size() + 1, output_dim=32, mask_zero=True
        )
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)
        # self.flatten1 = tf.keras.layers.GlobalAveragePooling1D()
        self.rnn = tf.keras.layers.GRU(128)
        self.dense1 = tf.keras.layers.Dense(units=128, activation="swish")
        self.out = tf.keras.layers.Dense(units=3, activation="softmax")

    def call(self, inputs):
        tokens = self.vectorizer(inputs)

        x = self.embedding(tokens)

        x = self.rnn(x)
        x = self.dense1(x)
        out = self.out(x)

        return out


model = AttentionRNN(vectorizer)
model.compile(
    tf.keras.optimizers.Adam(),
    tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

for x, y in trainset.batch(2).take(1):
    model(x)


tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./kerastrash/tb", histogram_freq=0, write_graph=True, write_images=True
)
model.fit(
    trainset.batch(64).prefetch(10),
    validation_data=valset.batch(512),  # .prefetch(10),
    epochs=10,
    callbacks=[tb_callback],
)

#%%
for x, y in valset.batch(1024):
    # print(vectorizer(x).shape)
    model(x)
    print(x, y)
