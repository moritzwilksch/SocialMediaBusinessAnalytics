# %%
from gc import callbacks
import optuna
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
df = pd.read_csv(root_path + "00_source_data/SRS_sentiment_labeled.csv", index_col='Unnamed: 0')
df


######################################################################
# %%
df = replace_with_generics(df)  # replace numbers, hashtags, tickers
df.tweet = df.tweet.apply(prepare_tweet_for_sentimodel)

cv = CountVectorizer(token_pattern=r'[^\s]+')
# cv = TfidfVectorizer(token_pattern=r'[^\s]+')
bow = cv.fit_transform(df.tweet)

# Check vocabulary:
# with open(root_path + "20_outputs/cv_vocab.txt", 'w') as f:
#     for v in cv.vocabulary_.keys():
#         f.writelines(v + "\n")

#%%
xtrain, xval, ytrain, yval = train_test_split(bow, df.sentiment, shuffle=True, random_state=42)
xtrain, xval = xtrain.toarray(), xval.toarray()
ytrain = np.eye(3)[ytrain.values]
yval = np.eye(3)[yval.values]

ss = StandardScaler()
ss.fit(xtrain)
xtrain = ss.transform(xtrain)
xval = ss.transform(xval)
#%%
import tensorflow as tf

def build_net():
    DROPOUT = 0.5
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dropout(DROPOUT),
        # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'),
        # tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    net.compile(tf.keras.optimizers.Adam(3e-4), 'categorical_crossentropy', metrics=['accuracy'])
    return net

#%%
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_net, epochs=3,)

cross_val_score(model, bow.toarray(), df.sentiment, cv=KFold(5, shuffle=True))

#%%
net = build_net()
hist = net.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=25, batch_size=8)
pd.DataFrame({'acc': hist.history['accuracy'], 'validation': hist.history['val_accuracy']}).plot()


#%%
preds = np.argmax(net.predict(xval), axis=1)
print(classification_report(np.argmax(yval, axis=1), preds))

# %%
if False:
    print("=== Random Forest ===")
    rs_rf = GridSearchCV(
        RandomForestClassifier(n_estimators=100, random_state=123),
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': [50, 100, 150, 200, 250, 300, 500, 1000],

        },
        cv=KFold(5, shuffle=True),
        n_jobs=-1,
        scoring="accuracy"
    )

    rs_rf.fit(bow, df.sentiment)
    print(f"{rs_rf.best_params_} -> {rs_rf.best_score_:.4f}")

# %%
if False:
    joblib.dump(cv, root_path + "20_outputs/count_vectorizer.joblib")
    joblib.dump(lr, root_path + "20_outputs/sentiment_model.joblib")
