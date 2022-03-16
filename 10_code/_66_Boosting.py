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

vectorizer = TfidfVectorizer()
vectorizer.fit(xtrain)

trainvec = vectorizer.transform(xtrain).astype(np.float32)
valvec = vectorizer.transform(xval).astype(np.float32)

print(trainvec.shape)
#%%
from lightgbm import LGBMClassifier

clf = LGBMClassifier(n_estimators=1000)

clf.fit(
    trainvec,
    ytrain,
    eval_set=(valvec, yval),
    eval_metric=["multiclass"],
    early_stopping_rounds=50,
)

from sklearn.metrics import classification_report, confusion_matrix

pred = clf.predict(valvec)
print(classification_report(yval, pred))


#%%
from lightgbm.plotting import plot_importance

plot_importance(clf, max_num_features=20)

#%%
feature_names = [
    t[1]
    for t in sorted(
        {v: k for k, v in vectorizer.vocabulary_.items()}.items(), key=lambda t: t[0]
    )
]

#%%
import shap

explainer = shap.explainers.Tree(clf, feature_names=feature_names)
shap_values = explainer(trainvec.toarray())

#%%
shap.plots.bar(shap_values[:, :, 0], max_display=20)

#%%
shap.plots.bar(shap_values[:, :, 0], order=shap.Explanation.argsort.flip)


#%%
def entropy(arr) -> np.ndarray:
    return (-arr * np.log2(arr)).sum(axis=1)


probas = clf.predict_proba(valvec)
entropies = entropy(probas)

extremes = np.argsort(entropies)[-25:]
pd.DataFrame(
    {
        "text": xval.iloc[extremes],
        "entropy": entropies[extremes],
        "label": yval.iloc[extremes],
    }
)
