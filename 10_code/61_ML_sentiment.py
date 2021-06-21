# %%
import optuna
import nltk
import joblib
import string
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.stats as stats
import numpy as np
import pandas as pd
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


df.tweet = df.tweet.str.replace("\d+", "NUM", regex=True)
df.tweet = df.tweet.str.replace("\$\w+", "TICKER", regex=True)
df.tweet = df.tweet.str.replace("#\w+", "HASHTAG", regex=True)


df = df.assign(vader=df.tweet.apply(get_vader_senti))

# %%

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
pd.crosstab(df.sentiment, df.vader_bin).to_csv(root_path + "30_results/vader_acc.csv", sep=";")

# %%
print(f"VADER accuracy = {(df.vader_bin == df.sentiment).mean():.3f}")
# %%
ps = nltk.stem.PorterStemmer()


def prep(tweet):
    """Per word: to lower, stem, remove punctuation (keep emojies) """
    return " ".join([ps.stem(x.lower().strip(string.punctuation + """”'’""")) for x in tweet.split(' ')])

df.tweet = df.tweet.apply(prep)

# cv = CountVectorizer(token_pattern=r'[^\s]+')
cv = TfidfVectorizer(token_pattern=r'[^\s]+')
bow = cv.fit_transform(df.tweet)

# Check vocabulary:
with open(root_path + "20_outputs/cv_vocab.txt", 'w') as f:
    for v in cv.vocabulary_.keys():
        f.writelines(v + "\n")

# %%


def objective_lr(trial):
    c = trial.suggest_float('c', 1e-5, 10, log=True)
    cv_scores = cross_val_score(LogisticRegression(C=c), X=bow, y=df.sentiment, n_jobs=-1, cv=5)
    return cv_scores.mean()


def objective_nb(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 10, log=True)
    cv_scores = cross_val_score(MultinomialNB(alpha=alpha), X=bow, y=df.sentiment, n_jobs=-1, cv=5)
    return cv_scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective_lr, n_trials=100)

# %%
lr = LogisticRegression(C=study.best_params['c'], max_iter=150)
lr.fit(bow, df.sentiment)


# %%
if False:
    print("=== Random Forest ===")
    rs_rf = GridSearchCV(
        RandomForestClassifier(n_estimators=100, random_state=123),
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': [50, 100, 150, 200, 250, 300, 500, 1000],

        },
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    rs_rf.fit(bow, df.sentiment)
    print(f"{rs_rf.best_params_} -> {rs_rf.best_score_:.4f}")

# %%
if False:
    joblib.dump(cv, root_path + "20_outputs/count_vectorizer.joblib")
    joblib.dump(rs_rf, root_path + "20_outputs/sentiment_model.joblib")


# %%

cv: CountVectorizer = joblib.load(root_path + "20_outputs/count_vectorizer.joblib")
model: GridSearchCV = joblib.load(root_path + "20_outputs/sentiment_model.joblib")

# %%
ticker = 'INTC'
df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")


# %%
preds = lr.predict(bow)
df['preds'] = preds
for row in df.sample(100).itertuples():
    print(f"[{row.preds}]: {row.tweet}")
    print("-"*80)


# %%
rev = {v: k for k, v in cv.vocabulary_.items()}
for i in lr.coef_[1].argsort()[-20:]:
    print(rev[i])
