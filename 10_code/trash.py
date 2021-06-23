# %%
import optuna
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers.preprocessing import prepare_tweet_for_sentimodel, replace_with_generics
vader = SentimentIntensityAnalyzer()
from lightgbm import LGBMClassifier, LGBMModel

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
df = pd.read_csv(root_path + "00_source_data/SRS_sentiment_labeled.csv", index_col='Unnamed: 0')
df

# %%


def get_vader_senti(tweet):
    return vader.polarity_scores(tweet).get('compound')


# df.tweet = df.tweet.str.replace("\d+", "NUM", regex=True)
# df.tweet = df.tweet.str.replace("\$\w+", "TICKER", regex=True)
# df.tweet = df.tweet.str.replace("#\w+", "HASHTAG", regex=True)

df = replace_with_generics(df)  # replace numbers, hashtags, tickers

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
pd.crosstab(df.sentiment, df.vader_bin).to_csv(root_path + "30_results/vader_acc.csv", sep=";")
#%%
print(f"VADER accuracy = {(df.vader_bin == df.sentiment).mean():.3f}")

######################################################################
# %%
df.tweet = df.tweet.apply(prepare_tweet_for_sentimodel)

# cv = CountVectorizer(token_pattern=r'[^\s]+')
cv = TfidfVectorizer(token_pattern=r'[^\s]+')
bow = cv.fit_transform(df.tweet)

# Check vocabulary:
# with open(root_path + "20_outputs/cv_vocab.txt", 'w') as f:
#     for v in cv.vocabulary_.keys():
#         f.writelines(v + "\n")

# %%
def objective_gbm(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    alpha = trial.suggest_float('alpha', 0, 25)
    lambda_ = trial.suggest_float('lambda_', 0, 25)
    subsample = trial.suggest_float('subsample', 0, 1)
    max_depth = trial.suggest_int('max_depth', 2, 32, 1)
    learning_rate = trial.suggest_float('learning_rate', 1e-1, 0.5, log=True)
    num_leaves = trial.suggest_int('num_leaves', 2, 32, 1)
    n_estimators = trial.suggest_int('n_estimators', 2, 300, 1)
    lgbm_model = LGBMClassifier(
        boosting_type=boosting_type,
        subsample=subsample,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        reg_alpha=alpha,
        reg_lambda=lambda_,
        random_state=42,
        n_jobs=-1
    )

    cv_scores = cross_val_score(lgbm_model, X=bow, y=df.sentiment, n_jobs=-1, cv=5, scoring='accuracy')
    return cv_scores.mean()



study = optuna.create_study(direction='maximize')
study.optimize(objective_gbm, n_trials=50)

#%%
model = LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1)
model.fit(bow, df.sentiment)

#%%
rev = {v: k for k, v in cv.vocabulary_.items()}
for i, w in enumerate(['NEGATIVE', 'NEUTRAL', 'POSITIVE']):
    print(f"======={'='*12}=======")
    print(f"======= {w:^10} =======")
    print(f"======={'='*12}=======")
    for i in model.coef_[i].argsort()[-20:]:
        print("- " + rev[i])
