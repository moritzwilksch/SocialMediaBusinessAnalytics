# %%
import optuna
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers.preprocessing import prepare_tweet_for_sentimodel, replace_with_generics
vader = SentimentIntensityAnalyzer()

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
# pd.crosstab(df.sentiment, df.vader_bin).to_csv(root_path + "30_results/vader_acc.csv", sep=";")
#%%
print(f"VADER accuracy = {(df.vader_bin == df.sentiment).mean():.3f}")

#%%
# Random Guess CI
accs = []
for i in range(10000):
    guess = np.random.choice([-1, 0, 1], p=(555/3000, 1276/3000, 1169/3000), size=3000)
    accs.append((df.vader_bin == guess).mean())
accs = np.array(accs)

print(accs.mean().round(3), accs.std().round(5))

#%%
# VADER CI
accs = []
for i in range(10000):
    resample = df.sample(frac=1, replace=True)
    accs.append((resample['vader_bin'] == resample['sentiment']).mean())
accs = np.array(accs)
print(accs.mean().round(3), accs.std().round(5))

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
RETRAIN = True
if RETRAIN:
    def objective_lr(trial):
        c = trial.suggest_float('c', 1e-5, 10, log=True)
        cv_scores = cross_val_score(LogisticRegression(C=c), X=bow, y=df.sentiment, n_jobs=-1, cv=KFold(5, shuffle=True))
        return cv_scores.mean()


    def objective_nb(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 10, log=True)
        cv_scores = cross_val_score(MultinomialNB(alpha=alpha), X=bow, y=df.sentiment, n_jobs=-1, cv=KFold(5, shuffle=True))
        return cv_scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(objective_lr, n_trials=100)


    lr = LogisticRegression(C=study.best_params['c'], max_iter=150)
    lr.fit(bow, df.sentiment)
    model = lr
else:
    cv: TfidfVectorizer = joblib.load(root_path + "20_outputs/count_vectorizer.joblib")
    model: LogisticRegression = joblib.load(root_path + "20_outputs/sentiment_model.joblib")

#%%
# CV - Score for metrics
from sklearn.model_selection import cross_validate
cross_validate(LogisticRegression(C=3.1225, max_iter=150), bow, df.sentiment, cv=KFold(shuffle=True), scoring='precision_weighted')

#%%
rev = {v: k for k, v in cv.vocabulary_.items()}
for i, w in enumerate(['NEGATIVE', 'NEUTRAL', 'POSITIVE']):
    print(f"======={'='*12}=======")
    print(f"======= {w:^10} =======")
    print(f"======={'='*12}=======")
    for i in model.coef_[i].argsort()[-20:]:
        print("??? " + rev[i], end=' ')

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

