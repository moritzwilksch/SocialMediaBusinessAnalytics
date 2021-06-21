# %%
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from rich.console import Console
from helpers.preprocessing import prepare_tweet_for_sentimodel, replace_with_generics

c = Console(highlight=False)
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

#%%
cv: TfidfVectorizer = joblib.load(root_path + "20_outputs/count_vectorizer.joblib")
model: LogisticRegression = joblib.load(root_path + "20_outputs/sentiment_model.joblib")

#%%
for ticker in tickers:
    print(f"[PROCESSING] {ticker}...")
    df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    df = replace_with_generics(df)
    df.tweet = joblib.Parallel(n_jobs=-1)(joblib.delayed(prepare_tweet_for_sentimodel)(tweet) for tweet in df.tweet)
    bow = cv.transform(df.tweet)

    preds = model.predict(bow)
    df['ml_sentiment'] = preds

    df[['id', 'ml_sentiment']].to_parquet(root_path + f"20_outputs/ml_sentiments/{ticker}_sentiment.parquet")
print("DONE!")