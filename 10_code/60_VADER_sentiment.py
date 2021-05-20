#%%
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from joblib.parallel import Parallel, delayed
root_path = "../"
vader = SentimentIntensityAnalyzer()
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


#%%
@delayed
def make_senti(id_, tweet):
    return id_, vader.polarity_scores(tweet).get('compound')

for ticker in tickers:
    print(f"{ticker}...")
    df = pd.read_parquet(f"../20_outputs/clean_tweets/{ticker}_clean.parquet")

    parallel = Parallel(n_jobs=-1, prefer='processes')(make_senti(id_, tweet) for _, id_, tweet in df[['id', 'tweet']].itertuples())

    senti_df = pd.DataFrame(parallel, columns=['id', 'vader'])
    # use later:
    result = pd.merge(df, senti_df, how='left', on='id', validate="1:1")
    senti_df.to_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")
