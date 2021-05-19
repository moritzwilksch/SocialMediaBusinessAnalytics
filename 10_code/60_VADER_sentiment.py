#%%
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from joblib.parallel import Parallel, delayed

vader = SentimentIntensityAnalyzer()
df = pd.read_parquet("../20_outputs/parquets/INTC_tweets.parquet")
#%%
tweets = df.tweet.sample(25000).tolist()

#%%
@delayed
def make_senti(tweet):
    return vader.polarity_scores(tweet).get('compound')
#%%
%%timeit
[vader.polarity_scores(tweet).get('compound') for tweet in tweets]

#%%
%%timeit
Parallel(n_jobs=-1, prefer='processes')(make_senti(tweet) for tweet in tweets)