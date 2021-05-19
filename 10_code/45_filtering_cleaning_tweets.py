# %%
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]
REGEX_URL = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
REGEX_USERMENTION = r"\@[A-Za-z0-9_-]*"

#%%
ticker = 'INTC'

df = pd.read_parquet(root_path + f"00_source_data/parquets/{ticker}_tweets.parquet")

#%%
def shape_decorator(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        print(f"{'[' + fn.__name__ + ']':25} Input shape: {args[0].shape}", end='')
        result = fn(*args, **kwargs)
        print(f" ---> Ouput shape: {result.shape}")
        return result
    return wrapper

#%%
@shape_decorator
def filter_language(data: pd.DataFrame, language_to_keep: str) -> pd.DataFrame:
    return data.loc[df.language == language_to_keep]


@shape_decorator
def filter_n_cashtags(data: pd.DataFrame, max_n: int) -> pd.DataFrame:
    return data[data.cashtags.str.split(", ").fillna("").apply(len) <= max_n]


def delete_urls(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(tweet=data.tweet.str.replace(pat=REGEX_URL, repl="", regex=True))
    return data
    
    
def mention_to_generic(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(tweet=data.tweet.str.replace(pat=REGEX_USERMENTION, repl="@USER", regex=True))
    return data
#%%
clean: pd.DataFrame = (
    df
    .pipe(filter_language, language_to_keep="en")
    .pipe(filter_n_cashtags, max_n=4)
    .pipe(delete_urls)
    .pipe(mention_to_generic)
    
)


#%%
vader = SentimentIntensityAnalyzer()
sample = clean.sample(100)
sentis = [vader.polarity_scores(text)['compound'] for text in sample.tweet]
sample['senti'] = sentis
sample = sample[['senti', 'tweet']]
#%%
