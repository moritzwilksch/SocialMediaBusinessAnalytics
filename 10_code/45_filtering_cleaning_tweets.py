# %%
import pandas as pd

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]
REGEX_URL = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
REGEX_USERMENTION = r"\@[A-Za-z0-9_-]*"

# %%


def shape_decorator(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        print(f"{'[' + fn.__name__ + ']':25} Input shape: {args[0].shape}", end='')
        result = fn(*args, **kwargs)
        print(f" ---> Ouput shape: {result.shape}")

        # logging
        with open(f"./logs/{ticker}_filtering.log", 'a') as f:
            f.write(f"{'[' + fn.__name__ + ']':25} Input shape: {args[0].shape}" + f" ---> Ouput shape: {result.shape}\n")
        return result
    return wrapper


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
    # data = data.assign(tweet=data.tweet.str.replace(pat=REGEX_USERMENTION, repl="@USER", regex=True))
    data = data.assign(tweet=data.tweet.str.replace(pat=REGEX_USERMENTION, repl="", regex=True))
    return data


# %%


for ticker in tickers:
    print(f"{ticker}...")
    df = pd.read_parquet(root_path + f"00_source_data/parquets/{ticker}_tweets.parquet")
    with open(f"./logs/{ticker}_filtering.log", 'w') as f:
        f.write("=" * 10 + ticker + "=" * 10 + "\n")

    clean: pd.DataFrame = (
        df
        .pipe(filter_language, language_to_keep="en")
        .pipe(filter_n_cashtags, max_n=4)
        .pipe(delete_urls)
        .pipe(mention_to_generic)
        .reset_index(drop=True)
    )

    clean.to_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
