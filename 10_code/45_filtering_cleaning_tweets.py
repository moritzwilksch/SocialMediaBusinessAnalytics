# %%
import pandas as pd
from rich.console import Console
c = Console()

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]



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

#%%
clean = (
    df
    .pipe(filter_language, language_to_keep="en")
    .pipe(filter_n_cashtags, max_n=4)
)
