# %%
import pandas as pd
from rich.console import Console
c = Console()

root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

keep_cols = ['id',
             'created_at',
             'date',
             'time',
             'user_id',
             'username',
             'name',
             'tweet',
             'language',
             'replies_count',
             'retweets_count',
             'likes_count',
             'hashtags',
             'cashtags',
             'link']

#%%
for ticker in tickers:
    c.print(f"[LOADING] ${ticker}...", style='white on blue')
    df = pd.read_csv(
        root_path + f"00_source_data/{ticker}_tweets.csv",
        usecols=keep_cols
    )


    def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
        c.print(f"[PROCESSING] Fixing dtypes...", style='white on blue')
        data = data.assign(created_at=pd.to_datetime(data.date + " " + data.time))
        data = data.drop(['date', 'time'], axis=1)

        string_cols = ['username', 'name', 'tweet', 'link']
        data[string_cols] = data[string_cols].astype('string')

        data['language'] = data['language'].astype('category')
        return data


    def clean_object_cols(data: pd.DataFrame) -> pd.DataFrame:
        c.print(f"[PROCESSING] Cleaning obj-cols...", style='white on blue')
        data['hashtags'] = data.hashtags.apply(lambda x: ", ".join(x.split("', '")).strip("[']")).astype('string')
        data.loc[data.hashtags.str.len() == 0, 'hashtags'] = pd.NA

        data['cashtags'] = data.cashtags.apply(lambda x: ", ".join(x.split("', '")).strip("[']")).astype('string')
        data.loc[data.cashtags.str.len() == 0, 'cashtags'] = pd.NA

        return data


    def drop_dupes(data: pd.DataFrame) -> pd.DataFrame:
        """Drop dupes."""
        len_before = len(data)
        data = data.drop_duplicates(subset='id')
        len_after = len(data)

        c.print(f"[PROCESSING] Dropped {len_before - len_after} duplicates...", style='white on blue')

        return data



    clean: pd.DataFrame = (
        df
        .pipe(fix_dtypes)
        .pipe(drop_dupes)
        .pipe(clean_object_cols)
        # .query("language == 'en'") # filtered later
    )

    c.print(f"[SAVING] To parquet...", style='white on blue')
    clean.to_parquet(root_path + f"00_source_data/parquets/{ticker}_tweets.parquet")
    c.print(f"[DONE]", style='white on green')
    del df
