# %%
from typing import Tuple
import pandas as pd
from functools import partial
import numpy as np
from rich.console import Console
root_path = "../"
VADER_THRESH = 0.05
c = Console(highlight=False)




def load_and_join_for_modeling(ticker: str) -> pd.DataFrame:
    """
    For `ticker`:
    - loads tweet & senti data
    - calculates features
    - joins it
    """
    # loading
    tweet_df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    senti_df = pd.read_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")

    prices = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet")
    prices.index = pd.DatetimeIndex(prices.index)

    result = pd.merge(tweet_df, senti_df, how='left', on='id', validate="1:1")

    result = result.assign(impressions=result.retweets_count + result.likes_count + 1)  # base value 1 to not loose tweets without likes

    # grouping daily
    def weighted_mean(x):
        """ Calculates mean weighted by impressions. """
        return np.average(x, weights=result.loc[x.index, "impressions"])

    daily_senti_ts = senti_df.groupby(result.created_at.dt.date).agg(vader=('vader', weighted_mean))
    # daily_senti_ts = senti_df.groupby(result.created_at.dt.date)['vader'].mean()  # OLD
    daily_senti_ts.index = pd.DatetimeIndex(daily_senti_ts.index)

    # returns
    returns = prices['Close'].pct_change()
    returns[returns == 0] = pd.NA

    # calculation of metrics
    pct_pos = (
        result
        .assign(is_pos=result.vader > VADER_THRESH)
        .groupby(result.created_at.dt.date)
        ['is_pos']
        .mean()
    )
    pct_pos.index = pd.DatetimeIndex(pct_pos.index)

    pct_neg = (
        result
        .assign(is_neg=result.vader < -VADER_THRESH)
        .groupby(result.created_at.dt.date)
        ['is_neg']
        .mean()
    )
    pct_neg.index = pd.DatetimeIndex(pct_neg.index)

    # produce final df through concat
    final: pd.DataFrame = pd.concat(
        [
            daily_senti_ts,
            pct_pos.rename("pct_pos"),
            pct_neg.rename("pct_neg"),
            prices["Volume"].fillna(0).rename('volume'),
            tweet_df.groupby(tweet_df.created_at.dt.date)['id'].count().rename('num_tweets'),
            returns.rename('return').fillna(0),  # return is zeroed on week ends
            returns.bfill().shift(-1).rename('label')  # label is bfilled
        ],
        axis=1)

    final.index = pd.DatetimeIndex(final.index)

    # backfill prediction label only
    final['label'] = final['label'].bfill()

    # Test for join errors...
    col_obj_mapping = {
        # 'vader': daily_senti_ts,  # Test for UNweighted equality
        'pct_pos': pct_pos,
        'pct_neg': pct_neg,
        'volume': prices['Volume'].fillna(0),
        'return': returns.fillna(0),
    }

    test_dates = ['2019-06-30', '2019-01-18',
                  '2019-03-07', '2019-06-16',
                  '2019-08-30', '2019-12-09',
                  '2020-07-20', '2019-12-29',
                  '2020-10-06', '2019-05-05']

    for date in test_dates:
        assert all([final.loc[date][col] == col_obj_mapping.get(col).loc[date] for col in ['pct_pos', 'pct_neg', 'volume', 'return']])

    return final


def train_val_test_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Given data returns xtrain, ytrain, xval, yval, xtest, ytest for 596/128/128 split """
    TRAIN_VAL_CUTOFF = "2020-08-18"
    VAL_TEST_CUTOFF = "2020-12-24"

    train = data.loc[:TRAIN_VAL_CUTOFF]
    val = data.loc[pd.to_datetime(TRAIN_VAL_CUTOFF) + pd.DateOffset(1):VAL_TEST_CUTOFF]
    test = data.loc[pd.to_datetime(VAL_TEST_CUTOFF) + pd.DateOffset(1):]

    return train.drop('label', axis=1), train['label'], val.drop('label', axis=1), val['label'], test.drop('label', axis=1), test['label']


# %%
if __name__ == "__main__":
    root_path = "../../"
    df = load_and_join_for_modeling('INTC')

# %%
