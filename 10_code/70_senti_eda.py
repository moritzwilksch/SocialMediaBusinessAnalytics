#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

colors = {
    "TSLA": '#ff1b22',
    "AAPL": 'black',
    "AMZN": '#ff9900',
    "FB": '#1977f3',
    "MSFT": '#7fba00',
    "TWTR": '#1da1f2',
    "AMD": 'black',
    "NFLX": '#e50913',
    "NVDA": '#77b900',
    "INTC": '#0071c5'
}

#%%
results = dict()
for ticker in tickers:
    print(f"{ticker}...")

    df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    senti_df = pd.read_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")

    result = pd.merge(df, senti_df, how='left', on='id', validate="1:1")
    results[ticker] = result

#%%

SMOOTHED = False

fig, axes = plt.subplots(5, 2, figsize=(25, 20), sharex=True)

for idx, (ticker, _) in enumerate(results.items()):
    curr_ax = axes[idx//2][idx % 2]
    if SMOOTHED:
        senti_ts = results[ticker].groupby(results[ticker].created_at.dt.date)['vader'].mean().rolling(7).mean()
    else:
        senti_ts = results[ticker].groupby(results[ticker].created_at.dt.date)['vader'].mean()
    sns.lineplot(x=senti_ts.index, y=senti_ts, ax=curr_ax, color=colors.get(ticker))
    curr_ax.set_title(f"{ticker}", size=18, family='Arial', weight='bold')
    curr_ax.set_ylabel("")
    curr_ax.set_xlabel("Date", size=16)
fig.suptitle("Mean Sentiment per Day", size=20, family='Arial', weight='bold')
sns.despine()
plt.tight_layout()

plt.savefig(root_path + f"30_results/plots/sentiment_over_time{'_smoothed' if SMOOTHED else ''}.png", dpi=200, facecolor='white')
plt.close()
