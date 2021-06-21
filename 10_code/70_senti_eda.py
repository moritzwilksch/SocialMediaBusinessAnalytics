# %%
from matplotlib.ticker import FuncFormatter
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

# %%
SENTI = 'ml_sentiment'
results = dict()
for ticker in tickers:
    print(f"{ticker}...")

    df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    senti_df = pd.read_parquet(root_path + f"20_outputs/{'vader_sentiments' if SENTI == 'vader' else 'ml_sentiments'}/{ticker}_sentiment.parquet")

    result = pd.merge(df, senti_df, how='left', on='id', validate="1:1")
    results[ticker] = result

# %%

SMOOTHED = False

fig, axes = plt.subplots(5, 2, figsize=(25, 20), sharex=True)

for idx, (ticker, _) in enumerate(results.items()):
    curr_ax = axes[idx//2][idx % 2]
    if SMOOTHED:
        senti_ts = results[ticker].groupby(results[ticker].created_at.dt.date)[SENTI].mean().rolling(7).mean()
    else:
        senti_ts = results[ticker].groupby(results[ticker].created_at.dt.date)[SENTI].mean()
    sns.lineplot(x=senti_ts.index, y=senti_ts, ax=curr_ax, color=colors.get(ticker))
    curr_ax.set_title(f"{ticker}", size=18, family='Arial', weight='bold')
    curr_ax.set_ylabel("")
    curr_ax.set_xlabel("Date", size=16)
fig.suptitle(f"Mean {SENTI} Sentiment per Day", size=20, family='Arial', weight='bold')
sns.despine()
plt.tight_layout()

plt.savefig(root_path + f"30_results/plots/sentiment_over_time{'_smoothed' if SMOOTHED else ''}_{SENTI}.png", dpi=200, facecolor='white')
plt.close()

# %% [markdown]
# # Sentiment vs. Return

# %%

for ticker in tickers:
    x = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet").dropna()['Close']
    ret = x.pct_change().dropna()
    df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
    senti = pd.read_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")
    result = pd.merge(df, senti, how='left', on='id', validate="1:1")

    senti_ts = result.groupby(result.created_at.dt.date)['vader'].mean()

    df = pd.merge(ret, senti_ts, left_index=True, right_index=True, how='left')

    # sns.lmplot(data=df, x='vader', y='Close')

    print(df.corr())

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(x/x[0]-1, color='blue', zorder=10)
    ax.plot(senti_ts, color='0.8', zorder=0)
    ax.set_title(ticker)
    plt.show()

# %% [markdown]
# # NVDA only
 # %%
ticker = 'NVDA'
x = pd.read_parquet(root_path + f"20_outputs/financial_ts/{ticker}_stock.parquet").dropna()['Close']
ret = x.pct_change().dropna()
df = pd.read_parquet(root_path + f"20_outputs/clean_tweets/{ticker}_clean.parquet")
senti = pd.read_parquet(root_path + f"20_outputs/vader_sentiments/{ticker}_sentiment.parquet")
result = pd.merge(df, senti, how='left', on='id', validate="1:1")

senti_ts = result.groupby(result.created_at.dt.date)['vader'].mean()

df = pd.merge(ret, senti_ts, left_index=True, right_index=True, how='left')



# %%
fg = sns.displot(senti['vader'], aspect=2, color='#77b900', kind='kde', fill=True)
fg.set_xlabels("Sentiment Score")
fg.ax.set_yticks([0, 1, 2, 3])





# %%

sns.set_context('talk')
fg = sns.lmplot(data=df, x='vader', y='Close', aspect=1.5, scatter_kws=dict(color='#77b900', alpha=0.5), line_kws=dict(color='darkgreen'))
fg.set_xlabels("Daily Average Sentiment Score")
fg.set_ylabels("Stock Return")
fg.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
fg.ax.text(0, 0.1, f"$\\rho = ${df.corr().loc['Close', 'vader']:.2f}")
