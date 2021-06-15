# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
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

# %% [markdown]
"""
# Number of Tweets per Day
---
"""
# %%
dfs = []
for ticker in tickers:
    print(f"Loading {ticker}...")
    df = pd.read_parquet(root_path + f"00_source_data/parquets/{ticker}_tweets.parquet")
    dfs.append(df.groupby(df.created_at.dt.date)['id'].count())
    del df

# %%
df = pd.concat(dfs, axis=1).set_axis(tickers, axis=1)

# df = (df/df.iloc[0]).reset_index()  # Normalized to t=0
df = df.reset_index()
df = df[~(df.created_at == pd.to_datetime("2021-05-01"))]

# %%


def k_formatter(x, pos):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x//100 * 100:.0f}" if x > 0 else "0"


fig, axes = plt.subplots(5, 2, figsize=(25, 20), sharex=True)
for idx, ticker in enumerate(tickers):
    curr_ax = axes[idx//2][idx % 2]
    sns.lineplot(data=df, x='created_at', y=ticker, ax=curr_ax, legend=False, color=colors[ticker], lw=1.2)
    curr_ax.set_title(f"{ticker} ($n={df.sum()[ticker]:,.0f}$)", size=18, family='Arial', weight='bold')
    curr_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    curr_ax.xaxis.set_major_formatter(DateFormatter("%b %y"))
    curr_ax.set_ylabel("")
    curr_ax.set_xlabel("Date")
sns.despine()
fig.suptitle("Daily Number of Tweets", size=20, family='Arial', weight='bold')
plt.tight_layout()

plt.savefig(root_path + "30_results/plots/ntweets_perday.png", dpi=200, facecolor='white')
plt.close()


# %% [markdown]
"""
# Number of Cashtags per Tweet
---
"""

# %%
series = []
for ticker in tickers[:1]:
    print(f"Loading {ticker}...")
    df = pd.read_parquet(root_path + f"00_source_data/parquets/{ticker}_tweets.parquet")
    df = df.loc[~(df.cashtags.isna())]
    series.append(df.cashtags.str.split(", ").apply(len))
    # del df

# %%
fig, axes = plt.subplots(5, 2, figsize=(25, 20), sharex=True)

for idx, s in enumerate(series):
    curr_ax = axes[idx//2][idx % 2]
    sns.histplot(s, binwidth=1, ax=curr_ax, color=colors.get(tickers[idx]))
    curr_ax.set_title(f"{tickers[idx]}", size=18, family='Arial', weight='bold')
    curr_ax.set_ylabel("")
    curr_ax.set_xlabel("Number of Cashtags", size=16)
fig.suptitle("Number of Cashtags per Tweet", size=20, family='Arial', weight='bold')
sns.despine()
plt.tight_layout();

plt.savefig(root_path + "30_results/plots/ncashtags_pertweet.png", dpi=200, facecolor='white')
plt.close()

#%%
