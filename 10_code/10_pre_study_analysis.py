# %%
from matplotlib import pyplot as plt
import pandas as pd
from rich.console import Console
import seaborn as sns
c = Console(highlight=False)

# %%
root_path = "../"  # IPyhton
# root_path = "./"  # CLI

with open(root_path + "00_source_data/sp500_tickers.txt") as f:
    a = f.read()

tickers = a.split("\n")

# %%
LOAD_FROM_DISK = True

if not LOAD_FROM_DISK:
    results = []
    for ticker in tickers:

        try:
            c.print(f"Loading {ticker}...")
            df = pd.read_csv(root_path + f"20_outputs/pre_study/{ticker}_tweets.csv")
            results.append(
                (ticker, len(df))
            )
        except FileNotFoundError:
            c.print(f"[ERROR] {ticker} not found!", style='white on red')

    # Not found: BIIB, MSI, PKI,
    df = pd.DataFrame(results, columns=['ticker', 'n_tweets'])
else:
    df = pd.read_parquet(root_path + "20_outputs/pre_study/pre_study_data.parquet")
# %%
df.sort_values(by='n_tweets', ascending=False).head(15)

# Top 15:
# |Symbol| #tweets|
# |:-----|-------:|
# | TSLA | 140359 |
# | AAPL |  50212 |
# | AMZN |  30716 |
# | FB   |  29097 |
# | MSFT |  19173 |
# | TWTR |  18781 |
# | AMD  |  18631 |
# | NFLX |  18023 |
# | NVDA |  14146 |
# | AAL  |  11530 |
# | BA   |  10817 |
# | F    |  10564 |
# | PENN |   9897 |
# | INTC |   9186 |
# | TEL  |   8795 |


# %%
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16

total = 15
used = 9

fig, ax = plt.subplots(figsize=(10, 6))
top_df = df.nlargest(total, 'n_tweets')
sns.barplot(
    data=top_df,
    y='ticker',
    x='n_tweets',
    ax=ax,
    palette=['#00305e'] * used + ['0.8'] * 4 + ['#00305e'] + ['0.8'] * (total - used),
    ec='k'
    )
ax.set(
    ylabel='',
    xticks=range(0, 150_001, 25_000),
    xticklabels=[f'{x/1000:.0f}{"k" if x > 0 else ""}' for x in range(0, 150_001, 25_000)],
)

ax.set_xlabel("Number of Tweets", weight='bold')

for idx, tup in enumerate(top_df.iloc[:14].itertuples()):
    if idx not in (9, 10, 11, 12):
        ax.text(x=tup.n_tweets, y=idx+0.25, s=f"{tup.n_tweets/1000:.0f}k", color='w', ha='right', size=14)
    #ax.text(x=0, y=idx+0.25, s=f"{tup.n_tweets/1000:.0f}k", color='w')


sns.despine()
plt.savefig(root_path + "30_results/plots/pre_study.svg")


