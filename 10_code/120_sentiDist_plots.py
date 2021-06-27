# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


senti = "vader"
ticker = "TSLA"

#%%
plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(figsize=(12, 6))

for ticker in tickers:
    senti_df = pd.read_parquet(root_path + f"20_outputs/{'vader_sentiments' if senti=='vader' else 'ml_sentiments'}/{ticker}_sentiment.parquet")
    sns.kdeplot(senti_df[senti], color='k', ax=ax, label=ticker)
    ax.set_xlabel("VADER Sentiment")
    sns.despine()
    # plt.legend(ncol=2)
for line in ax.get_lines():
    line.set_alpha(0.3)
# plt.show()
plt.savefig(root_path + "30_results/plots/all-senti-distsVADER.svg")
plt.close()



#%%

vcs = []
for ticker in tickers:
    senti_df = pd.read_parquet(root_path + f"20_outputs/ml_sentiments/{ticker}_sentiment.parquet")
    vcs.append(senti_df['ml_sentiment'].value_counts())

#%%
vcs_df = pd.DataFrame(vcs).T
vcs_df.columns = tickers
vcs_df = vcs_df/vcs_df.sum()
melted_df = vcs_df.T.melt()

plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(data=melted_df, x='variable', y='value', ci='sd', palette=['#B3001B', '0.7', '#8BCB52'], errcolor='k', ec='k', ax=ax, capsize=0.1)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_xlabel("Predicted Sentiment Class")
ax.yaxis.set_major_formatter(lambda _, x: f"{x*10:.0f}%")
ax.set_ylabel("Mean (SD) Percentage")
sns.despine()
plt.savefig(root_path + "30_results/plots/all-senti-distsML.svg")
plt.close()




#%%
