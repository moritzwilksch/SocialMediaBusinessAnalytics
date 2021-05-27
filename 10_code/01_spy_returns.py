#%%
import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'CMU Sans Serif'

#%%
spy = yf.Ticker('SPY').history(start="2008-01-01", end="2021-01-01")
returns = spy['Close'].pct_change().dropna()


#%%

mle = stats.norm.fit(returns)
mle_norm = stats.norm(*mle)

#%%
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(returns, ax=ax, ec='k', binwidth=0.005, stat='density', color='blue')
x = np.arange(returns.min(), returns.max(), 0.001)
y = mle_norm.pdf(x)
ax.plot(x, y, color='red')
sns.despine()
ax.text(0.01, 40, "Daily return of \$SPY", color='blue', family='CMU Sans Serif')
ax.text(0.025, 15, "Best fit normal distribution", color='red', family='CMU Sans Serif')
ax.set_xlabel("Daily return", family='CMU Sans Serif')
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15])
ax.set_xticklabels("–0.10 –0.05 0 0.05 0.10 0.15".split())
plt.savefig("../30_results/plots/spy_returns.png", dpi=200, facecolor='white')