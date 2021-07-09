# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14

# %%
real_models = pd.read_csv("../30_results/model_benchmarksNEW.csv", sep=";", header=1, index_col='senti').drop('ticker')
real_models

# %%
baseline = pd.read_csv("../30_results/naive_baseline_benchmarks.csv", sep=';', index_col="ticker")['accuracy']
baseline

# %%
######################## Tree-based Models ################################
nosenti_models = pd.read_csv("../30_results/model_benchmarks_NOSENTI-NEW.csv", sep=";", header=0, index_col='ticker')
nosenti_models

# %%
best_real = real_models.max(axis=1)
best_nosenti = nosenti_models.max(axis=1)

# %%
delta = best_real - best_nosenti
delta = delta.sort_values(ascending=False)

# %%
beats_baseline = (best_real - baseline) > 0
beats_baseline

# %%
fig, ax = plt.subplots()
color_palette = beats_baseline.map({True: '#00305e', False: '0.8'})[delta.index].tolist()
ax.bar(delta.index, delta, zorder=5, color=color_palette, ec='k')

ax.set(
    xticks=[]
)

ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}%p")

ax.yaxis.grid('major', ls='--')

for idx, rec in enumerate(ax.patches):
    x = rec.get_xy()
    y = rec.get_height()
    y = y + 0.005 if y >= 0 else y - 0.01
    ax.text(x=idx, y=y, s=delta.index[idx], ha='center', weight='bold')


ax.spines['bottom'].set_position('zero')
sns.despine()


# %%
real_delta_bl = best_real - baseline
real_delta_bl = real_delta_bl.sort_values(ascending=False)  # Sort!
nosenti_delta_bl = best_nosenti - baseline
nosenti_delta_bl = nosenti_delta_bl[real_delta_bl.index]  # carry over sort

WIDTH = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

xs = np.arange(10)


ax.bar(x=xs - WIDTH/2, width=WIDTH, height=real_delta_bl, zorder=5, color='#3CBBB1', ec='k', label="With Sentiment")
ax.bar(x=xs + WIDTH/2, width=WIDTH, height=nosenti_delta_bl, zorder=5,  color='#005366', ec='k', label="Without Sentiment")
ax.legend(frameon=False, loc='lower left')

ax.set(
    xticks=np.arange(10),
    xticklabels=[],
    ylabel="Improvement over Baseline (%p)"
)

ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}")

ax.yaxis.grid('major', ls='--', color='0.8')

label_ys = np.hstack((real_delta_bl.values.reshape(-1, 1), nosenti_delta_bl.values.reshape(-1, 1))).max(axis=1)

for idx, rec in enumerate(ax.patches[::2]):
    x = rec.get_xy()
    y = label_ys[idx]
    y = y + 0.0025 if y > 0 else 0.0025
    ax.text(x=idx, y=y, s=real_delta_bl.index[idx], ha='center', weight='normal', size=14, zorder=10)

ax.text(x=9.9, y=0, s="Naive Baseline", va='center', weight='bold')
ax.spines['bottom'].set_position('zero')
sns.despine()
plt.tight_layout()
plt.show()

fig.savefig("../30_results/plots/performancePlot.svg", facecolor='white')
plt.close()


# %%
######################## VARX Models ################################
varx = pd.read_csv("../30_results/CorrectVARX.csv", sep=';', index_col="ticker")
varx

diff_df = (
    pd.DataFrame({'ml_sentiment': varx['ml_sentiment'] - baseline, 'vader': varx['vader'] - baseline})
    .sort_values('ml_sentiment', ascending=False)
    # .reset_index()
    # .melt(value_vars=('ml_sentiment', 'vader'), id_vars='ticker', var_name='sentiment', value_name='accuracy')
)

#%%

WIDTH = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

xs = np.arange(10)


ax.bar(x=xs - WIDTH/2, width=WIDTH, height=diff_df['ml_sentiment'], zorder=5, color='#56cfe1', ec='k', label="ML Sentiment")
ax.bar(x=xs + WIDTH/2, width=WIDTH, height=diff_df['vader'], zorder=5,  color='#5e60ce', ec='k', label="VADER")
ax.legend(frameon=False, loc='lower left')

ax.set(
    xticks=np.arange(10),
    xticklabels=[],
    ylabel="Improvement over Baseline (%p)"
)


ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}")

ax.yaxis.grid('major', ls='--', color='0.8')

label_ys = diff_df.max(axis=1)

for idx, rec in enumerate(ax.patches[::2]):
    x = rec.get_xy()
    y = label_ys[idx]
    y = y + 0.005 if y > 0 else 0.005
    ax.text(x=idx, y=y, s=diff_df.index[idx], ha='center', weight='normal', size=14, zorder=10)

ax.text(x=9.9, y=0, s="Naive Baseline", va='center', weight='bold')
ax.spines['bottom'].set_position('zero')
sns.despine()
plt.tight_layout()
plt.show()

fig.savefig("../30_results/plots/VARXperformancePlot.svg", facecolor='white')
plt.close()
