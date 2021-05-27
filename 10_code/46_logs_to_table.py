# %%
import re
import pandas as pd
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]
d = {}
# %%

for ticker in tickers:
    with open(root_path + f"10_code/logs/{ticker}_filtering.log") as f:
        s = f.read()

    regex = re.compile(r"(?<=shape: \()\d+(?=, \d+\))")
    results = re.findall(regex, s)

    d.update({ticker: pd.Series(results).drop_duplicates().astype('int32')})

# %%
df = (
    pd.DataFrame
    .from_dict(d)
    .set_axis(['initial', 'language filter', '#cashtags filter'], axis=0)
    .T
    )

#%%
df.applymap(lambda x: f"{x:,}").to_csv(root_path + "30_results/filtering_stats.csv", sep=';')

#%%
rel_reduction = df['#cashtags filter'].astype('int32') / df['initial'].astype('int32')

print(f"relative reduction: {rel_reduction.mean()*100:.1f}% (SD={rel_reduction.std()*100:.2f}%p)")