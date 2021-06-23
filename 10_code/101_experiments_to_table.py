# %%
from email.mime import base
import re
from itertools import product
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


senti = "ml_sentiment"
ticker = "TSLA"


baseline_acc = pd.read_csv(root_path + f"30_results/naive_baseline_benchmarks.csv", sep=";")['accuracy']

# %%
results = []

for ticker, senti in product(tickers, ['ml_sentiment', 'vader']):
    with open(root_path + f"20_outputs/benchmarks/{ticker}/{senti}/stats.log", 'r') as f:
        s = f.read()
    rfscore, lgbmscore = [float(x) for x in re.findall("(?<=TEST Accuracy = )(.*)\n", s)]
    results.append((ticker, senti, rfscore, lgbmscore))

#%%
import pandas as pd
df = pd.DataFrame(results, columns=['ticker', 'senti', 'rf', 'lgbm']).set_index('ticker')
df

#%%
final = df.pivot(columns='senti').loc[tickers, :].round(3)
# final['baseline'] = baseline_acc.reset_index(drop=True)
# final

#%%
final.to_csv(root_path + f"30_results/model_benchmarks.csv", sep=";")

#%%