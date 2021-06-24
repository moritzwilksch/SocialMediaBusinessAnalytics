# %%
import pandas as pd
import re
from itertools import product
root_path = "../"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]


senti = "ml_sentiment"
ticker = "TSLA"


baseline_acc = pd.read_csv(root_path + f"30_results/naive_baseline_benchmarks.csv", sep=";")['accuracy']

# %%
results = []
no_senti_results = []

for ticker, senti in product(tickers, ['ml_sentiment', 'vader']):
    with open(root_path + f"20_outputs/benchmarks/{ticker}/{senti}/stats.log", 'r') as f:
        s = f.read()
    rfscore, lgbmscore = [float(x) for x in re.findall("(?<=TEST Accuracy = )(.*)\n", s)]
    results.append((ticker, senti, rfscore, lgbmscore))

for ticker in tickers:
    with open(root_path + f"20_outputs/benchmarks/{ticker}/stats_noSenti.log", 'r') as f:
        s = f.read()
    rfscore, lgbmscore = [float(x) for x in re.findall("(?<=TEST Accuracy = )(.*)\n", s)]
    no_senti_results.append((ticker, rfscore, lgbmscore))

# %%
df = pd.DataFrame(results, columns=['ticker', 'senti', 'rf', 'lgbm']).set_index('ticker')
df

# %%
final = df.pivot(columns='senti').loc[tickers, :].round(3)
# final['baseline'] = baseline_acc.reset_index(drop=True)
final

# %%
final.to_csv(root_path + f"30_results/model_benchmarks.csv", sep=";")

# %%
df_nosenti = pd.DataFrame(no_senti_results, columns=['ticker', 'rf', 'lgbm']).set_index('ticker')
df_nosenti.round(3).to_csv(root_path + "30_results/model_benchmarks_NOSENTI.csv", sep=';')
