#%%
import pandas as pd
from rich.console import Console
c = Console(highlight=False)

#%%
root_path = "../"  # IPyhton
# root_path = "./"  # CLI

with open(root_path + "00_source_data/sp500_tickers.txt") as f:
    a = f.read()

tickers = a.split("\n")

#%%
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

#%%
print( pd.DataFrame(results).sort_values(by=1, ascending=False).head(15).to_markdown())

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