#%%
import twint

root_path = "../"
root_path = "./"

with open(root_path + "00_source_data/sp500_tickers.txt") as f:
    a = f.read()

tickers = a.split("\n")
print(f"Found {len(tickers)} tickers!")

#%%
for ticker in tickers:
    print(f"### Scraping ticker {ticker} ###")
    c = twint.Config()
    c.Search = f"\${ticker}"

    c.Since = "2021-01-01"
    c.Until = "2021-01-31"

    c.Store_csv = True
    c.Output = root_path + f"20_outputs/pre_study/{ticker}_tweets.csv"

    twint.run.Search(c)
