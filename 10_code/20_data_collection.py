#%%
import twint
c = twint.Config()

# root_path = "../"  # IPyhton
root_path = "./"  # CLI

# TODO: 
# [X] TSLA
# [X] AAPL
# [X] AMZN
# [X] FB
# [X] MSFT
# [X] TWTR
# [X] AMD
# [X] NFLX
# [X] NVDA
# [X] INTC

TICKER = "INTC" 

c.Search = f"\${TICKER}"

c.Since = "2019-01-01 00:00:00"
c.Until = "2021-04-30 23:59:59"

c.Store_csv = True
c.Output = root_path + f"00_source_data/{TICKER}_tweets.csv"

twint.run.Search(c)
