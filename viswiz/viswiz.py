import polars as pl
df = pl.read_parquet('val-0000*of-00005-*.parquet')
print(df.tail())