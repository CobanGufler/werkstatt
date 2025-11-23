from datasetsforecast.m4 import M4

data_dir = "./data"

# load DAILY data
daily_df, _, _ = M4.load(directory=data_dir, group="Daily")

print(daily_df.shape)
print(daily_df.head())