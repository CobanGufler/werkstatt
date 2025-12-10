from datasetsforecast.m4 import M4
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_DIR = "./data/m4/datasets"

def load_m4_hourly():
    hourly_df, _, _ = M4.load(directory=DATA_DIR, group="Hourly")
    return hourly_df

def load_m4_daily():

    daily_full, _, _ = M4.load(directory=DATA_DIR, group="Daily")
    daily_full = daily_full.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    test_path = os.path.join(DATA_DIR, "Daily-test.csv")
    if os.path.exists(test_path):
        daily_test = pd.read_csv(test_path)
    else:
        daily_test = None

    return daily_full, daily_test

def load_m4_weekly():
    weekly_df, _, _ = M4.load(directory=DATA_DIR, group="Weekly")
    return weekly_df

def load_m4_monthly():
    monthly_df, _, _ = M4.load(directory=DATA_DIR, group="Monthly")
    return monthly_df

def load_m4_quarterly():
    quarterly_df, _, _ = M4.load(directory=DATA_DIR, group="Quarterly")
    return quarterly_df

def load_m4_yearly():
    yearly_df, _, _ = M4.load(directory=DATA_DIR, group="Yearly")
    return yearly_df


def plot_example_series(data, n=5):
    uids = sorted(data["unique_id"].unique(), key=lambda s: int(s[1:]))[:n]

    fig, axes = plt.subplots(len(uids), 1, figsize=(10, 8), sharex=True)

    for ax, uid in zip(axes, uids):
        ts = data[data["unique_id"] == uid].sort_values("ds")
        ax.plot(ts["ds"], ts["y"])
        ax.set_title(f"Serie {uid}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # daily data
    df_train, df_test = load_m4_daily()
    print(df_train.dtypes)
    plot_example_series(df_train, n=7)

