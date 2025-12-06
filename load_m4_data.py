from datasetsforecast.m4 import M4
import matplotlib.pyplot as plt

DATA_DIR = "./data"

def load_m4_hourly():
    hourly_df, _, _ = M4.load(directory=DATA_DIR, group="Hourly")
    return hourly_df

def load_m4_daily():
    daily_df, _, _ = M4.load(directory=DATA_DIR, group="Daily")
    return daily_df

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
    df = load_m4_daily()
    plot_example_series(df, n=7)