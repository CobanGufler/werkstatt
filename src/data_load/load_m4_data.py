from datasetsforecast.m4 import M4
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_DIR = "../../data/m4/datasets"

GROUPS = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]

def load_m4(group: str, data_dir: str = DATA_DIR):
    train_path = os.path.join(data_dir, f"{group}-train.csv")
    test_path  = os.path.join(data_dir, f"{group}-test.csv")

    # 1) bevorzugt: lokale CSVs (passt zu deinem Ordnerlayout)
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    else:
        # 2) fallback: datasetsforecast Loader
        train_df, test_df, _ = M4.load(directory=data_dir, group=group)

        # falls test_df bei dir None ist: splitte aus train_df
        if test_df is None:
            H = {"Hourly":48, "Daily":14, "Weekly":13, "Monthly":18, "Quarterly":8, "Yearly":6}[group]
            train_df = train_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            test_df = train_df.groupby("unique_id", sort=False).tail(H).copy()
            train_df = train_df.drop(test_df.index).copy()

    # Schema + sort
    train_df = train_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    train_df["y"] = pd.to_numeric(train_df["y"], errors="coerce")
    test_df["y"]  = pd.to_numeric(test_df["y"], errors="coerce")

    return train_df, test_df


def plot_example_series(train_df, n=5):
    # robuste uid-sortierung: nimmt den numerischen Teil, falls vorhanden
    def uid_key(s):
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        return int(digits) if digits else 0

    uids = sorted(train_df["unique_id"].unique(), key=uid_key)[:n]

    fig, axes = plt.subplots(len(uids), 1, figsize=(10, 8), sharex=True)
    if len(uids) == 1:
        axes = [axes]

    for ax, uid in zip(axes, uids):
        ts = train_df[train_df["unique_id"] == uid].sort_values("ds")
        ax.plot(ts["ds"], ts["y"])
        ax.set_title(f"Serie {uid}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for g in GROUPS:
        tr, te = load_m4(g)
        print(f"{g}: train={len(tr):,} test={len(te):,} series={tr['unique_id'].nunique():,}")
        print("  files:",
              os.path.exists(os.path.join(DATA_DIR, f"{g}-train.csv")),
              os.path.exists(os.path.join(DATA_DIR, f"{g}-test.csv")))

    # Beispiel plot für Daily
    df_train, df_test = load_m4("Daily")
    plot_example_series(df_train, n=7)


