from __future__ import annotations

import argparse
import os
import glob
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FREQ_ORDER = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]


def _latest_all_csv(freq_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(freq_dir, "ALL_m4_*.csv"))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _find_metric_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "model_id"})
    elif df.columns[0] != "model_id":
        df = df.rename(columns={df.columns[0]: "model_id"})

    time_col = _find_metric_col(df, ["total_time_eval", "runtime", "time", "eval_time"])
    if time_col is None:
        raise ValueError(f"Could not find runtime column. Columns={list(df.columns)}")

    out = df[["model_id", time_col]].copy()
    out = out.rename(columns={time_col: "time"})
    return out


def _pretty_model_name(model_id: str) -> str:
    s = str(model_id).lower()
    if "timesfm" in s:
        return "TimesFM"
    if "chronos" in s:
        if "tiny" in s:
            return "Chronos Tiny"
        if "base" in s:
            return "Chronos Base"
        return "Chronos"
    if "moirai" in s:
        if "small" in s:
            return "Moirai Small"
        if "base" in s:
            return "Moirai Base"
        return "Moirai"
    return str(model_id)


def _read_results_by_freq(results_root: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Not a directory: {results_root}")

    for freq in sorted(os.listdir(results_root)):
        freq_dir = os.path.join(results_root, freq)
        if not os.path.isdir(freq_dir):
            continue
        csv_path = _latest_all_csv(freq_dir)
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
        df = _standardize_df(df)
        df["model_name"] = df["model_id"].apply(_pretty_model_name)
        df["freq"] = freq
        out[freq] = df

    ordered = {}
    for f in FREQ_ORDER:
        if f in out:
            ordered[f] = out[f]
    for f in out:
        if f not in ordered:
            ordered[f] = out[f]
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--small_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="plots/final")
    ap.add_argument("--out_csv", type=str, default="runtime_all_models_by_freq.csv")
    ap.add_argument("--out_png", type=str, default="runtime_all_models_by_freq.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_by_freq = _read_results_by_freq(args.base_dir)
    small_by_freq = _read_results_by_freq(args.small_dir)

    rows = []
    for freq, df in base_by_freq.items():
        for _, r in df.iterrows():
            rows.append(
                {"freq": freq, "model": r["model_name"], "time": float(r["time"]), "source": "base"}
            )
    for freq, df in small_by_freq.items():
        for _, r in df.iterrows():
            rows.append(
                {"freq": freq, "model": r["model_name"], "time": float(r["time"]), "source": "small"}
            )

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No runtime data found.")


    data.loc[(data["model"] == "Chronos") & (data["source"] == "base"), "model"] = "Chronos Base"
    data.loc[(data["model"] == "Chronos") & (data["source"] == "small"), "model"] = "Chronos Tiny"
    data.loc[(data["model"] == "Moirai") & (data["source"] == "base"), "model"] = "Moirai Base"
    data.loc[(data["model"] == "Moirai") & (data["source"] == "small"), "model"] = "Moirai Small"

    model_order = ["TimesFM", "Chronos Base", "Moirai Base", "Chronos Tiny", "Moirai Small"]

    data["freq"] = data["freq"].astype(str).str.strip().str.lower().str.title()
    data["freq"] = pd.Categorical(data["freq"], categories=FREQ_ORDER, ordered=True)
    data["model"] = pd.Categorical(data["model"], categories=model_order, ordered=True)
    data = data.sort_values(["freq", "model"])

    data.to_csv(os.path.join(args.out_dir, args.out_csv), index=False)

    x = np.arange(len(FREQ_ORDER))
    width = 0.8 / len(model_order)

    plt.figure(figsize=(11, 4.5))
    for i, m in enumerate(model_order):
        sub = data[data["model"] == m].set_index("freq")
        vals = [float(sub.loc[f, "time"]) if f in sub.index else np.nan for f in FREQ_ORDER]
        plt.bar(x + (i - (len(model_order) - 1) / 2) * width, vals, width=width, label=m)

    plt.xticks(x, FREQ_ORDER)
    plt.title("Laufzeit pro Frequenz und Modell")
    plt.ylabel("Runtime (Sekunden)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, args.out_png), dpi=200)
    plt.close()

    print(f"Saved: {os.path.join(args.out_dir, args.out_png)}")
    print(f"Saved: {os.path.join(args.out_dir, args.out_csv)}")


if __name__ == "__main__":
    main()
