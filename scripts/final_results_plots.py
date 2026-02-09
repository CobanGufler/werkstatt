from __future__ import annotations

import argparse
import os
import glob
from typing import Dict, List, Optional

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

    mase_col = _find_metric_col(df, ["MASE[0.5]", "MASE", "mase"])
    smape_col = _find_metric_col(df, ["sMAPE[0.5]", "SMAPE", "smape", "smapE", "sMape"])
    time_col = _find_metric_col(df, ["total_time_eval", "runtime", "time", "eval_time"])

    if mase_col is None:
        raise ValueError(f"Could not find MASE column. Columns={list(df.columns)}")
    if smape_col is None:
        raise ValueError(f"Could not find sMAPE column. Columns={list(df.columns)}")

    out = df[["model_id", mase_col, smape_col] + ([time_col] if time_col else [])].copy()
    out = out.rename(columns={mase_col: "MASE", smape_col: "sMAPE"})
    if time_col:
        out = out.rename(columns={time_col: "time"})
    else:
        out["time"] = np.nan

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


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bar_with_runtime_delta_labels(
    freqs: List[str],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_path: str,
    delta_runtime_seconds: Dict[str, float],
    delta_label_prefix: str = "Δt",
) -> None:
    """
    Draw grouped bars for MASE and annotate each frequency with a single delta-runtime label (e.g., +100s).
    delta_runtime_seconds is per frequency: base_time - other_time
    """
    plt.figure(figsize=(11, 5))

    x = np.arange(len(freqs))
    names = list(series.keys())
    width = 0.8 / max(len(names), 1)

    # bars
    for i, name in enumerate(names):
        vals = series[name]
        plt.bar(x + (i - (len(names) - 1) / 2) * width, vals, width=width, label=name)

    # annotate delta runtime per frequency above the taller bar
    for j, freq in enumerate(freqs):
        vals_here = [series[name][j] for name in names]
        y_top = np.nanmax(vals_here) if np.isfinite(np.nanmax(vals_here)) else 0.0

        dt = delta_runtime_seconds.get(freq, np.nan)
        if np.isfinite(dt):
            sign = "+" if dt > 0 else ""
            label = f"{delta_label_prefix} {sign}{int(round(dt))}s"
        else:
            label = f"{delta_label_prefix} n/a"

        plt.text(
            x[j],
            y_top + (0.02 * max(1e-9, np.nanmax(vals_here))),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(x, freqs)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    if len(names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_1_base_models_mase(base_by_freq: Dict[str, pd.DataFrame], out_dir: str) -> None:
    rows = []
    for freq, df in base_by_freq.items():
        keep = df[df["model_name"].isin(["TimesFM", "Chronos Base", "Moirai Base", "Chronos", "Moirai"])].copy()
        keep.loc[keep["model_name"] == "Chronos", "model_name"] = "Chronos Base"
        keep.loc[keep["model_name"] == "Moirai", "model_name"] = "Moirai Base"
        for _, r in keep.iterrows():
            rows.append({"freq": freq, "model": r["model_name"], "MASE": float(r["MASE"])})

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("Plot1: No data found for base models (TimesFM/Chronos Base/Moirai Base).")

    # simple grouped bar
    freqs = [f for f in FREQ_ORDER if f in data["freq"].unique().tolist()] + [
        f for f in data["freq"].unique().tolist() if f not in FREQ_ORDER
    ]
    models = ["TimesFM", "Chronos Base", "Moirai Base"]

    series = {}
    for m in models:
        sub = data[data["model"] == m].set_index("freq")
        series[m] = [float(sub.loc[f, "MASE"]) if f in sub.index else np.nan for f in freqs]

    plt.figure(figsize=(11, 5))
    x = np.arange(len(freqs))
    width = 0.8 / len(models)

    for i, m in enumerate(models):
        plt.bar(x + (i - (len(models) - 1) / 2) * width, series[m], width=width, label=m)

    plt.xticks(x, freqs)
    plt.title("(1) Base Modelle – MASE Vergleich über alle Frequenzen")
    plt.ylabel("MASE (lower = better)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_base_models_MASE.png"), dpi=200)
    plt.close()


def plot_2_chronos_mase_with_runtime_delta(
    base_by_freq: Dict[str, pd.DataFrame],
    small_by_freq: Dict[str, pd.DataFrame],
    out_dir: str,
) -> None:
    freqs = list(dict.fromkeys([*base_by_freq.keys(), *small_by_freq.keys()]))
    # keep nice ordering
    freqs = [f for f in FREQ_ORDER if f in freqs] + [f for f in freqs if f not in FREQ_ORDER]

    chronos_base_mase, chronos_tiny_mase = [], []
    delta_t = {}

    for freq in freqs:
        df_b = base_by_freq.get(freq)
        df_s = small_by_freq.get(freq)

        base_mase = base_time = np.nan
        tiny_mase = tiny_time = np.nan

        if df_b is not None:
            cb = df_b[df_b["model_name"].isin(["Chronos Base", "Chronos"])].copy()
            if not cb.empty:
                base_mase = float(cb.iloc[0]["MASE"])
                base_time = float(cb.iloc[0]["time"]) if not pd.isna(cb.iloc[0]["time"]) else np.nan

        if df_s is not None:
            ct = df_s[df_s["model_name"].isin(["Chronos Tiny", "Chronos"])].copy()
            if not ct.empty:
                if (df_s["model_name"] == "Chronos Tiny").any():
                    ct = df_s[df_s["model_name"] == "Chronos Tiny"]
                tiny_mase = float(ct.iloc[0]["MASE"])
                tiny_time = float(ct.iloc[0]["time"]) if not pd.isna(ct.iloc[0]["time"]) else np.nan

        chronos_base_mase.append(base_mase)
        chronos_tiny_mase.append(tiny_mase)

        # delta runtime: Base - Tiny
        if np.isfinite(base_time) and np.isfinite(tiny_time):
            delta_t[freq] = base_time - tiny_time
        else:
            delta_t[freq] = np.nan

    series = {"Chronos Tiny": chronos_tiny_mase, "Chronos Base": chronos_base_mase}

    _bar_with_runtime_delta_labels(
        freqs=freqs,
        series=series,
        title="(2) Chronos Tiny vs Base – MASE + ΔRuntime (Base − Tiny) pro Frequenz",
        ylabel="MASE (lower = better)",
        out_path=os.path.join(out_dir, "02_chronos_tiny_vs_base_MASE_with_delta_runtime.png"),
        delta_runtime_seconds=delta_t,
        delta_label_prefix="Δt",
    )


def plot_3_moirai_mase_with_runtime_delta(
    base_by_freq: Dict[str, pd.DataFrame],
    small_by_freq: Dict[str, pd.DataFrame],
    out_dir: str,
) -> None:
    freqs = list(dict.fromkeys([*base_by_freq.keys(), *small_by_freq.keys()]))
    freqs = [f for f in FREQ_ORDER if f in freqs] + [f for f in freqs if f not in FREQ_ORDER]

    moirai_base_mase, moirai_small_mase = [], []
    delta_t = {}

    for freq in freqs:
        df_b = base_by_freq.get(freq)
        df_s = small_by_freq.get(freq)

        base_mase = base_time = np.nan
        small_mase = small_time = np.nan

        if df_b is not None:
            mb = df_b[df_b["model_name"].isin(["Moirai Base", "Moirai"])].copy()
            if not mb.empty:
                base_mase = float(mb.iloc[0]["MASE"])
                base_time = float(mb.iloc[0]["time"]) if not pd.isna(mb.iloc[0]["time"]) else np.nan

        if df_s is not None:
            ms = df_s[df_s["model_name"].isin(["Moirai Small", "Moirai"])].copy()
            if not ms.empty:
                if (df_s["model_name"] == "Moirai Small").any():
                    ms = df_s[df_s["model_name"] == "Moirai Small"]
                small_mase = float(ms.iloc[0]["MASE"])
                small_time = float(ms.iloc[0]["time"]) if not pd.isna(ms.iloc[0]["time"]) else np.nan

        moirai_base_mase.append(base_mase)
        moirai_small_mase.append(small_mase)

        # delta runtime: Base - Small
        if np.isfinite(base_time) and np.isfinite(small_time):
            delta_t[freq] = base_time - small_time
        else:
            delta_t[freq] = np.nan

    series = {"Moirai Small": moirai_small_mase, "Moirai Base": moirai_base_mase}

    _bar_with_runtime_delta_labels(
        freqs=freqs,
        series=series,
        title="(3) Moirai Small vs Base – MASE + ΔRuntime (Base − Small) pro Frequenz",
        ylabel="MASE (lower = better)",
        out_path=os.path.join(out_dir, "03_moirai_small_vs_base_MASE_with_delta_runtime.png"),
        delta_runtime_seconds=delta_t,
        delta_label_prefix="Δt",
    )


def plot_4_base_score_mase_smape(base_by_freq: Dict[str, pd.DataFrame], out_dir: str) -> None:
    panel = []
    for freq, df in base_by_freq.items():
        keep = df[df["model_name"].isin(["TimesFM", "Chronos Base", "Moirai Base", "Chronos", "Moirai"])].copy()
        keep.loc[keep["model_name"] == "Chronos", "model_name"] = "Chronos Base"
        keep.loc[keep["model_name"] == "Moirai", "model_name"] = "Moirai Base"

        for _, r in keep.iterrows():
            panel.append(
                {
                    "freq": freq,
                    "model": r["model_name"],
                    "MASE": float(r["MASE"]),
                    "sMAPE": float(r["sMAPE"]),
                }
            )

    panel = pd.DataFrame(panel)
    if panel.empty:
        raise RuntimeError("Plot4: No base models found for composite score.")

    scores = []
    for freq, g in panel.groupby("freq", sort=False):
        mase = g["MASE"].to_numpy(dtype=float)
        smape = g["sMAPE"].to_numpy(dtype=float)

        def minmax(x: np.ndarray) -> np.ndarray:
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        mase_s = minmax(mase)
        smape_s = minmax(smape)

        sf = 0.5 * mase_s + 0.5 * smape_s
        for (model, val) in zip(g["model"].tolist(), sf.tolist()):
            scores.append({"freq": freq, "model": model, "score_freq": float(val)})

    scores = pd.DataFrame(scores)
    overall = scores.groupby("model", as_index=False)["score_freq"].mean().rename(columns={"score_freq": "score"})
    overall = overall.sort_values("score", ascending=True).reset_index(drop=True)

    overall.to_csv(os.path.join(out_dir, "04_base_models_composite_score_table.csv"), index=False)

    plt.figure(figsize=(9, 4.8))
    plt.bar(overall["model"], overall["score"])
    plt.title("(4) Base Modelle – Composite Score (MASE & sMAPE, min-max je Frequenz)")
    plt.ylabel("Score (lower = better)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_base_models_composite_score.png"), dpi=200)
    plt.close()

    pivot = scores.pivot_table(index="freq", columns="model", values="score_freq", aggfunc="mean")
    pivot = pivot.reindex([f for f in FREQ_ORDER if f in pivot.index] + [f for f in pivot.index if f not in FREQ_ORDER])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(pivot.fillna(np.nan).to_numpy(), 3),
        rowLabels=pivot.index.tolist(),
        colLabels=pivot.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.scale(1, 1.4)
    ax.set_title("Base Modelle – Score pro Frequenz (0=best innerhalb Frequenz)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_base_models_score_per_freq_table.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Path to results_final_base")
    ap.add_argument("--small_dir", type=str, required=True, help="Path to results_final_small")
    ap.add_argument("--out_dir", type=str, default="plots_final", help="Output folder for PNGs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_by_freq = _read_results_by_freq(args.base_dir)
    small_by_freq = _read_results_by_freq(args.small_dir)

    plot_1_base_models_mase(base_by_freq, args.out_dir)
    plot_2_chronos_mase_with_runtime_delta(base_by_freq, small_by_freq, args.out_dir)
    plot_3_moirai_mase_with_runtime_delta(base_by_freq, small_by_freq, args.out_dir)
    plot_4_base_score_mase_smape(base_by_freq, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")
    for p in sorted(glob.glob(os.path.join(args.out_dir, "*.png")) + glob.glob(os.path.join(args.out_dir, "*.csv"))):
        print(" -", p)


if __name__ == "__main__":
    main()


