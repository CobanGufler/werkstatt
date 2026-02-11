# scripts/plot_results_final.py
from __future__ import annotations

import argparse
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FREQ_ORDER = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
_FREQ_MAP = {f.lower(): f for f in FREQ_ORDER}


def _normalize_freq(label: str) -> str:
    s = str(label).strip()
    key = s.lower()
    return _FREQ_MAP.get(key, s)


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
    # First column is model id (often "Unnamed: 0" from your ALL csv)
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

    # handle chronos tiny/base; moirai small/base; timesfm (single)
    if "timesfm" in s:
        return "TimesFM"
    if "chronos" in s:
        # heuristics for tiny/base
        if "tiny" in s:
            return "Chronos Tiny"
        if "base" in s:
            return "Chronos Base"
        # fallback: if you only have base in results_final_base and tiny in results_final_small,
        # infer by folder usage in calling functions; keep generic here:
        return "Chronos"
    if "moirai" in s:
        if "small" in s:
            return "Moirai Small"
        if "base" in s:
            return "Moirai Base"
        return "Moirai"
    return str(model_id)


def _read_results_by_freq(results_root: str) -> Dict[str, pd.DataFrame]:
    """
    Expects:
      results_root/
        Daily/
          ALL_m4_*.csv
        Hourly/
          ALL_m4_*.csv
        ...
    Returns dict[freq] = standardized df with columns: model_id, MASE, sMAPE, time
    """
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
        norm_freq = _normalize_freq(freq)
        df["freq"] = norm_freq
        out[norm_freq] = df

    # keep only known order if possible
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


def _grouped_bar(
    data: pd.DataFrame,
    x_col: str,
    series_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    series_order: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
) -> None:
    plt.figure(figsize=(11, 5))

    if x_order is not None:
        present = set(data[x_col].tolist())
        x_vals = [x for x in x_order if x in present]
        x_vals += [x for x in list(dict.fromkeys(data[x_col].tolist())) if x not in set(x_vals)]
    else:
        x_vals = list(dict.fromkeys(data[x_col].tolist()))
    if series_order is None:
        series_vals = sorted(data[series_col].unique().tolist())
    else:
        series_vals = [s for s in series_order if s in set(data[series_col].unique().tolist())]

    x = np.arange(len(x_vals))
    width = 0.8 / max(len(series_vals), 1)

    for i, s in enumerate(series_vals):
        sub = data[data[series_col] == s].set_index(x_col)
        ys = [sub.loc[v, y_col] if v in sub.index else np.nan for v in x_vals]
        plt.bar(x + (i - (len(series_vals) - 1) / 2) * width, ys, width=width, label=s)

    plt.xticks(x, x_vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    if len(series_vals) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_1_base_models_mase(base_by_freq: Dict[str, pd.DataFrame], out_dir: str) -> None:
    rows = []
    for freq, df in base_by_freq.items():
        keep = df[df["model_name"].isin(["TimesFM", "Chronos Base", "Moirai Base", "Chronos", "Moirai"])]
        # If base files don't explicitly say "Base" in model_name, map Chronos->Chronos Base, Moirai->Moirai Base
        keep = keep.copy()
        keep.loc[keep["model_name"] == "Chronos", "model_name"] = "Chronos Base"
        keep.loc[keep["model_name"] == "Moirai", "model_name"] = "Moirai Base"
        for _, r in keep.iterrows():
            rows.append({"freq": freq, "model": r["model_name"], "MASE": float(r["MASE"])})

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("Plot1: No data found for base models (TimesFM/Chronos Base/Moirai Base).")

    _grouped_bar(
        data=data,
        x_col="freq",
        series_col="model",
        y_col="MASE",
        title="(1) Base Modelle – MASE Vergleich über alle Frequenzen",
        ylabel="MASE (lower = better)",
        out_path=os.path.join(out_dir, "01_base_models_MASE.png"),
        series_order=["TimesFM", "Chronos Base", "Moirai Base"],
        x_order=FREQ_ORDER,
    )


def plot_1_base_models_mase_with_naive2(
    base_by_freq: Dict[str, pd.DataFrame],
    naive2_csv: str,
    out_dir: str,
) -> None:
    if not naive2_csv or not os.path.exists(naive2_csv):
        return

    naive_df = pd.read_csv(naive2_csv)
    if "group" not in naive_df.columns or "MASE_Naive2" not in naive_df.columns:
        return

    # map group to lowercase for matching (results folders are lowercase)
    naive_map = {
        str(r["group"]).strip().lower(): float(r["MASE_Naive2"])
        for _, r in naive_df.iterrows()
        if pd.notna(r.get("MASE_Naive2", np.nan))
    }

    rows = []
    for freq, df in base_by_freq.items():
        keep = df[df["model_name"].isin(["TimesFM", "Chronos Base", "Moirai Base", "Chronos", "Moirai"])]
        keep = keep.copy()
        keep.loc[keep["model_name"] == "Chronos", "model_name"] = "Chronos Base"
        keep.loc[keep["model_name"] == "Moirai", "model_name"] = "Moirai Base"
        for _, r in keep.iterrows():
            rows.append({"freq": freq, "model": r["model_name"], "MASE": float(r["MASE"])})

    data = pd.DataFrame(rows)
    if data.empty:
        return

    plt.figure(figsize=(11, 5))

    present = set(data["freq"].tolist())
    x_vals = [x for x in FREQ_ORDER if x in present]
    x_vals += [x for x in list(dict.fromkeys(data["freq"].tolist())) if x not in set(x_vals)]
    series_vals = ["TimesFM", "Chronos Base", "Moirai Base"]
    x = np.arange(len(x_vals))
    width = 0.8 / max(len(series_vals), 1)

    for i, s in enumerate(series_vals):
        sub = data[data["model"] == s].set_index("freq")
        ys = [sub.loc[v, "MASE"] if v in sub.index else np.nan for v in x_vals]
        plt.bar(x + (i - (len(series_vals) - 1) / 2) * width, ys, width=width, label=s)

    # Naive2 reference lines per frequency
    for j, f in enumerate(x_vals):
        nv = naive_map.get(str(f).strip().lower(), np.nan)
        if np.isfinite(nv):
            x0 = x[j] - 0.45
            x1 = x[j] + 0.45
            plt.plot([x0, x1], [nv, nv], color="red", linewidth=2, label="_nolegend_")

    # legend entry for Naive2
    plt.plot([], [], color="red", linewidth=2, label="Naive2")

    plt.xticks(x, x_vals)
    plt.title("(1) Base Modelle – MASE Vergleich + Naive2 Referenz")
    plt.ylabel("MASE (lower = better)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_base_models_MASE_with_naive2.png"), dpi=200)
    plt.close()


def plot_2_chronos_tiny_vs_base(
    base_by_freq: Dict[str, pd.DataFrame],
    small_by_freq: Dict[str, pd.DataFrame],
    out_dir: str,
) -> None:
    rows_mase = []
    rows_time = []

    for freq in list(dict.fromkeys(list(base_by_freq.keys()) + list(small_by_freq.keys()))):
        df_b = base_by_freq.get(freq)
        df_s = small_by_freq.get(freq)

        # base chronos
        base_mase = base_time = None
        if df_b is not None:
            cb = df_b[df_b["model_name"].isin(["Chronos Base", "Chronos"])].copy()
            if not cb.empty:
                base_mase = float(cb.iloc[0]["MASE"])
                base_time = float(cb.iloc[0]["time"]) if not pd.isna(cb.iloc[0]["time"]) else np.nan

        # tiny chronos
        tiny_mase = tiny_time = None
        if df_s is not None:
            ct = df_s[df_s["model_name"].isin(["Chronos Tiny", "Chronos"])].copy()
            # in small folder it might just be "Chronos" -> interpret as Tiny
            if not ct.empty:
                # prefer explicit tiny if present
                if (df_s["model_name"] == "Chronos Tiny").any():
                    ct = df_s[df_s["model_name"] == "Chronos Tiny"]
                tiny_mase = float(ct.iloc[0]["MASE"])
                tiny_time = float(ct.iloc[0]["time"]) if not pd.isna(ct.iloc[0]["time"]) else np.nan

        if base_mase is not None:
            rows_mase.append({"freq": freq, "model": "Chronos Base", "MASE": base_mase})
            rows_time.append({"freq": freq, "model": "Chronos Base", "time": base_time})
        if tiny_mase is not None:
            rows_mase.append({"freq": freq, "model": "Chronos Tiny", "MASE": tiny_mase})
            rows_time.append({"freq": freq, "model": "Chronos Tiny", "time": tiny_time})

    dm = pd.DataFrame(rows_mase)
    dt = pd.DataFrame(rows_time)

    if dm.empty:
        raise RuntimeError("Plot2: No Chronos Tiny/Base data found.")

    _grouped_bar(
        data=dm,
        x_col="freq",
        series_col="model",
        y_col="MASE",
        title="(2) Chronos Tiny vs Base – MASE Vergleich über alle Frequenzen",
        ylabel="MASE (lower = better)",
        out_path=os.path.join(out_dir, "02_chronos_tiny_vs_base_MASE.png"),
        series_order=["Chronos Tiny", "Chronos Base"],
        x_order=FREQ_ORDER,
    )

    # runtime plot (only if there is any non-nan)
    if dt["time"].notna().any():
        _grouped_bar(
            data=dt,
            x_col="freq",
            series_col="model",
            y_col="time",
            title="(2) Chronos Tiny vs Base – Laufzeit (total_time_eval) über alle Frequenzen",
            ylabel="Runtime (seconds)",
            out_path=os.path.join(out_dir, "02_chronos_tiny_vs_base_runtime.png"),
            series_order=["Chronos Tiny", "Chronos Base"],
            x_order=FREQ_ORDER,
        )


def plot_3_moirai_small_vs_base(
    base_by_freq: Dict[str, pd.DataFrame],
    small_by_freq: Dict[str, pd.DataFrame],
    out_dir: str,
) -> None:
    rows_mase = []
    rows_time = []

    for freq in list(dict.fromkeys(list(base_by_freq.keys()) + list(small_by_freq.keys()))):
        df_b = base_by_freq.get(freq)
        df_s = small_by_freq.get(freq)

        # base moirai
        base_mase = base_time = None
        if df_b is not None:
            mb = df_b[df_b["model_name"].isin(["Moirai Base", "Moirai"])].copy()
            if not mb.empty:
                base_mase = float(mb.iloc[0]["MASE"])
                base_time = float(mb.iloc[0]["time"]) if not pd.isna(mb.iloc[0]["time"]) else np.nan

        # small moirai
        small_mase = small_time = None
        if df_s is not None:
            ms = df_s[df_s["model_name"].isin(["Moirai Small", "Moirai"])].copy()
            # in small folder it might just be "Moirai" -> interpret as Small
            if not ms.empty:
                if (df_s["model_name"] == "Moirai Small").any():
                    ms = df_s[df_s["model_name"] == "Moirai Small"]
                small_mase = float(ms.iloc[0]["MASE"])
                small_time = float(ms.iloc[0]["time"]) if not pd.isna(ms.iloc[0]["time"]) else np.nan

        if base_mase is not None:
            rows_mase.append({"freq": freq, "model": "Moirai Base", "MASE": base_mase})
            rows_time.append({"freq": freq, "model": "Moirai Base", "time": base_time})
        if small_mase is not None:
            rows_mase.append({"freq": freq, "model": "Moirai Small", "MASE": small_mase})
            rows_time.append({"freq": freq, "model": "Moirai Small", "time": small_time})

    dm = pd.DataFrame(rows_mase)
    dt = pd.DataFrame(rows_time)

    if dm.empty:
        raise RuntimeError("Plot3: No Moirai Small/Base data found.")

    _grouped_bar(
        data=dm,
        x_col="freq",
        series_col="model",
        y_col="MASE",
        title="(3) Moirai Small vs Base – MASE Vergleich über alle Frequenzen",
        ylabel="MASE (lower = better)",
        out_path=os.path.join(out_dir, "03_moirai_small_vs_base_MASE.png"),
        series_order=["Moirai Small", "Moirai Base"],
        x_order=FREQ_ORDER,
    )

    if dt["time"].notna().any():
        _grouped_bar(
            data=dt,
            x_col="freq",
            series_col="model",
            y_col="time",
            title="(3) Moirai Small vs Base – Laufzeit (total_time_eval) über alle Frequenzen",
            ylabel="Runtime (seconds)",
            out_path=os.path.join(out_dir, "03_moirai_small_vs_base_runtime.png"),
            series_order=["Moirai Small", "Moirai Base"],
            x_order=FREQ_ORDER,
        )


def plot_4_base_score_mase_smape(base_by_freq: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """
    Composite score (lower=better):
      For each frequency:
        - min-max scale MASE across base models
        - min-max scale sMAPE across base models
        - score_freq = 0.5*scaled_MASE + 0.5*scaled_sMAPE
      Overall score per model = mean(score_freq over freqs where model exists)
    """
    # build panel: rows = (freq, model) with MASE, sMAPE
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

    # save table too
    overall.to_csv(os.path.join(out_dir, "04_base_models_composite_score_table.csv"), index=False)

    # bar plot
    plt.figure(figsize=(9, 4.8))
    plt.bar(overall["model"], overall["score"])
    plt.title("(4) Base Modelle – Composite Score (MASE & sMAPE, min-max je Frequenz)")
    plt.ylabel("Score (lower = better)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_base_models_composite_score.png"), dpi=200)
    plt.close()

    # optional: per-freq heatmap-like table plot as image (simple)
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
    ap.add_argument(
        "--naive2_csv",
        type=str,
        default=os.path.join("owa_naive2_uni2ts", "naive2_metrics_per_group_uni2ts.csv"),
        help="Naive2 metrics CSV for reference lines",
    )
    args = ap.parse_args()

    _ensure_outdir(args.out_dir)

    base_by_freq = _read_results_by_freq(args.base_dir)
    small_by_freq = _read_results_by_freq(args.small_dir)

    # 1
    plot_1_base_models_mase(base_by_freq, args.out_dir)
    plot_1_base_models_mase_with_naive2(base_by_freq, args.naive2_csv, args.out_dir)

    # 2
    plot_2_chronos_tiny_vs_base(base_by_freq, small_by_freq, args.out_dir)

    # 3
    plot_3_moirai_small_vs_base(base_by_freq, small_by_freq, args.out_dir)

    # 4
    plot_4_base_score_mase_smape(base_by_freq, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")
    print("Files created:")
    for p in sorted(glob.glob(os.path.join(args.out_dir, "*.png")) + glob.glob(os.path.join(args.out_dir, "*.csv"))):
        print(" -", p)


if __name__ == "__main__":
    main()



