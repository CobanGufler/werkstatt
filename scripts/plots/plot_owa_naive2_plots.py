import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FREQ_ORDER = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
MODEL_ORDER = ["TimesFM", "Chronos Base", "Moirai Base"]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })


def plot_owa_overall(df_overall: pd.DataFrame, out_dir: str) -> None:
    df = df_overall.copy()
    df = df.set_index("model").reindex(MODEL_ORDER).reset_index()

    plt.figure(figsize=(6, 3.5))
    plt.bar(df["model"], df["OWA_overall_weighted"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("OWA Overall (gewichteter Mittelwert)")
    plt.ylabel("OWA (niedriger = besser)")
    plt.ylim(0, max(1.05, df["OWA_overall_weighted"].max() * 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "owa_overall_weighted.png"))
    plt.close()


def plot_owa_per_freq(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df = df[df["model"].isin(MODEL_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=FREQ_ORDER, ordered=True)
    df = df.sort_values(["group", "model"])

    x = np.arange(len(FREQ_ORDER))
    width = 0.25

    plt.figure(figsize=(10, 4))
    for i, m in enumerate(MODEL_ORDER):
        sub = df[df["model"] == m].set_index("group")
        vals = [float(sub.loc[g, "OWA"]) if g in sub.index else np.nan for g in FREQ_ORDER]
        plt.bar(x + (i - 1) * width, vals, width=width, label=m)

    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(x, FREQ_ORDER)
    plt.ylabel("OWA (niedriger = besser)")
    plt.title("OWA pro Frequenz (vs. Naive2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "owa_per_frequency.png"))
    plt.close()


def plot_owa_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df = df[df["model"].isin(MODEL_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=FREQ_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    pivot = df.pivot_table(index="group", columns="model", values="OWA", aggfunc="mean")
    pivot = pivot.reindex(index=FREQ_ORDER, columns=MODEL_ORDER)

    plt.figure(figsize=(6, 4))
    data = pivot.to_numpy()
    im = plt.imshow(data, cmap="RdYlGn_r", vmin=0.5, vmax=1.2)
    plt.colorbar(im, label="OWA (niedriger = besser)")
    plt.xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    plt.yticks(range(len(FREQ_ORDER)), FREQ_ORDER)
    plt.title("OWA Heatmap pro Frequenz")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "owa_heatmap.png"))
    plt.close()


def plot_owa_improvement(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df = df[df["model"].isin(MODEL_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=FREQ_ORDER, ordered=True)

    x = np.arange(len(FREQ_ORDER))
    width = 0.25

    plt.figure(figsize=(10, 4))
    for i, m in enumerate(MODEL_ORDER):
        sub = df[df["model"] == m].set_index("group")
        vals = [float(1.0 - sub.loc[g, "OWA"]) * 100 if g in sub.index else np.nan for g in FREQ_ORDER]
        plt.bar(x + (i - 1) * width, vals, width=width, label=m)

    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, FREQ_ORDER)
    plt.ylabel("Verbesserung ggü. Naive2 (%, höher = besser)")
    plt.title("OWA-Verbesserung pro Frequenz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "owa_improvement_percent.png"))
    plt.close()


def plot_components_vs_naive2(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df = df[df["model"].isin(MODEL_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=FREQ_ORDER, ordered=True)

    # MASE ratio and sMAPE ratio (model / naive2)
    df["MASE_ratio"] = df["MASE"] / df["MASE_Naive2"]
    df["sMAPE_ratio"] = df["sMAPE"] / df["sMAPE_Naive2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title in [
        (axes[0], "MASE_ratio", "MASE / Naive2 (niedriger = besser)"),
        (axes[1], "sMAPE_ratio", "sMAPE / Naive2 (niedriger = besser)"),
    ]:
        x = np.arange(len(FREQ_ORDER))
        width = 0.25
        for i, m in enumerate(MODEL_ORDER):
            sub = df[df["model"] == m].set_index("group")
            vals = [float(sub.loc[g, col]) if g in sub.index else np.nan for g in FREQ_ORDER]
            ax.bar(x + (i - 1) * width, vals, width=width, label=m)
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(FREQ_ORDER)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Ratio")
    axes[0].legend()
    fig.suptitle("MASE und sMAPE relativ zu Naive2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "components_vs_naive2.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="owa_naive2_uni2ts", help="Input folder")
    ap.add_argument("--out_dir", type=str, default="plots/owa", help="Output folder")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = args.out_dir
    _ensure_dir(out_dir)
    _style()

    df_overall = pd.read_csv(root / "owa_overall_weighted.csv")
    df_group = pd.read_csv(root / "owa_per_group_and_model.csv")

    plot_owa_overall(df_overall, out_dir)
    plot_owa_per_freq(df_group, out_dir)
    plot_owa_heatmap(df_group, out_dir)
    plot_owa_improvement(df_group, out_dir)
    plot_components_vs_naive2(df_group, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
