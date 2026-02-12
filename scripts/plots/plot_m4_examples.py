import argparse
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from scripts.data_load.m4_gluonts_loader import get_m4_test_dataset


CANONICAL_GROUPS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
}

DISPLAY_NAMES = {
    "Hourly": "Hourly",
    "Daily": "Daily",
    "Weekly": "Weekly",
    "Monthly": "Monthly",
    "Quarterly": "Quarterly",
    "Yearly": "Yearly",
}


def parse_groups(groups_arg: str) -> List[str]:
    parts = [p.strip() for p in groups_arg.split(",") if p.strip()]
    out = []
    for p in parts:
        key = p.lower()
        out.append(CANONICAL_GROUPS.get(key, p))
    return out


def safe_item_id(entry: Dict, fallback: str) -> str:
    item_id = entry.get("item_id", None)
    return str(item_id) if item_id is not None else fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="paper_plots")
    parser.add_argument("--groups", type=str, default="Hourly,Yearly",
                        help="Exactly two groups, comma-separated (e.g. Hourly,Yearly)")
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--history_max", type=int, default=400,
                        help="Max history points to plot. 0 = plot full history")
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    if len(groups) < 1:
        raise ValueError(f"Groups must be greater than 0")

    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(args.save_dir, args.run_name, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.2,
    })

    n_groups = len(groups)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_groups,
        figsize=(6 * n_groups, 6),
        constrained_layout=True
    )

    if n_groups == 1:
        axes = [axes]

    for ax_i, group in enumerate(groups):
        ax = axes[ax_i]

        test_data, metadata = get_m4_test_dataset(group=group, data_dir=args.data_dir)
        inputs = list(test_data.input)
        labels = list(test_data.label)

        idx = int(rng.integers(0, len(inputs)))
        in_entry = inputs[idx]
        lb_entry = labels[idx]

        y_hist = np.asarray(in_entry["target"], dtype=float)
        y_test = np.asarray(lb_entry["target"], dtype=float)

        if args.history_max and args.history_max > 0 and len(y_hist) > args.history_max:
            y_hist_plot = y_hist[-args.history_max:]
        else:
            y_hist_plot = y_hist

        split = len(y_hist_plot)
        x_hist = np.arange(split)
        x_test = np.arange(split, split + len(y_test))

        ax.plot(x_hist, y_hist_plot, label="History")

        x_test_conn = np.concatenate(([x_hist[-1]], x_test))
        y_test_conn = np.concatenate(([y_hist_plot[-1]], y_test))
        ax.plot(x_test_conn, y_test_conn, label="Test (ground truth)")

        ax.axvline(split - 1, linestyle="--", linewidth=2)

        ax.set_title(f"{DISPLAY_NAMES.get(group, group)} example")
        ax.set_ylabel("Value")

        if ax_i == 1:
            ax.set_xlabel("Time index")

        if ax_i == 0:
            ax.legend(loc="upper left", frameon=True)

        ax.tick_params(axis="both", which="major", length=6, width=1.5)

    safe_groups = "_".join([g.lower() for g in groups])
    out_png = os.path.join(out_dir, f"m4_examples_{safe_groups}_seed{args.seed}.png")
    fig.savefig(out_png, dpi=300)  # higher dpi for PNG
    print(f"Saved: {out_png}")

    if args.save_pdf:
        out_pdf = os.path.join(out_dir, f"m4_examples_{safe_groups}_seed{args.seed}.pdf")
        fig.savefig(out_pdf)  # vector -> best for paper
        print(f"Saved: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()



