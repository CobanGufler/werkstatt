from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import timesfm
from chronos import ChronosPipeline
from gluonts.dataset.common import ListDataset

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


GROUPS = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]

M4_INFO = {
    "Hourly":    {"freq": "H", "h": 48, "seasonality": 24},
    "Daily":     {"freq": "D", "h": 14, "seasonality": 7},
    "Weekly":    {"freq": "W", "h": 13, "seasonality": 52},
    "Monthly":   {"freq": "M", "h": 18, "seasonality": 12},
    "Quarterly": {"freq": "Q", "h": 8,  "seasonality": 4},
    "Yearly":    {"freq": "Y", "h": 6,  "seasonality": 1},
}

torch.manual_seed(54)
np.random.seed(54)


def _read_row_wide_csv(path: str, row_idx: int, chunksize: int = 512) -> pd.Series:
    it = pd.read_csv(path, chunksize=chunksize)
    target_chunk = row_idx // chunksize
    offset_in_chunk = row_idx % chunksize
    for ci, chunk in enumerate(it):
        if ci == target_chunk:
            return chunk.iloc[offset_in_chunk]
    raise IndexError(f"row_idx={row_idx} out of range for {path}")


def load_one_m4_series_wide(data_dir: str, group: str, series_index: int) -> Tuple[dict, dict, str, int]:
    train_path = os.path.join(data_dir, f"{group}-train.csv")
    test_path = os.path.join(data_dir, f"{group}-test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Missing files for group={group} in data_dir={data_dir}\n"
            f"Expected: {train_path} and {test_path}"
        )

    freq = M4_INFO[group]["freq"]
    h = M4_INFO[group]["h"]
    start_period = pd.Period("2000-01-01", freq=freq)

    tr = _read_row_wide_csv(train_path, series_index)
    te = _read_row_wide_csv(test_path, series_index)

    uid = str(tr.iloc[0])
    y_hist = pd.to_numeric(tr.iloc[1:], errors="coerce").to_numpy(dtype=float)
    y_true = pd.to_numeric(te.iloc[1:], errors="coerce").to_numpy(dtype=float)

    y_hist = y_hist[~np.isnan(y_hist)]
    y_true = y_true[~np.isnan(y_true)]
    y_true = y_true[:h]

    in_entry = {"item_id": uid, "start": start_period, "target": y_hist}
    lb_entry = {"item_id": uid, "start": start_period, "target": y_true}
    return in_entry, lb_entry, freq, h


def maybe_truncate(x: np.ndarray, context_len: int) -> np.ndarray:
    if context_len is None or context_len <= 0:
        return x
    if len(x) <= context_len:
        return x
    return x[-context_len:]


def load_timesfm_hf(checkpoint: str, device: str = "cpu"):
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend=device),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )


def forecast_timesfm_one(model, y_hist: np.ndarray, freq: str, h: int, num_jobs: int) -> np.ndarray:
    uid = "series_0"
    start_ts = pd.Timestamp("2000-01-01")
    ds = pd.date_range(start=start_ts, periods=len(y_hist), freq=freq)
    inp_df = pd.DataFrame({"unique_id": uid, "ds": ds, "y": y_hist.astype(float)})
    pred_df = model.forecast_on_df(inputs=inp_df, freq=freq, value_name="y", num_jobs=num_jobs)
    pred_df = pred_df.sort_values(["unique_id", "ds"])
    val_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
    col = "timesfm" if "timesfm" in val_cols else val_cols[0]
    yhat = pred_df[pred_df["unique_id"] == uid][col].to_numpy(dtype=float)[:h]
    return yhat


def forecast_chronos_one(pipeline: ChronosPipeline, y_hist: np.ndarray, h: int, num_samples: int) -> np.ndarray:
    context = torch.tensor(y_hist, dtype=torch.float32)
    samples = pipeline.predict(
        [context],
        prediction_length=h,
        num_samples=num_samples,
        limit_prediction_length=False,
    ).numpy()
    sample_arr = samples[0]
    return np.median(sample_arr, axis=0)


def make_moirai_predictor(repo_id: str, h: int, context_len: int, num_samples: int, batch_size: int):
    module = MoiraiModule.from_pretrained(repo_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=h,
        context_length=context_len,
        patch_size="auto",
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    return model.create_predictor(batch_size=batch_size)


def forecast_moirai_one(predictor, y_hist: np.ndarray, start: pd.Period, freq: str) -> np.ndarray:
    ds = ListDataset(
        [{"item_id": "series_0", "start": start, "target": y_hist.astype(float)}],
        freq=freq,
    )
    fcst = next(iter(predictor.predict(ds)))
    sample_arr = fcst.samples
    return np.median(sample_arr, axis=0)


def plot_group_multi_series(
    out_path: str,
    group: str,
    series_results: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, np.ndarray]]],
):
    n = len(series_results)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3.0, 2.6 * n)), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (uid, y_hist_plot, y_true, preds) in zip(axes, series_results):
        split = len(y_hist_plot)
        x_hist = np.arange(split)
        x_test = np.arange(split, split + len(y_true))

        x_true_conn = np.concatenate(([x_hist[-1]], x_test))
        y_true_conn = np.concatenate(([y_hist_plot[-1]], y_true))

        ax.plot(x_hist, y_hist_plot, label="History")
        ax.plot(x_true_conn, y_true_conn, label="Test (ground truth)")

        styles = [
            dict(linestyle="-", alpha=0.95),
            dict(linestyle="--", alpha=0.95),
            dict(linestyle=":", alpha=0.95),
            dict(linestyle="-.", alpha=0.95),
            dict(linestyle="-", marker="o", markersize=3, alpha=0.80),
        ]
        for i, (name, yhat) in enumerate(preds.items()):
            st = styles[i % len(styles)]
            x_fcst_conn = np.concatenate(([x_hist[-1]], x_test))
            y_fcst_conn = np.concatenate(([y_hist_plot[-1]], yhat))
            ax.plot(x_fcst_conn, y_fcst_conn, label=name, **st)

        ax.axvline(split - 1, linestyle="--", linewidth=1)
        ax.set_title(f"{group} | {uid}")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.suptitle(f"M4 {group} – mehrere Serien", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="plots_m4_multi")
    ap.add_argument("--run_name", type=str, default="multi_series")
    ap.add_argument("--series_indices", type=str, default="0,1,2", help="Comma-separated indices")
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--history_max", type=int, default=300)
    ap.add_argument("--num_jobs", type=int, default=-1)

    ap.add_argument("--variant", type=str, default="base", choices=["base", "small", "all"])

    ap.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m-pytorch")
    ap.add_argument("--chronos_tiny_ckpt", type=str, default="amazon/chronos-t5-tiny")
    ap.add_argument("--chronos_base_ckpt", type=str, default="amazon/chronos-t5-base")
    ap.add_argument("--chronos_samples", type=int, default=10)

    ap.add_argument("--moirai_small_repo", type=str, default="Salesforce/moirai-1.0-R-small")
    ap.add_argument("--moirai_base_repo", type=str, default="Salesforce/moirai-1.0-R-base")
    ap.add_argument("--moirai_samples", type=int, default=10)
    ap.add_argument("--moirai_batch", type=int, default=64)

    args = ap.parse_args()

    out_root = os.path.join(args.save_dir, args.run_name)
    os.makedirs(out_root, exist_ok=True)

    series_indices = [int(x.strip()) for x in args.series_indices.split(",") if x.strip()]

    timesfm_model = load_timesfm_hf(args.timesfm_ckpt, device="cpu")

    chronos_tiny = None
    chronos_base = None
    mo_small_pred = None
    mo_base_pred = None

    if args.variant in ("small", "all"):
        chronos_tiny = ChronosPipeline.from_pretrained(args.chronos_tiny_ckpt, device_map="cpu", torch_dtype=torch.float32)
        mo_small_pred = make_moirai_predictor(args.moirai_small_repo, h=1, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)

    if args.variant in ("base", "all"):
        chronos_base = ChronosPipeline.from_pretrained(args.chronos_base_ckpt, device_map="cpu", torch_dtype=torch.float32)
        mo_base_pred = make_moirai_predictor(args.moirai_base_repo, h=1, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)

    for group in GROUPS:
        series_results = []
        for idx in series_indices:
            in_entry, lb_entry, freq, h = load_one_m4_series_wide(args.data_dir, group, idx)
            uid = in_entry["item_id"]

            y_hist_full = np.asarray(in_entry["target"], dtype=float)
            y_true = np.asarray(lb_entry["target"], dtype=float)

            y_hist_model = maybe_truncate(y_hist_full, args.context_len)
            y_hist_plot = y_hist_model[-args.history_max:] if args.history_max > 0 and len(y_hist_model) > args.history_max else y_hist_model

            preds: Dict[str, np.ndarray] = {}

            preds["TimesFM"] = forecast_timesfm_one(timesfm_model, y_hist_model, freq=freq, h=h, num_jobs=args.num_jobs)

            if chronos_tiny is not None:
                preds["ChronosTiny"] = forecast_chronos_one(chronos_tiny, y_hist_model, h=h, num_samples=args.chronos_samples)
            if chronos_base is not None:
                preds["ChronosBase"] = forecast_chronos_one(chronos_base, y_hist_model, h=h, num_samples=args.chronos_samples)

            if mo_small_pred is not None:
                # recreate predictor with correct horizon per group
                mo_small_pred = make_moirai_predictor(args.moirai_small_repo, h=h, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)
                preds["MoiraiSmall"] = forecast_moirai_one(mo_small_pred, y_hist_model, in_entry["start"], freq=freq)
            if mo_base_pred is not None:
                mo_base_pred = make_moirai_predictor(args.moirai_base_repo, h=h, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)
                preds["MoiraiBase"] = forecast_moirai_one(mo_base_pred, y_hist_model, in_entry["start"], freq=freq)

            series_results.append((uid, y_hist_plot, y_true, preds))

        out_path = os.path.join(out_root, f"{group}_multi_series.png")
        plot_group_multi_series(out_path, group, series_results)
        print(f"Saved: {out_path}")

    print(f"\nDone. Output root: {out_root}")


if __name__ == "__main__":
    main()
