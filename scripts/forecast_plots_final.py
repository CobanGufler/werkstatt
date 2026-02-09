from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

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
    test_path  = os.path.join(data_dir, f"{group}-test.csv")
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


def forecast_chronos_one(pipeline: ChronosPipeline, y_hist: np.ndarray, h: int, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    context = torch.tensor(y_hist, dtype=torch.float32)
    samples = pipeline.predict(
        [context],
        prediction_length=h,
        num_samples=num_samples,
        limit_prediction_length=False,
    ).numpy()
    sample_arr = samples[0]
    median = np.median(sample_arr, axis=0)
    return median, sample_arr


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


def forecast_moirai_one(predictor, y_hist: np.ndarray, start: pd.Period, freq: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = ListDataset(
        [{"item_id": "series_0", "start": start, "target": y_hist.astype(float)}],
        freq=freq,
    )
    fcst = next(iter(predictor.predict(ds)))
    sample_arr = fcst.samples
    median = np.median(sample_arr, axis=0)
    return median, sample_arr


def smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    num = 2.0 * np.abs(yhat - y)
    den = np.abs(y) + np.abs(yhat) + eps
    return float(np.mean(num / den))


def mase_denom(y_hist: np.ndarray, m: int, eps: float = 1e-8) -> float:
    n = len(y_hist)
    if m < 1:
        m = 1
    if n <= m:
        if n <= 1:
            return 1.0
        return float(np.mean(np.abs(np.diff(y_hist))) + eps)
    return float(np.mean(np.abs(y_hist[m:] - y_hist[:-m])) + eps)


def compute_metrics(y_true: np.ndarray, yhat: np.ndarray, y_hist_for_mase: np.ndarray, seasonality_m: int, eps: float = 1e-8) -> Dict[str, float]:
    err = yhat - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    nd = float(np.sum(np.abs(err)) / (np.sum(np.abs(y_true)) + eps))
    nrmse = float(np.sqrt(mse) / (np.mean(np.abs(y_true)) + eps))
    sm = smape(y_true, yhat, eps=eps)
    denom = mase_denom(y_hist_for_mase, seasonality_m, eps=eps)
    mase = float(mae / denom)
    return {"MSE": mse, "MAE": mae, "ND": nd, "NRMSE": nrmse, "sMAPE": sm, "MASE": mase}


def metrics_block(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"MAE:   {m['MAE']:.4g}\n"
        f"sMAPE: {m['sMAPE']:.4g}\n"
        f"MASE:  {m['MASE']:.4g}\n"
        f"NRMSE: {m['NRMSE']:.4g}\n"
        f"ND:    {m['ND']:.4g}\n"
        f"MSE:   {m['MSE']:.4g}"
    )


def plot_one_panel_with_metrics(
    out_path: str,
    title: str,
    y_hist_plot: np.ndarray,
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
):
    split = len(y_hist_plot)
    x_hist = np.arange(split)
    x_test = np.arange(split, split + len(y_true))

    x_true_conn = np.concatenate(([x_hist[-1]], x_test))
    y_true_conn = np.concatenate(([y_hist_plot[-1]], y_true))

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.6], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_m.axis("off")

    ax.plot(x_hist, y_hist_plot, label="History")
    ax.plot(x_true_conn, y_true_conn, label="Test (ground truth)")

    styles = [
        dict(linestyle="-",  marker=None, alpha=0.95),
        dict(linestyle="--", marker=None, alpha=0.95),
        dict(linestyle=":",  marker=None, alpha=0.95),
        dict(linestyle="-.", marker=None, alpha=0.95),
        dict(linestyle="-",  marker="o", markersize=3, alpha=0.80),
        dict(linestyle="--", marker="s", markersize=3, alpha=0.80),
    ]

    for i, (name, yhat) in enumerate(preds.items()):
        st = styles[i % len(styles)]
        x_fcst_conn = np.concatenate(([x_hist[-1]], x_test))
        y_fcst_conn = np.concatenate(([y_hist_plot[-1]], yhat))
        ax.plot(x_fcst_conn, y_fcst_conn, label=name, **st)

    ax.axvline(split - 1, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)

    blocks = [metrics_block(name, mm) for name, mm in metrics.items()]
    ax_m.text(0.0, 1.0, "\n\n".join(blocks), va="top", ha="left", transform=ax_m.transAxes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="plots_m4")
    parser.add_argument("--run_name", type=str, default="uni2ts_variants")

    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--history_max", type=int, default=400)
    parser.add_argument("--num_jobs", type=int, default=-1)

    parser.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m-pytorch")

    parser.add_argument("--chronos_tiny_ckpt", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--chronos_base_ckpt", type=str, default="amazon/chronos-t5-base")
    parser.add_argument("--chronos_samples", type=int, default=10)

    parser.add_argument("--moirai_small_repo", type=str, default="Salesforce/moirai-1.0-R-small")
    parser.add_argument("--moirai_base_repo", type=str, default="Salesforce/moirai-1.0-R-base")
    parser.add_argument("--moirai_samples", type=int, default=10)
    parser.add_argument("--moirai_batch", type=int, default=64)

    args = parser.parse_args()

    out_root = os.path.join(args.save_dir, args.run_name)
    os.makedirs(out_root, exist_ok=True)

    timesfm_model = load_timesfm_hf(args.timesfm_ckpt, device="cpu")
    chronos_tiny = ChronosPipeline.from_pretrained(args.chronos_tiny_ckpt, device_map="cpu", torch_dtype=torch.float32)
    chronos_base = ChronosPipeline.from_pretrained(args.chronos_base_ckpt, device_map="cpu", torch_dtype=torch.float32)

    for group in GROUPS:
        in_entry, lb_entry, freq, h = load_one_m4_series_wide(args.data_dir, group, args.series_index)
        uid = in_entry["item_id"]

        y_hist_full = np.asarray(in_entry["target"], dtype=float)
        y_true = np.asarray(lb_entry["target"], dtype=float)

        y_hist_model = maybe_truncate(y_hist_full, args.context_len)
        y_hist_plot = y_hist_model[-args.history_max:] if args.history_max > 0 and len(y_hist_model) > args.history_max else y_hist_model

        yhat_timesfm = forecast_timesfm_one(timesfm_model, y_hist_model, freq=freq, h=h, num_jobs=args.num_jobs)

        yhat_ch_tiny, ch_tiny_samples = forecast_chronos_one(chronos_tiny, y_hist_model, h=h, num_samples=args.chronos_samples)
        yhat_ch_base, ch_base_samples = forecast_chronos_one(chronos_base, y_hist_model, h=h, num_samples=args.chronos_samples)

        mo_small_pred = make_moirai_predictor(args.moirai_small_repo, h=h, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)
        mo_base_pred = make_moirai_predictor(args.moirai_base_repo,  h=h, context_len=args.context_len, num_samples=args.moirai_samples, batch_size=args.moirai_batch)

        yhat_mo_small, mo_small_samples = forecast_moirai_one(mo_small_pred, y_hist_model, in_entry["start"], freq=freq)
        yhat_mo_base,  mo_base_samples  = forecast_moirai_one(mo_base_pred,  y_hist_model, in_entry["start"], freq=freq)

        m = M4_INFO[group]["seasonality"]

        met_timesfm  = compute_metrics(y_true, yhat_timesfm,  y_hist_full, m)
        met_ch_tiny  = compute_metrics(y_true, yhat_ch_tiny,  y_hist_full, m)
        met_ch_base  = compute_metrics(y_true, yhat_ch_base,  y_hist_full, m)
        met_mo_small = compute_metrics(y_true, yhat_mo_small, y_hist_full, m)
        met_mo_base  = compute_metrics(y_true, yhat_mo_base,  y_hist_full, m)

        print(f"\n[{group}] uid={uid} idx={args.series_index}")
        for name, mm in [
            ("TimesFM", met_timesfm),
            ("ChronosTiny", met_ch_tiny),
            ("ChronosBase", met_ch_base),
            ("MoiraiSmall", met_mo_small),
            ("MoiraiBase", met_mo_base),
        ]:
            print(f"  {name:11s}  MAE={mm['MAE']:.4g}  sMAPE={mm['sMAPE']:.4g}  MASE={mm['MASE']:.4g}  NRMSE={mm['NRMSE']:.4g}  ND={mm['ND']:.4g}")

        group_dir = os.path.join(out_root, group)
        os.makedirs(group_dir, exist_ok=True)

        # Plot 1: TimesFM + ChronosTiny + MoiraiSmall
        plot_one_panel_with_metrics(
            out_path=os.path.join(group_dir, f"{group}_{uid}_P1_timesfm_chronosTiny_moiraiSmall.png"),
            title=f"M4 {group} | {uid} | TimesFM + ChronosTiny + MoiraiSmall",
            y_hist_plot=y_hist_plot,
            y_true=y_true,
            preds={"TimesFM": yhat_timesfm, "ChronosTiny": yhat_ch_tiny, "MoiraiSmall": yhat_mo_small},
            metrics={"TimesFM": met_timesfm, "ChronosTiny": met_ch_tiny, "MoiraiSmall": met_mo_small},
        )

        # Plot 2: TimesFM + ChronosBase + MoiraiBase
        plot_one_panel_with_metrics(
            out_path=os.path.join(group_dir, f"{group}_{uid}_P2_timesfm_chronosBase_moiraiBase.png"),
            title=f"M4 {group} | {uid} | TimesFM + ChronosBase + MoiraiBase",
            y_hist_plot=y_hist_plot,
            y_true=y_true,
            preds={"TimesFM": yhat_timesfm, "ChronosBase": yhat_ch_base, "MoiraiBase": yhat_mo_base},
            metrics={"TimesFM": met_timesfm, "ChronosBase": met_ch_base, "MoiraiBase": met_mo_base},
        )

        # Plot 3: ONLY Moirai versions
        plot_one_panel_with_metrics(
            out_path=os.path.join(group_dir, f"{group}_{uid}_P3_moirai_versions.png"),
            title=f"M4 {group} | {uid} | Moirai (Small vs Base)",
            y_hist_plot=y_hist_plot,
            y_true=y_true,
            preds={"MoiraiSmall": yhat_mo_small, "MoiraiBase": yhat_mo_base},
            metrics={"MoiraiSmall": met_mo_small, "MoiraiBase": met_mo_base},
        )

        # Plot 4: ONLY Chronos versions
        plot_one_panel_with_metrics(
            out_path=os.path.join(group_dir, f"{group}_{uid}_P4_chronos_versions.png"),
            title=f"M4 {group} | {uid} | Chronos (Tiny vs Base)",
            y_hist_plot=y_hist_plot,
            y_true=y_true,
            preds={"ChronosTiny": yhat_ch_tiny, "ChronosBase": yhat_ch_base},
            metrics={"ChronosTiny": met_ch_tiny, "ChronosBase": met_ch_base},
        )

        print(f"  -> saved 4 plots to {group_dir}")

    print(f"\nDone. Output root: {out_root}")


if __name__ == "__main__":
    main()


