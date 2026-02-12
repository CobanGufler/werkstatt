# Adapted from the official TimesFM implementation (google-research/timesfm).
# License: see model card for google/timesfm-1.0-200m-pytorch
# Modifications: integrated into our unified M4 evaluation pipeline (uni2ts-compatible).

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import timesfm
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast

from uni2ts.eval_util.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MSE, MAE, SMAPE, MASE, ND, NRMSE

from scripts.data_load.m4_gluonts_loader import get_m4_test_dataset

torch.manual_seed(54)

SEASONALITY = {
    "hourly": 24,
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

EPS = 1e-9


def _metrics():
    return [
        MSE(forecast_type="mean"),
        MAE(forecast_type="0.5"),
        SMAPE(forecast_type="0.5"),
        MASE(forecast_type="0.5"),
        ND(forecast_type="0.5"),
        NRMSE(forecast_type="mean"),
    ]


def _pred_start(entry: dict) -> pd.Period:
    return entry["start"] + len(entry["target"])


def load_timesfm_hf(checkpoint: str, device: str = "cpu"):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend=device),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )
    return tfm


def maybe_truncate(x: np.ndarray, context_len: int) -> np.ndarray:
    if context_len is None or context_len <= 0:
        return x
    if len(x) <= context_len:
        return x
    return x[-context_len:]


def _minmax_norm(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    scale = y_max - y_min
    if scale < EPS:
        scale = 1.0
    y_norm = (y - y_min) / scale
    return y_norm, y_min, scale


def _minmax_denorm(samples: np.ndarray, y_min: float, scale: float) -> np.ndarray:
    return samples * scale + y_min


def evaluate_timesfm_m4_uni2ts(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    batch_size: int = 256,
    num_jobs: int = -1,
):
    print("-" * 5, f"Evaluating TimesFM (uni2ts) on M4 {group}", "-" * 5)
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

    model = load_timesfm_hf(checkpoint=checkpoint, device="cpu")
    print(f"Model loaded: {checkpoint} (CPU)")
    print(f"M4 setting: group={group} freq={metadata.freq} h={h} context_len={context_len}")

    inputs = list(test_data.input)
    forecasts: list[SampleForecast] = []

    start_time = time.time()

    for in_batch in tqdm(
        batcher(inputs, batch_size=batch_size),
        total=(len(inputs) + batch_size - 1) // batch_size,
    ):
        rows = []
        uid_order = []
        entries_order = []
        for entry in in_batch:
            uid = entry.get("item_id", None)
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)

            uid_order.append(uid)
            entries_order.append(entry)

            start_ts = entry["start"].to_timestamp()
            ds = pd.date_range(start=start_ts, periods=len(y), freq=metadata.freq)
            rows.extend([(uid, t, float(v)) for t, v in zip(ds, y)])

        inp_df = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

        pred_df = (
            model.forecast_on_df(
                inputs=inp_df,
                freq=metadata.freq,
                value_name="y",
                num_jobs=num_jobs,
            )
            .sort_values(["unique_id", "ds"])
        )

        val_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
        col = "timesfm" if "timesfm" in val_cols else val_cols[0]

        for uid, entry in zip(uid_order, entries_order):
            series_pred = pred_df[pred_df["unique_id"] == uid][col].to_numpy(dtype=float)[:h]
            samples = series_pred[None, :]
            forecasts.append(
                SampleForecast(
                    samples=samples,
                    start_date=_pred_start(entry),
                    item_id=uid,
                )
            )

    end_time = time.time()

    df = evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=_metrics(),
        batch_size=batch_size,
        seasonality=SEASONALITY.get(group.lower(), None),
    )
    df.index = [f"m4_{group}"]
    df["total_time_eval"] = end_time - start_time

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(df)
    print(f"Saved results -> {save_path}")
    print("-" * 5, f"Done TimesFM (uni2ts) on M4 {group}", "-" * 5)
    return df


def evaluate_timesfm_m4_uni2ts_minmax(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    batch_size: int = 64,
    num_jobs: int = -1,
):
    print("-" * 5, f"Evaluating TimesFM (uni2ts, minmax) on M4 {group}", "-" * 5)
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

    model = load_timesfm_hf(checkpoint=checkpoint, device="cpu")
    print(f"Model loaded: {checkpoint} (CPU)")
    print(f"M4 setting: group={group} freq={metadata.freq} h={h} context_len={context_len}")

    inputs = list(test_data.input)
    forecasts: list[SampleForecast] = []

    start_time = time.time()

    for in_batch in tqdm(
        batcher(inputs, batch_size=batch_size),
        total=(len(inputs) + batch_size - 1) // batch_size,
    ):
        rows = []
        uid_order = []
        entries_order = []
        norm_params: list[tuple[float, float]] = []
        for entry in in_batch:
            uid = entry.get("item_id", None)
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)

            y_norm, y_min, scale = _minmax_norm(y)
            norm_params.append((y_min, scale))

            uid_order.append(uid)
            entries_order.append(entry)

            start_ts = entry["start"].to_timestamp()
            ds = pd.date_range(start=start_ts, periods=len(y_norm), freq=metadata.freq)
            rows.extend([(uid, t, float(v)) for t, v in zip(ds, y_norm)])

        inp_df = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

        pred_df = (
            model.forecast_on_df(
                inputs=inp_df,
                freq=metadata.freq,
                value_name="y",
                num_jobs=num_jobs,
            )
            .sort_values(["unique_id", "ds"])
        )

        val_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
        col = "timesfm" if "timesfm" in val_cols else val_cols[0]

        for (uid, entry), (y_min, scale) in zip(zip(uid_order, entries_order), norm_params):
            series_pred = pred_df[pred_df["unique_id"] == uid][col].to_numpy(dtype=float)[:h]
            series_pred = _minmax_denorm(series_pred, y_min, scale)
            samples = series_pred[None, :]
            forecasts.append(
                SampleForecast(
                    samples=samples,
                    start_date=_pred_start(entry),
                    item_id=uid,
                )
            )

    end_time = time.time()

    df = evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=_metrics(),
        batch_size=batch_size,
        seasonality=SEASONALITY.get(group.lower(), None),
    )
    df.index = [f"m4_{group}"]
    df["total_time_eval"] = end_time - start_time

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(df)
    print(f"Saved results -> {save_path}")
    print("-" * 5, f"Done TimesFM (uni2ts, minmax) on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, help="M4 group: Daily/Weekly/Hourly/...")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to M4 datasets folder")
    parser.add_argument("--checkpoint", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--save_dir", type=str, default="results_base")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_jobs", type=int, default=-1)

    args = parser.parse_args()
    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"timesfm_m4_{args.group}.csv")

    evaluate_timesfm_m4_uni2ts(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        save_path=save_path,
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_jobs=args.num_jobs,
    )
