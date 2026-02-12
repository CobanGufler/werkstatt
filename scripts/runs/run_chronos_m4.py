# Adapted from the official Chronos implementation (amazon-science/chronos-forecasting).
# License: see model card for amazon/chronos-t5-tiny and amazon/chronos-t5-base
# Modifications: integrated into our unified M4 evaluation pipeline (uni2ts-compatible).

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from chronos import ChronosPipeline
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


def evaluate_chronos_m4_uni2ts(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 20,
    batch_size: int = 256,
):
    print("-" * 5, f"Evaluating Chronos (uni2ts) on M4 {group}", "-" * 5)
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

    pipeline = ChronosPipeline.from_pretrained(
        checkpoint,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"Model loaded: {checkpoint} (CPU)")
    print(
        f"M4 setting: group={group} freq={metadata.freq} h={h} "
        f"context_len={context_len} num_samples={num_samples}"
    )

    inputs = list(test_data.input)
    forecasts: list[SampleForecast] = []

    start_time = time.time()

    for in_batch in tqdm(
        batcher(inputs, batch_size=batch_size),
        total=(len(inputs) + batch_size - 1) // batch_size,
    ):
        context_list = []
        entries_order = []
        for entry in in_batch:
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)
            context_list.append(torch.tensor(y, dtype=torch.float32))
            entries_order.append(entry)

        samples = pipeline.predict(
            context_list,
            prediction_length=h,
            num_samples=num_samples,
            limit_prediction_length=False,
        ).numpy()  # (bs, num_samples, h)

        for entry, sample_arr in zip(entries_order, samples):
            forecasts.append(
                SampleForecast(
                    samples=sample_arr,
                    start_date=_pred_start(entry),
                    item_id=entry.get("item_id", None),
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
    print("-" * 5, f"Done Chronos (uni2ts) on M4 {group}", "-" * 5)
    return df


def evaluate_chronos_m4_uni2ts_minmax(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 10,
    batch_size: int = 64,
):
    print("-" * 5, f"Evaluating Chronos (uni2ts, minmax) on M4 {group}", "-" * 5)
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

    pipeline = ChronosPipeline.from_pretrained(
        checkpoint,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"Model loaded: {checkpoint} (CPU)")
    print(
        f"M4 setting: group={group} freq={metadata.freq} h={h} "
        f"context_len={context_len} num_samples={num_samples}"
    )

    inputs = list(test_data.input)
    forecasts: list[SampleForecast] = []

    start_time = time.time()

    for in_batch in tqdm(
        batcher(inputs, batch_size=batch_size),
        total=(len(inputs) + batch_size - 1) // batch_size,
    ):
        context_list = []
        entries_order = []
        norm_params: list[tuple[float, float]] = []
        for entry in in_batch:
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)

            y_norm, y_min, scale = _minmax_norm(y)
            norm_params.append((y_min, scale))

            context_list.append(torch.tensor(y_norm, dtype=torch.float32))
            entries_order.append(entry)

        samples = pipeline.predict(
            context_list,
            prediction_length=h,
            num_samples=num_samples,
            limit_prediction_length=False,
        ).numpy()  # (bs, num_samples, h)

        for (entry, sample_arr), (y_min, scale) in zip(zip(entries_order, samples), norm_params):
            sample_arr = _minmax_denorm(sample_arr, y_min, scale)
            forecasts.append(
                SampleForecast(
                    samples=sample_arr,
                    start_date=_pred_start(entry),
                    item_id=entry.get("item_id", None),
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
    print("-" * 5, f"Done Chronos (uni2ts, minmax) on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="amazon/chronos-t5-base")
    parser.add_argument("--save_dir", type=str, default="results_base")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"chronos_m4_{args.group}.csv")

    evaluate_chronos_m4_uni2ts(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        save_path=save_path,
        context_len=args.context_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
