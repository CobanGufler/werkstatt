# Adapted from the official Moirai implementation (SalesforceAIResearch/uni2ts).
# License: see model card for Salesforce/moirai-1.0-R-small and Salesforce/moirai-1.0-R-base
# Modifications: integrated into our unified M4 evaluation pipeline (uni2ts-compatible).

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast

from uni2ts.eval_util.evaluation import evaluate_forecasts
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
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


def evaluate_moirai_m4_uni2ts(
    group: str,
    data_dir: str,
    repo_id: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 20,
    batch_size: int = 256,
):
    print("-" * 5, f"Evaluating Moirai (uni2ts) on M4 {group}", "-" * 5)

    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

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

    predictor = model.create_predictor(batch_size=batch_size)

    print(f"Model loaded: {repo_id} (CPU)")
    print(
        f"M4 setting: group={group} freq={metadata.freq} h={h} "
        f"context_len={context_len} num_samples={num_samples} batch_size={batch_size}"
    )

    inputs = list(test_data.input)

    ds_items = []
    for entry in inputs:
        y = np.asarray(entry["target"], dtype=float)
        y = maybe_truncate(y, context_len)
        ds_items.append(
            {
                "item_id": entry.get("item_id", None),
                "start": entry["start"],
                "target": y,
            }
        )

    ds = ListDataset(ds_items, freq=metadata.freq)

    start_time = time.time()
    forecasts = list(predictor.predict(ds))
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
    print("-" * 5, f"Done Moirai (uni2ts) on M4 {group}", "-" * 5)
    return df


def evaluate_moirai_m4_uni2ts_minmax(
    group: str,
    data_dir: str,
    repo_id: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 10,
    batch_size: int = 64,
):
    print("-" * 5, f"Evaluating Moirai (uni2ts, minmax) on M4 {group}", "-" * 5)

    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

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

    predictor = model.create_predictor(batch_size=batch_size)

    print(f"Model loaded: {repo_id} (CPU)")
    print(
        f"M4 setting: group={group} freq={metadata.freq} h={h} "
        f"context_len={context_len} num_samples={num_samples} batch_size={batch_size}"
    )

    inputs = list(test_data.input)

    ds_items = []
    norm_params: list[tuple[float, float]] = []
    for entry in inputs:
        y = np.asarray(entry["target"], dtype=float)
        y = maybe_truncate(y, context_len)

        y_norm, y_min, scale = _minmax_norm(y)
        norm_params.append((y_min, scale))

        ds_items.append(
            {
                "item_id": entry.get("item_id", None),
                "start": entry["start"],
                "target": y_norm,
            }
        )

    ds = ListDataset(ds_items, freq=metadata.freq)

    start_time = time.time()
    forecasts_raw = list(predictor.predict(ds))
    end_time = time.time()

    forecasts: list[SampleForecast] = []
    for fcst, (y_min, scale) in zip(forecasts_raw, norm_params):
        samples = _minmax_denorm(fcst.samples, y_min, scale)
        forecasts.append(
            SampleForecast(
                samples=samples,
                start_date=fcst.start_date,
                item_id=fcst.item_id,
            )
        )

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
    print("-" * 5, f"Done Moirai (uni2ts, minmax) on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="Salesforce/moirai-1.0-R-base")
    parser.add_argument("--save_dir", type=str, default="results_base")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"moirai_m4_{args.group}.csv")

    evaluate_moirai_m4_uni2ts(
        group=args.group,
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        save_path=save_path,
        context_len=args.context_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
