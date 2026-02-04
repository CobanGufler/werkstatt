# run_all_m4_uni2ts_minmax.py
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import timesfm
from chronos import ChronosPipeline
from gluonts.dataset.common import ListDataset
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast

from uni2ts.eval_util.evaluation import evaluate_forecasts
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.ev.metrics import MSE, MAE, SMAPE, MASE, ND, NRMSE

from src.data_load.m4_gluonts_loader import get_m4_test_dataset

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


def maybe_truncate(x: np.ndarray, context_len: int) -> np.ndarray:
    if context_len is None or context_len <= 0:
        return x
    if len(x) <= context_len:
        return x
    return x[-context_len:]


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


def load_timesfm_hf(checkpoint: str, device: str = "cpu"):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend=device),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )
    return tfm


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

    for in_batch in tqdm(batcher(inputs, batch_size=batch_size),
                         total=(len(inputs) + batch_size - 1) // batch_size):
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

        pred_df = model.forecast_on_df(
            inputs=inp_df,
            freq=metadata.freq,
            value_name="y",
            num_jobs=num_jobs,
        ).sort_values(["unique_id", "ds"])

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
    print(f"M4 setting: group={group} freq={metadata.freq} h={h} context_len={context_len} num_samples={num_samples}")

    inputs = list(test_data.input)

    forecasts: list[SampleForecast] = []
    start_time = time.time()

    for in_batch in tqdm(batcher(inputs, batch_size=batch_size),
                         total=(len(inputs) + batch_size - 1) // batch_size):
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
    parser.add_argument("--save_dir", type=str, default="results_base")
    parser.add_argument("--run_name", type=str, default="test")

    parser.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--chronos_ckpt", type=str, default="amazon/chronos-t5-base")
    parser.add_argument("--moirai_repo", type=str, default="Salesforce/moirai-1.0-R-base")

    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chronos_samples", type=int, default=10)
    parser.add_argument("--moirai_samples", type=int, default=10)
    parser.add_argument("--moirai_batch", type=int, default=64)

    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    param_tag = (
        f"minmax_c{args.context_len}_b{args.batch_size}_"
        f"cs{args.chronos_samples}_ms{args.moirai_samples}_mb{args.moirai_batch}"
    )

    df1 = evaluate_timesfm_m4_uni2ts_minmax(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.timesfm_ckpt,
        save_path=os.path.join(out_dir, f"timesfm_m4_{args.group}_uni2ts_{param_tag}.csv"),
        context_len=args.context_len,
        batch_size=args.batch_size,
    )

    df2 = evaluate_chronos_m4_uni2ts_minmax(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.chronos_ckpt,
        save_path=os.path.join(out_dir, f"chronos_m4_{args.group}_uni2ts_{param_tag}.csv"),
        context_len=args.context_len,
        num_samples=args.chronos_samples,
        batch_size=args.batch_size,
    )

    df3 = evaluate_moirai_m4_uni2ts_minmax(
        group=args.group,
        data_dir=args.data_dir,
        repo_id=args.moirai_repo,
        save_path=os.path.join(out_dir, f"moirai_m4_{args.group}_uni2ts_{param_tag}.csv"),
        context_len=args.context_len,
        num_samples=args.moirai_samples,
        batch_size=args.moirai_batch,
    )

    merged = pd.concat([df1, df2, df3], axis=0)
    merged.index = [
        f"timesfm_m4_{args.group}_uni2ts_minmax",
        f"chronos_m4_{args.group}_uni2ts_minmax",
        f"moirai_m4_{args.group}_uni2ts_minmax",
    ]
    merged.to_csv(os.path.join(out_dir, f"ALL_m4_{args.group}_uni2ts_{param_tag}.csv"))
    print(merged)
