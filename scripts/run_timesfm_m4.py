# run_timesfm_m4.py
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

from src.data_load.m4_gluonts_loader import get_m4_test_dataset
from src.eval.metrics import RunningMetrics

torch.manual_seed(54)

def load_timesfm_hf(checkpoint: str, device: str = "cpu"):
    # HF PyTorch checkpoint (wie du es bisher nutzt)
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


def evaluate_timesfm_m4(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    batch_size: int = 256,
    num_jobs: int = -1,
):
    print("-" * 5, f"Evaluating TimesFM on M4 {group}", "-" * 5)
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length

    model = load_timesfm_hf(checkpoint=checkpoint, device="cpu")
    print(f"Model loaded: {checkpoint} (CPU)")
    print(f"M4 setting: group={group} freq={metadata.freq} h={h} context_len={context_len}")

    inputs = list(test_data.input)
    labels = list(test_data.label)

    # We’ll predict batch-wise, using forecast_on_df per batch.
    # We build a small DF per batch (only context_len points per series).
    metrics = RunningMetrics()

    start_time = time.time()
    for in_batch, lb_batch in tqdm(zip(batcher(inputs, batch_size=batch_size),
                                       batcher(labels, batch_size=batch_size)),
                                  total=(len(inputs) + batch_size - 1) // batch_size):
        # Build batch input df
        rows = []
        uid_order = []
        for entry in in_batch:
            uid = entry.get("item_id", None)
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)

            uid_order.append(uid)
            start_ts = entry["start"].to_timestamp()
            ds = pd.date_range(start=start_ts, periods=len(y), freq=metadata.freq)
            rows.extend([(uid, t, float(v)) for t, v in zip(ds, y)])

        inp_df = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

        # Forecast
        pred_df = model.forecast_on_df(
            inputs=inp_df,
            freq=metadata.freq,
            value_name="y",
            num_jobs=num_jobs,
        )
        pred_df = pred_df.sort_values(["unique_id", "ds"])

        val_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
        col = "timesfm" if "timesfm" in val_cols else val_cols[0]

        # Align predictions to batch order
        yhat = np.stack(
            [
                pred_df[pred_df["unique_id"] == uid][col].to_numpy(dtype=float)[:h]
                for uid in uid_order
            ],
            axis=0,
        )

        ytrue = np.stack(
            [np.asarray(e["target"], dtype=float) for e in lb_batch],
            axis=0,
        )
        SEASONALITY = {
            "hourly": 24,
            "daily": 7,
            "weekly": 52,
            "monthly": 12,
            "quarterly": 4,
            "yearly": 1,
        }

        insample = np.stack([np.asarray(e["target"], dtype=float) for e in in_batch], axis=0)

        metrics.update(yhat, ytrue)
        metrics.update_mase(yhat, ytrue, insample=insample, m=SEASONALITY.get(group.lower(), 1))

    end_time = time.time()

    result = metrics.finalize()
    result["total_time_eval"] = end_time - start_time

    df = pd.DataFrame(result, index=[f"m4_{group}"])
    print(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Saved results -> {save_path}")
    print("-" * 5, f"Done TimesFM on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, help="M4 group: Daily/Weekly/Hourly/...")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to M4 datasets folder")
    parser.add_argument("--checkpoint", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_jobs", type=int, default=-1)

    args = parser.parse_args()
    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"timesfm_m4_{args.group}.csv")

    evaluate_timesfm_m4(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        save_path=save_path,
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_jobs=args.num_jobs,
    )
