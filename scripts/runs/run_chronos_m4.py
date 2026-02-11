# run_chronos_m4.py
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

from scripts.data_load.m4_gluonts_loader import get_m4_test_dataset
from src.eval.metrics import RunningMetrics

torch.manual_seed(54)

def maybe_truncate(x: np.ndarray, context_len: int) -> np.ndarray:
    if context_len is None or context_len <= 0:
        return x
    if len(x) <= context_len:
        return x
    return x[-context_len:]


def evaluate_chronos_m4(
    group: str,
    data_dir: str,
    checkpoint: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 5,
    batch_size: int = 256,
):
    print("-" * 5, f"Evaluating Chronos on M4 {group}", "-" * 5)
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
    labels = list(test_data.label)

    metrics = RunningMetrics()

    start_time = time.time()
    for in_batch, lb_batch in tqdm(zip(batcher(inputs, batch_size=batch_size),
                                       batcher(labels, batch_size=batch_size)),
                                  total=(len(inputs) + batch_size - 1) // batch_size):
        context_list = []
        for entry in in_batch:
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)
            context_list.append(torch.tensor(y, dtype=torch.float32))

        samples = pipeline.predict(
            context_list,
            prediction_length=h,
            num_samples=num_samples,
            limit_prediction_length=False,
        ).numpy()  # (bs, num_samples, h) typically

        # point forecast: median over samples
        yhat = np.median(samples, axis=1)  # (bs, h)

        ytrue = np.stack([np.asarray(e["target"], dtype=float) for e in lb_batch], axis=0)

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
    print("-" * 5, f"Done Chronos on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"chronos_m4_{args.group}.csv")

    evaluate_chronos_m4(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        save_path=save_path,
        context_len=args.context_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
