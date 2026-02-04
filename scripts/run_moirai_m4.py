# run_moirai_m4.py
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from gluonts.dataset.common import ListDataset

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from src.data_load.m4_gluonts_loader import get_m4_test_dataset
from src.eval.metrics import RunningMetrics

torch.manual_seed(54)

def maybe_truncate(x: np.ndarray, context_len: int) -> np.ndarray:
    if context_len is None or context_len <= 0:
        return x
    if len(x) <= context_len:
        return x
    return x[-context_len:]


def evaluate_moirai_m4(
    group: str,
    data_dir: str,
    repo_id: str,
    save_path: str,
    context_len: int = 512,
    num_samples: int = 20,
    batch_size: int = 32,
):
    print("-" * 5, f"Evaluating Moirai on M4 {group}", "-" * 5)

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
    labels = list(test_data.label)

    # Build ListDataset from history-only inputs and keep the corresponding (truncated) insample arrays
    start_time = time.time()
    ds_items = []
    insample_list: list[np.ndarray] = []

    for entry in inputs:
        y = np.asarray(entry["target"], dtype=float)
        y = maybe_truncate(y, context_len)
        insample_list.append(y)
        ds_items.append(
            {
                "item_id": entry.get("item_id", None),
                "start": entry["start"],
                "target": y,
            }
        )

    ds = ListDataset(ds_items, freq=metadata.freq)

    metrics = RunningMetrics()

    SEASONALITY = {
        "hourly": 24,
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }
    m = SEASONALITY.get(group.lower(), 1)

    forecasts = predictor.predict(ds)

    for i, (fcst, lb) in enumerate(tqdm(zip(forecasts, labels), total=len(labels))):
        # fcst.samples: (num_samples, h)
        yhat = np.median(fcst.samples, axis=0)[None, :]  # (1, h)
        ytrue = np.asarray(lb["target"], dtype=float)[None, :]  # (1, h)

        insample = insample_list[i][None, :]  # (1, context_len_truncated)

        metrics.update(yhat, ytrue)
        metrics.update_mase(yhat, ytrue, insample=insample, m=m)


    end_time = time.time()

    result = metrics.finalize()
    result["total_time_eval"] = end_time - start_time

    df = pd.DataFrame(result, index=[f"m4_{group}"])
    print(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Saved results -> {save_path}")
    print("-" * 5, f"Done Moirai on M4 {group}", "-" * 5)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="Salesforce/moirai-1.0-R-small")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, args.run_name)
    save_path = os.path.join(out_dir, f"moirai_m4_{args.group}.csv")

    evaluate_moirai_m4(
        group=args.group,
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        save_path=save_path,
        context_len=args.context_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

