"""Utilities to load M4 data into GluonTS datasets.

Supports local M4 CSVs (wide or long format). If the files are missing,
we fall back to the datasetsforecast loader and split the last horizon.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split


# M4 frequency and prediction horizon per group
M4_INFO = {
    "Hourly":    {"freq": "H", "h": 48},
    "Daily":     {"freq": "D", "h": 14},
    "Weekly":    {"freq": "W", "h": 13},
    "Monthly":   {"freq": "M", "h": 18},
    "Quarterly": {"freq": "Q", "h": 8},
    "Yearly":    {"freq": "Y", "h": 6},
}


@dataclass
class SimpleMetadata:
    freq: str
    prediction_length: int


def _wide_to_series_items(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[dict]:
    """
    Converts M4 wide CSVs:
      - first col = id (often "V1")
      - remaining cols = values
    into ListDataset items with full target = train + test.
    """
    id_col = train_df.columns[0]

    # values (drop id col)
    train_vals = train_df.drop(columns=[id_col]).to_numpy(dtype=float)
    test_vals = test_df.drop(columns=[id_col]).to_numpy(dtype=float)

    # ids
    ids = train_df[id_col].astype(str).to_list()

    items = []
    for i, uid in enumerate(ids):
        y_full = np.concatenate([train_vals[i], test_vals[i]], axis=0)

        # M4 can contain trailing NaNs in wide format; remove them
        # (keeps actual length per series)
        if np.isnan(y_full).any():
            y_full = y_full[~np.isnan(y_full)]

        items.append({"item_id": uid, "target": y_full})

    return items


def _long_to_series_items(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[dict]:
    """Converts long format unique_id/ds/y into full target arrays."""
    train_df = train_df.sort_values(["unique_id", "ds"])
    test_df = test_df.sort_values(["unique_id", "ds"])

    items = []
    for uid, gtr in train_df.groupby("unique_id", sort=False):
        y_tr = pd.to_numeric(gtr["y"], errors="raise").to_numpy(dtype=float)
        gte = test_df[test_df["unique_id"] == uid]
        y_te = pd.to_numeric(gte["y"], errors="raise").to_numpy(dtype=float)
        y_full = np.concatenate([y_tr, y_te], axis=0)
        items.append({"item_id": str(uid), "target": y_full})
    return items


def get_m4_test_dataset(group: str, data_dir: str, start: str = "2000-01-01") -> Tuple[object, SimpleMetadata]:
    """Build a GluonTS test dataset and metadata for the given M4 group."""
    if group not in M4_INFO:
        raise ValueError(f"Unknown M4 group '{group}'. Choose from {list(M4_INFO.keys())}")

    freq = M4_INFO[group]["freq"]
    h = M4_INFO[group]["h"]

    train_path = os.path.join(data_dir, f"{group}-train.csv")
    test_path = os.path.join(data_dir, f"{group}-test.csv")

    # 1) Prefer local CSVs if present
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Detect format
        is_long = {"unique_id", "ds", "y"}.issubset(set(train_df.columns))
        if is_long:
            series_items = _long_to_series_items(train_df, test_df)
        else:
            series_items = _wide_to_series_items(train_df, test_df)

    else:
        # 2) fallback: datasetsforecast loader
        out = M4.load(directory=data_dir, group=group)
        full_df = out[0] if isinstance(out, (tuple, list)) else out
        full_df = full_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # split last h
        test_df = full_df.groupby("unique_id", sort=False).tail(h).copy()
        train_df = full_df.drop(test_df.index).copy()

        series_items = _long_to_series_items(train_df, test_df)

    # Build ListDataset
    if freq == "Y":
        # Avoid pandas Period bounds by choosing a safe start year.
        max_len = max(len(it["target"]) for it in series_items)
        safe_end_year = 2262
        min_year = 1700
        start_year = max(min_year, safe_end_year - (max_len - 1))
        start = f"{start_year}-01-01"

    start_period = pd.Period(start, freq=freq)
    ds = ListDataset(
        [{"item_id": it["item_id"], "start": start_period, "target": it["target"]} for it in series_items],
        freq=freq,
    )

    # Split input/label (label = last h)
    _, test_gen = split(ds, offset=-h)
    test_data = test_gen.generate_instances(prediction_length=h, windows=1)

    metadata = SimpleMetadata(freq=freq, prediction_length=h)
    return test_data, metadata
