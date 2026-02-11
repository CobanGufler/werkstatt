from __future__ import annotations

import argparse
import os
import time
import glob
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from gluonts.ev.metrics import MSE, MAE, SMAPE, MASE, ND, NRMSE

from scripts.data_load.m4_gluonts_loader import get_m4_test_dataset
from uni2ts.eval_util.evaluation import evaluate_forecasts


SEASONALITY = {
    "hourly": 24,
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

FREQ_WEIGHTS = {
    "Hourly": 414,
    "Daily": 4227,
    "Weekly": 359,
    "Monthly": 48000,
    "Quarterly": 24000,
    "Yearly": 23000,
}

FREQ_ORDER = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]


def canon_group(g: str) -> str:
    """Make group keys consistent: 'daily'/'Daily' -> 'Daily'."""
    s = str(g).strip()
    if not s:
        return s
    s = s.lower()
    return s[0].upper() + s[1:]


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


def naive2_forecast(y: np.ndarray, h: int, m: Optional[int]) -> np.ndarray:
    """
    M4-style Naive2 (operationally Seasonal Naive):
      - m <= 1: repeat last value
      - m  > 1: repeat last m values (last season), tiled to horizon h
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(h, dtype=float)

    if m is None or m <= 1:
        return np.full(h, y[-1], dtype=float)

    last_season = y[-m:]
    reps = int(np.ceil(h / m))
    fc = np.tile(last_season, reps)[:h]
    return fc.astype(float)


def evaluate_naive2_m4_uni2ts(
    group: str,
    data_dir: str,
    save_path: str,
    context_len: int = 512,
    batch_size: int = 64,
) -> pd.DataFrame:
    print("-" * 5, f"Evaluating Naive2 (uni2ts) on M4 {group}", "-" * 5)

    # NOTE: group must be "Daily"/"Hourly"/... for loader
    test_data, metadata = get_m4_test_dataset(group=group, data_dir=data_dir)
    h = metadata.prediction_length
    m = SEASONALITY.get(group.lower(), None)

    inputs = list(test_data.input)
    forecasts: list[SampleForecast] = []

    start_time = time.time()
    for in_batch in tqdm(
        batcher(inputs, batch_size=batch_size),
        total=(len(inputs) + batch_size - 1) // batch_size
    ):
        for entry in in_batch:
            y = np.asarray(entry["target"], dtype=float)
            y = maybe_truncate(y, context_len)
            pred = naive2_forecast(y, h=h, m=m)
            samples = pred[None, :]  # (1, h)

            forecasts.append(
                SampleForecast(
                    samples=samples,
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
        seasonality=m,
    )
    df.index = [f"naive2_m4_{group}_uni2ts"]
    df["total_time_eval"] = end_time - start_time

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(df)
    print(f"Saved results -> {save_path}")
    print("-" * 5, f"Done Naive2 (uni2ts) on M4 {group}", "-" * 5)
    return df


def _latest_all_csv(freq_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(freq_dir, "ALL_m4_*uni2ts*.csv"))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def read_base_all_results(results_base_dir: str) -> pd.DataFrame:
    rows = []
    for group_folder in os.listdir(results_base_dir):
        gdir = os.path.join(results_base_dir, group_folder)
        if not os.path.isdir(gdir):
            continue

        all_path = _latest_all_csv(gdir)
        if all_path is None:
            continue

        df = pd.read_csv(all_path)

        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "model_id"})
        elif df.columns[0] != "model_id":
            df = df.rename(columns={df.columns[0]: "model_id"})

        mase_col = _find_col(df, ["MASE[0.5]", "MASE", "mase"])
        smape_col = _find_col(df, ["sMAPE[0.5]", "SMAPE", "smape", "smapE", "sMape"])
        if mase_col is None or smape_col is None:
            raise ValueError(f"Missing MASE/sMAPE in {all_path}. Columns={list(df.columns)}")

        def pretty(mid: str) -> str:
            s = str(mid).lower()
            if "timesfm" in s:
                return "TimesFM"
            if "chronos" in s:
                return "Chronos Base"
            if "moirai" in s:
                return "Moirai Base"
            return str(mid)

        df["model"] = df["model_id"].apply(pretty)
        df = df[df["model"].isin(["TimesFM", "Chronos Base", "Moirai Base"])].copy()

        group = canon_group(group_folder)
        for _, r in df.iterrows():
            rows.append(
                {
                    "group": group,
                    "model": r["model"],
                    "MASE": float(r[mase_col]),
                    "sMAPE": float(r[smape_col]),
                    "source_all_csv": os.path.basename(all_path),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No base model rows found in results_base_dir.")
    return out


def compute_owa(model_mase: float, model_smape: float, naive_mase: float, naive_smape: float) -> float:
    eps = 1e-12
    if not np.isfinite(naive_smape) or not np.isfinite(naive_mase) or naive_smape <= eps or naive_mase <= eps:
        return np.nan
    return 0.5 * (model_smape / naive_smape) + 0.5 * (model_mase / naive_mase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--results_base_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="owa_naive2_uni2ts")

    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Naive2 per group
    naive_rows = []
    for g in FREQ_ORDER:
        save_path = os.path.join(
            args.out_dir,
            g,
            f"naive2_m4_{g}_uni2ts_c{args.context_len}_b{args.batch_size}.csv",
        )
        df = evaluate_naive2_m4_uni2ts(
            group=g,
            data_dir=args.data_dir,
            save_path=save_path,
            context_len=args.context_len,
            batch_size=args.batch_size,
        )

        # Extract MASE/sMAPE columns robustly
        df0 = df.reset_index(drop=True)
        mase_col = _find_col(df0, ["MASE[0.5]", "MASE"])
        smape_col = _find_col(df0, ["sMAPE[0.5]", "SMAPE"])
        if mase_col is None or smape_col is None:
            raise RuntimeError(f"Naive2 df missing MASE/sMAPE for group={g}. Columns={list(df.columns)}")

        naive_rows.append(
            {
                "group": canon_group(g),
                "MASE_Naive2": float(df.iloc[0][mase_col]),
                "sMAPE_Naive2": float(df.iloc[0][smape_col]),
                "total_time_eval": float(df.iloc[0]["total_time_eval"]) if "total_time_eval" in df.columns else np.nan,
            }
        )

    naive_df = pd.DataFrame(naive_rows)
    naive_df.to_csv(os.path.join(args.out_dir, "naive2_metrics_per_group_uni2ts.csv"), index=False)

    # 2) Base results
    base_df = read_base_all_results(args.results_base_dir)

    # 3) Merge + sanity check
    merged = base_df.merge(naive_df, on="group", how="left")

    missing = merged[merged["MASE_Naive2"].isna() | merged["sMAPE_Naive2"].isna()]["group"].unique().tolist()
    if missing:
        raise RuntimeError(
            "Merge failed for some groups (Naive2 missing). "
            f"Groups with missing Naive2 after merge: {missing}. "
            "This usually means inconsistent group naming in folders."
        )

    merged["OWA"] = merged.apply(
        lambda r: compute_owa(r["MASE"], r["sMAPE"], r["MASE_Naive2"], r["sMAPE_Naive2"]),
        axis=1,
    )
    merged.to_csv(os.path.join(args.out_dir, "owa_per_group_and_model.csv"), index=False)

    # 4) Weighted overall OWA
    overall_rows = []
    for model, g in merged.groupby("model", sort=False):
        weights = np.array([float(FREQ_WEIGHTS.get(grp, 1.0)) for grp in g["group"].tolist()], dtype=float)
        owas = g["OWA"].to_numpy(dtype=float)
        mask = np.isfinite(weights) & np.isfinite(owas)
        overall = float(np.sum(weights[mask] * owas[mask]) / np.sum(weights[mask])) if mask.sum() else np.nan
        overall_rows.append({"model": model, "OWA_overall_weighted": overall})

    overall_df = pd.DataFrame(overall_rows).sort_values("OWA_overall_weighted", ascending=True).reset_index(drop=True)
    overall_df.to_csv(os.path.join(args.out_dir, "owa_overall_weighted.csv"), index=False)

    print("\nSaved:")
    print(" -", os.path.join(args.out_dir, "naive2_metrics_per_group_uni2ts.csv"))
    print(" -", os.path.join(args.out_dir, "owa_per_group_and_model.csv"))
    print(" -", os.path.join(args.out_dir, "owa_overall_weighted.csv"))
    print("\nOverall OWA (lower=better):")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()

