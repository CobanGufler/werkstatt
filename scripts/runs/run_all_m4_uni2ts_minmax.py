# This file integrates multiple pretrained foundation models into a unified M4 evaluation pipeline.
#
# TimesFM:
#   Adapted from the official TimesFM implementation (google-research/timesfm).
#   License: see model card for google/timesfm-1.0-200m-pytorch
#
# Chronos:
#   Adapted from the official Chronos implementation (amazon-science/chronos-forecasting).
#   License: see model cards for amazon/chronos-t5-tiny and amazon/chronos-t5-base
#
# Moirai:
#   Adapted from the official Moirai implementation (SalesforceAIResearch/uni2ts).
#   License: see model cards for Salesforce/moirai-1.0-R-small and Salesforce/moirai-1.0-R-base
#
# Modifications: orchestrates the minmax model-specific scripts in a unified run.

from __future__ import annotations

import argparse
import os

import pandas as pd

from scripts.runs.run_timesfm_m4 import evaluate_timesfm_m4_uni2ts_minmax
from scripts.runs.run_chronos_m4 import evaluate_chronos_m4_uni2ts_minmax
from scripts.runs.run_moirai_m4 import evaluate_moirai_m4_uni2ts_minmax


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
