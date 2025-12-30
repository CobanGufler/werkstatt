# run_all_m4.py
import argparse
import os
import pandas as pd

from scripts.run_timesfm_m4 import evaluate_timesfm_m4
from scripts.run_chronos_m4 import evaluate_chronos_m4
from scripts.run_moirai_m4 import evaluate_moirai_m4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="test")

    parser.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--chronos_ckpt", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--moirai_repo", type=str, default="Salesforce/moirai-1.0-R-small")

    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chronos_samples", type=int, default=20)
    parser.add_argument("--moirai_samples", type=int, default=100)
    parser.add_argument("--moirai_batch", type=int, default=32)

    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    df1 = evaluate_timesfm_m4(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.timesfm_ckpt,
        save_path=os.path.join(out_dir, f"timesfm_m4_{args.group}.csv"),
        context_len=args.context_len,
        batch_size=args.batch_size,
    )

    df2 = evaluate_chronos_m4(
        group=args.group,
        data_dir=args.data_dir,
        checkpoint=args.chronos_ckpt,
        save_path=os.path.join(out_dir, f"chronos_m4_{args.group}.csv"),
        context_len=args.context_len,
        num_samples=args.chronos_samples,
        batch_size=args.batch_size,
    )

    df3 = evaluate_moirai_m4(
        group=args.group,
        data_dir=args.data_dir,
        repo_id=args.moirai_repo,
        save_path=os.path.join(out_dir, f"moirai_m4_{args.group}.csv"),
        context_len=args.context_len,
        num_samples=args.moirai_samples,
        batch_size=args.moirai_batch,
    )

    merged = pd.concat([df1, df2, df3], axis=0)
    merged.index = [f"timesfm_m4_{args.group}", f"chronos_m4_{args.group}", f"moirai_m4_{args.group}"]
    merged.to_csv(os.path.join(out_dir, f"ALL_m4_{args.group}.csv"))
    print(merged)
