import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gluonts.ev.metrics import MSE, MAE, SMAPE, MASE, ND, NRMSE
from gluonts.model.forecast import SampleForecast
from uni2ts.eval_util.evaluation import evaluate_forecasts

from scripts.data_load.m4_gluonts_loader import get_m4_test_dataset


CANONICAL_GROUPS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
}

DISPLAY_NAMES = {
    "Hourly": "Hourly",
    "Daily": "Daily",
    "Weekly": "Weekly",
    "Monthly": "Monthly",
    "Quarterly": "Quarterly",
    "Yearly": "Yearly",
}

SEASONALITY = {
    "hourly": 24,
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

@dataclass
class SimpleTestData:
    input: List[dict]
    label: List[dict]


def parse_groups(groups_arg: str) -> List[str]:
    parts = [p.strip() for p in groups_arg.split(",") if p.strip()]
    out = []
    for p in parts:
        key = p.lower()
        out.append(CANONICAL_GROUPS.get(key, p))
    return out


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


def load_timesfm(checkpoint: str, device: str = "cpu"):
    import timesfm

    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend=device),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )


def forecast_timesfm(
    model,
    y_hist: np.ndarray,
    start_ts,
    freq: str,
    h: int,
    num_jobs: int,
) -> np.ndarray:
    import pandas as pd

    uid = "series_0"
    ds = pd.date_range(start=start_ts, periods=len(y_hist), freq=freq)
    rows = [(uid, t, float(v)) for t, v in zip(ds, y_hist)]
    inp_df = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

    pred_df = model.forecast_on_df(
        inputs=inp_df,
        freq=freq,
        value_name="y",
        num_jobs=num_jobs,
    )
    pred_df = pred_df.sort_values(["unique_id", "ds"])
    val_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
    col = "timesfm" if "timesfm" in val_cols else val_cols[0]
    return pred_df[pred_df["unique_id"] == uid][col].to_numpy(dtype=float)[:h]


def load_chronos(checkpoint: str):
    import torch
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(
        checkpoint,
        device_map="cpu",
        torch_dtype=torch.float32,
    )


def forecast_chronos(
    pipeline,
    y_hist: np.ndarray,
    h: int,
    num_samples: int,
) -> np.ndarray:
    import torch

    context = torch.tensor(y_hist, dtype=torch.float32)
    samples = pipeline.predict(
        [context],
        prediction_length=h,
        num_samples=num_samples,
        limit_prediction_length=False,
    ).numpy()  # (1, num_samples, h)
    return np.median(samples[0], axis=0), samples[0]


def forecast_moirai(
    repo_id: str,
    y_hist: np.ndarray,
    start,
    freq: str,
    h: int,
    context_len: int,
    num_samples: int,
    batch_size: int,
):
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

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

    ds = ListDataset(
        [{"item_id": "series_0", "start": start, "target": y_hist}],
        freq=freq,
    )
    fcst = next(iter(predictor.predict(ds)))
    return fcst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="paper_plots")
    parser.add_argument(
        "--groups",
        type=str,
        default="Hourly,Daily,Weekly,Monthly,Quarterly,Yearly",
    )
    parser.add_argument("--model", type=str, default="timesfm",
                        choices=["timesfm", "chronos", "moirai"])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--history_max", type=int, default=400)
    parser.add_argument("--save_pdf", action="store_true")

    parser.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--chronos_ckpt", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--moirai_repo", type=str, default="Salesforce/moirai-1.0-R-small")

    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chronos_samples", type=int, default=10)
    parser.add_argument("--moirai_samples", type=int, default=10)
    parser.add_argument("--moirai_batch", type=int, default=256)
    parser.add_argument("--num_jobs", type=int, default=-1)

    args = parser.parse_args()

    groups = parse_groups(args.groups)
    if not groups:
        raise ValueError("--groups must contain at least one group")

    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
    })

    n = len(groups)
    ncols = 1
    nrows = int(math.ceil(n / ncols))
    fig_w = 11.5
    fig_h = 3.8 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4.8, 0.9]},
    )
    axes = np.atleast_2d(axes)

    timesfm_model = None
    chronos_pipeline = None

    if args.model == "timesfm":
        timesfm_model = load_timesfm(args.timesfm_ckpt, device="cpu")
    elif args.model == "chronos":
        chronos_pipeline = load_chronos(args.chronos_ckpt)

    for ax_i, group in enumerate(groups):
        ax = axes[ax_i, 0]
        ax_metrics = axes[ax_i, 1]
        ax_metrics.axis("off")

        test_data, metadata = get_m4_test_dataset(group=group, data_dir=args.data_dir)
        inputs = list(test_data.input)
        labels = list(test_data.label)

        idx = int(rng.integers(0, len(inputs)))
        in_entry = inputs[idx]
        lb_entry = labels[idx]

        y_hist = np.asarray(in_entry["target"], dtype=float)
        y_true = np.asarray(lb_entry["target"], dtype=float)

        y_hist_model = maybe_truncate(y_hist, args.context_len)
        if args.history_max and args.history_max > 0 and len(y_hist_model) > args.history_max:
            y_hist_plot = y_hist_model[-args.history_max:]
        else:
            y_hist_plot = y_hist_model

        start_ts = in_entry["start"].to_timestamp()
        h = metadata.prediction_length

        if args.model == "timesfm":
            y_hat = forecast_timesfm(
                timesfm_model,
                y_hist_model,
                start_ts,
                metadata.freq,
                h,
                args.num_jobs,
            )
            forecasts = [
                SampleForecast(
                    samples=y_hat[None, :],
                    start_date=_pred_start(in_entry),
                    item_id=in_entry.get("item_id", None),
                )
            ]
        elif args.model == "chronos":
            y_hat, sample_arr = forecast_chronos(
                chronos_pipeline,
                y_hist_model,
                h,
                args.chronos_samples,
            )
            forecasts = [
                SampleForecast(
                    samples=sample_arr,
                    start_date=_pred_start(in_entry),
                    item_id=in_entry.get("item_id", None),
                )
            ]
        else:
            fcst = forecast_moirai(
                args.moirai_repo,
                y_hist_model,
                in_entry["start"],
                metadata.freq,
                h,
                args.context_len,
                args.moirai_samples,
                args.moirai_batch,
            )
            y_hat = np.median(fcst.samples, axis=0)
            forecasts = [fcst]

        split = len(y_hist_plot)
        x_hist = np.arange(split)
        x_test = np.arange(split, split + len(y_true))

        x_test_conn = np.concatenate(([x_hist[-1]], x_test))
        y_test_conn = np.concatenate(([y_hist_plot[-1]], y_true))

        x_fcst_conn = np.concatenate(([x_hist[-1]], x_test))
        y_fcst_conn = np.concatenate(([y_hist_plot[-1]], y_hat))

        ax.plot(x_hist, y_hist_plot, label="History")
        ax.plot(x_test_conn, y_test_conn, label="Test (ground truth)")
        ax.plot(x_fcst_conn, y_fcst_conn, label="Forecast")

        ax.axvline(split - 1, linestyle="--", linewidth=2)

        single_test = SimpleTestData(input=[in_entry], label=[lb_entry])
        df = evaluate_forecasts(
            forecasts=forecasts,
            test_data=single_test,
            metrics=_metrics(),
            batch_size=1,
            seasonality=SEASONALITY.get(group.lower(), None),
        )
        row = df.iloc[0]

        metrics_text = (
            f"MSE: {row.get('MSE[mean]', np.nan):.3g}\n"
            f"MAE: {row.get('MAE[0.5]', np.nan):.3g}\n"
            f"ND: {row.get('ND[0.5]', np.nan):.3g}\n"
            f"sMAPE: {row.get('sMAPE[0.5]', np.nan):.3g}\n"
            f"MASE: {row.get('MASE[0.5]', np.nan):.3g}\n"
            f"NRMSE: {row.get('NRMSE[mean]', np.nan):.3g}"
        )
        ax_metrics.text(
            0.0,
            1.0,
            metrics_text,
            transform=ax_metrics.transAxes,
            va="top",
            ha="left",
        )

        ax.set_title(f"{DISPLAY_NAMES.get(group, group)} example")
        ax.set_ylabel("Value")
        if ax_i == n - 1:
            ax.set_xlabel("Time index")

        if ax_i == 0:
            ax.legend(loc="upper left", frameon=True)

        ax.tick_params(axis="both", which="major", length=6, width=1.5)

        print(
            f"{group}: idx={idx} "
            f"MAE={row.get('MAE[0.5]', np.nan):.4g} "
            f"sMAPE={row.get('sMAPE[0.5]', np.nan):.4g} "
            f"MASE={row.get('MASE[0.5]', np.nan):.4g} "
            f"NRMSE={row.get('NRMSE[mean]', np.nan):.4g}"
        )

    for j in range(n, axes.shape[0]):
        fig.delaxes(axes[j, 0])
        fig.delaxes(axes[j, 1])

    safe_groups = "_".join([g.lower() for g in groups])
    out_png = os.path.join(out_dir, f"m4_forecast_examples_{args.model}_{safe_groups}_seed{args.seed}.png")
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

    if args.save_pdf:
        out_pdf = os.path.join(out_dir, f"m4_forecast_examples_{args.model}_{safe_groups}_seed{args.seed}.pdf")
        fig.savefig(out_pdf)
        print(f"Saved: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
