import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from load_m4_data import load_m4_daily
from src.models import ModelFactory


def run_timesfm(
    id: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tfm=None,
    freq: str = "D",
    value_col: str = "y",
    plot: bool = True,
):

    # train data
    train_ts = train_data[train_data["unique_id"] == id].sort_values("ds").copy()
    if train_ts.empty:
        raise ValueError(f"Keine Train-Daten für {id}")

    ds_num = pd.to_numeric(train_ts["ds"])
    train_ts["ds"] = pd.to_datetime(ds_num, unit="D", origin="1970-01-01")

    # test data
    test_row = test_data[test_data["V1"] == id]
    if test_row.empty:
        raise ValueError(f"Keine Test-Daten für {id}")

    horizon_cols = [c for c in test_data.columns if c != "V1"]

    test_y = test_row[horizon_cols].to_numpy().flatten()

    # laod model
    if tfm is None:
        factory = ModelFactory()
        tfm = factory.load_timesfm()

    h = len(test_y)

    # forecast
    fcst_df = tfm.forecast_on_df(
        inputs=train_ts,
        freq=freq,
        value_name=value_col,
        num_jobs=1,
    )

    fcst_df = fcst_df.groupby("unique_id", as_index=False).head(h)

    forecast_col = [c for c in fcst_df.columns if c not in ("unique_id", "ds")][0]
    preds = fcst_df[forecast_col].to_numpy()

    if plot:
        train_y = train_ts[value_col].to_numpy()

        plt.figure(figsize=(12, 5))
        plt.plot(range(len(train_y)), train_y, label="Train")
        plt.plot(range(len(train_y), len(train_y)+h), test_y, label="Test")
        plt.plot(range(len(train_y), len(train_y)+h), preds, label="Forecast")
        plt.title(f"TimesFM – M4 Forecast für {id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "train": train_ts[value_col].to_numpy(),
        "test": test_y,
        "forecast": preds,
    }


def main():
    daily_train, daily_test = load_m4_daily()

    factory = ModelFactory()
    tfm = factory.load_timesfm()

    result = run_timesfm(
        id="D1",
        train_data=daily_train,
        test_data=daily_test,
        tfm=tfm,
        freq="D",
        plot=True,
    )

    test = result["test"]
    pred = result["forecast"]

    smape = 100 * np.mean(2 * np.abs(test - pred) / (np.abs(test) + np.abs(pred)))
    print("SMAPE für D1:", smape)

if __name__ == "__main__":
    main()