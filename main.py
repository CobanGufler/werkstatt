import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from load_m4_data import load_m4_daily
from src.models import ModelFactory   # bei dir evtl. nur `from models import ModelFactory`


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 100 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    )


def run_timesfm(
    id: str,
    train_data: pd.DataFrame,   # = FULL (Train + Test) aus M4.load
    test_data: pd.DataFrame,    # = Daily-test.csv
    tfm=None,
    freq: str = "D",
    value_col: str = "y",
    plot: bool = True,
):
    # -------- FULL-Serie für diese ID laden --------
    full = train_data[train_data["unique_id"] == id].copy()
    if full.empty:
        raise ValueError(f"Keine Daten für {id}")

    # ds von int -> datetime
    ds_num = pd.to_numeric(full["ds"])
    full["ds"] = pd.to_datetime(ds_num, unit="D", origin="1970-01-01")
    full = full.sort_values("ds").reset_index(drop=True)

    # -------- Horizont aus Daily-test.csv bestimmen --------
    if test_data is not None:
        test_row = test_data[test_data["V1"] == id]
        if test_row.empty:
            raise ValueError(f"Keine Test-Daten für {id} in Daily-test.csv")
        horizon_cols = [c for c in test_data.columns if c != "V1"]
        h = len(horizon_cols)
    else:
        # Fallback für Daily
        h = 14

    # -------- Full in echtes Train + Test splitten --------
    train_ts = full.iloc[:-h].copy()   # Kontext
    test_ts = full.iloc[-h:].copy()    # offizieller M4-Horizont

    # -------- TimesFM-Modell laden --------
    if tfm is None:
        factory = ModelFactory()
        tfm = factory.load_timesfm()

    # Nur die relevanten Spalten an TimesFM übergeben
    tfm_input = train_ts[["unique_id", "ds", value_col]].copy()

    # -------- Forecast mit TimesFM --------
    fcst_df = tfm.forecast_on_df(
        inputs=tfm_input,
        freq=freq,
        value_name=value_col,
        num_jobs=1,
    )

    # Nur unsere ID behalten
    fcst_id = fcst_df[fcst_df["unique_id"] == id].copy()

    # Sicherstellen, dass wir nur echte Zukunftspunkte nach dem letzten Train-ds nehmen
    last_train_ds = train_ts["ds"].max()
    fcst_id = fcst_id[fcst_id["ds"] > last_train_ds].copy()
    fcst_id = fcst_id.sort_values("ds").reset_index(drop=True)

    # Spalte mit den Vorhersagen finden (z.B. "TimesFM" oder "0")
    forecast_col = [c for c in fcst_id.columns if c not in ("unique_id", "ds")][0]

    # Auf den Testhorizont kürzen (falls TimesFM mehr ausspuckt)
    fcst_id = fcst_id.head(h)

    # -------- Test und Forecast nach Datum ausrichten --------
    test_merge = test_ts[["ds", value_col]].copy()
    merged = pd.merge(
        test_merge,
        fcst_id[["ds", forecast_col]],
        on="ds",
        how="inner",
    )

    y_train = train_ts[value_col].to_numpy()
    y_test = merged[value_col].to_numpy()
    y_pred = merged[forecast_col].to_numpy()

    # -------- Plot --------
    if plot:
        x_train = np.arange(len(y_train))
        x_future = np.arange(len(y_train), len(y_train) + len(y_test))

        plt.figure(figsize=(12, 5))
        plt.plot(x_train, y_train, label="Train")
        plt.plot(x_future, y_test, label="Test")
        plt.plot(x_future, y_pred, label="Forecast")
        plt.title(f"TimesFM – M4 Forecast für {id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "train": y_train,
        "test": y_test,
        "forecast": y_pred,
    }


def main():
    # komplette Daily-Serie + offizielle Testdatei laden
    daily_full, daily_test = load_m4_daily()

    factory = ModelFactory()
    tfm = factory.load_timesfm()

    # Beispiel: D1
    result = run_timesfm(
        id="D4",
        train_data=daily_full,
        test_data=daily_test,
        tfm=tfm,
        freq="D",   # oder "d"
        plot=True,
    )

    test = result["test"]
    pred = result["forecast"]

    print("SMAPE für D1:", smape(test, pred))


if __name__ == "__main__":
    main()