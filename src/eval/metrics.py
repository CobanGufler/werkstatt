# src/eval/metrics.py
from __future__ import annotations
import numpy as np

EPS = 1e-7

def _mse(y_pred, y_true):
    return np.square(y_pred - y_true)

def _mae(y_pred, y_true):
    return np.abs(y_pred - y_true)

def _smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
    abs_val = np.where(abs_val > EPS, abs_val, 1.0)
    abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
    return abs_diff / abs_val


class RunningMetrics:
    def __init__(self):
        self.mae_sum = 0.0
        self.mse_sum = 0.0
        self.smape_sum = 0.0
        self.num_elements = 0
        self.abs_sum = 0.0

        # NEW: MASE (mean over series)
        self.mase_sum = 0.0
        self.mase_count = 0

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert y_pred.shape == y_true.shape, f"{y_pred.shape} != {y_true.shape}"
        self.mae_sum += float(_mae(y_pred, y_true).sum())
        self.mse_sum += float(_mse(y_pred, y_true).sum())
        self.smape_sum += float(_smape(y_pred, y_true).sum())
        self.num_elements += int(y_true.size)
        self.abs_sum += float(np.abs(y_true).sum())

    # NEW
    def update_mase(
        self,
        y_pred: np.ndarray,     # (bs, h)
        y_true: np.ndarray,     # (bs, h)
        insample: np.ndarray,   # (bs, context_len)
        m: int = 1,             # seasonality
    ):
        assert y_pred.shape == y_true.shape
        assert insample.ndim == 2 and y_pred.ndim == 2

        m = int(max(m, 1))
        # denom_i = mean(|y_t - y_{t-m}|) over insample
        if insample.shape[1] <= m:
            return  # nicht genug Kontext, skip

        denom = np.mean(np.abs(insample[:, m:] - insample[:, :-m]), axis=1)  # (bs,)
        mae_per_series = np.mean(np.abs(y_pred - y_true), axis=1)            # (bs,)

        valid = denom > EPS
        if not np.any(valid):
            return

        mase_vals = mae_per_series[valid] / denom[valid]
        self.mase_sum += float(mase_vals.sum())
        self.mase_count += int(valid.sum())

    def finalize(self):
        mse_val = self.mse_sum / max(self.num_elements, 1)
        mae_val = self.mae_sum / max(self.num_elements, 1)
        smape_val = self.smape_sum / max(self.num_elements, 1)
        wape_val = self.mae_sum / max(self.abs_sum, EPS)
        nrmse_val = (np.sqrt(mse_val)) / max((self.abs_sum / max(self.num_elements, 1)), EPS)

        mase_val = self.mase_sum / max(self.mase_count, 1)

        return {
            "mse": mse_val,
            "mae": mae_val,
            "smape": smape_val,
            "wape": wape_val,
            "nrmse": nrmse_val,
            "mase": mase_val,  # NEW
            "num_elements": self.num_elements,
            "abs_sum": self.abs_sum,
            "mase_count": self.mase_count,  # optional/debug
        }

