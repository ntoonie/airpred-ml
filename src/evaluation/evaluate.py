"""Inference + metric computation on the held-out test set."""
from __future__ import annotations

import numpy as np
import torch


def inverse_scale_pm25(arr: np.ndarray, scaler, pm25_col_idx: int = 0) -> np.ndarray:
    """scaler: fitted sklearn StandardScaler over FEATURE_COLS, where pm25
    is column `pm25_col_idx` (0 by default -- matches FEATURE_COLS order)."""
    mean = scaler.mean_[pm25_col_idx]
    std = scaler.scale_[pm25_col_idx]
    return arr * std + mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, exclude_near_zero: bool = True,
                     eps: float = 1e-3) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mask = np.abs(y_true) > eps if exclude_near_zero else np.ones_like(y_true, dtype=bool)
    mape = float(100 * np.mean(np.abs(err[mask] / y_true[mask]))) if mask.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


@torch.no_grad()
def evaluate_model(model, X_pm25, X_met, Y, cities, scaler, device, batch_size: int = 256):
    """Returns (preds_ug_m3, trues_ug_m3, cities) -- all in original PM2.5 units."""
    model.eval()
    preds = []
    n = len(Y)
    for start in range(0, n, batch_size):
        xb_pm25 = torch.as_tensor(X_pm25[start:start + batch_size]).to(device)
        xb_met = torch.as_tensor(X_met[start:start + batch_size]).to(device)
        y_hat = model(xb_pm25, xb_met).cpu().numpy()
        preds.append(y_hat)
    preds = np.concatenate(preds)
    preds_ug = inverse_scale_pm25(preds, scaler)
    trues_ug = inverse_scale_pm25(Y, scaler)
    return preds_ug, trues_ug, cities


def metrics_per_city(preds_ug: np.ndarray, trues_ug: np.ndarray, cities: np.ndarray) -> dict:
    """One metrics dict per city, computed over ALL that city's test windows
    and ALL 24 horizon steps pooled together."""
    out = {}
    for city in np.unique(cities):
        mask = cities == city
        out[city] = compute_metrics(trues_ug[mask].ravel(), preds_ug[mask].ravel())
    return out


def metrics_per_horizon_step(preds_ug: np.ndarray, trues_ug: np.ndarray, steps=(1, 6, 12, 24)) -> dict:
    """One metrics dict per requested horizon step (1-indexed, per Appendix
    tables' T+1/T+6/T+12/T+24 spot checks), pooled across all cities/windows."""
    out = {}
    for step in steps:
        idx = step - 1
        out[f"T+{step}"] = compute_metrics(trues_ug[:, idx], preds_ug[:, idx])
    return out
