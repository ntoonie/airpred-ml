"""Paired t-tests for RQ1 (B vs A), RQ2 (C vs B), RQ3 (C vs D).

"Per-sample RMSE" is not literally defined in the thesis (RMSE is normally
an aggregate over many samples). This module implements the recommended
reading from the implementation guide (Section 27): one RMSE (and MAE,
MAPE, R2) per sliding-window TEST SAMPLE, computed over that window's own
24 horizon steps. This must be the SAME set of test windows (same city,
same timestamps) for both models being compared, or the pairing is invalid.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def per_window_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> np.ndarray:
    """y_true, y_pred: (n_windows, horizon). Returns (n_windows,)."""
    err = y_pred - y_true
    if metric == "rmse":
        return np.sqrt(np.mean(err ** 2, axis=1))
    if metric == "mae":
        return np.mean(np.abs(err), axis=1)
    if metric == "mape":
        mask = np.abs(y_true) > 1e-3
        out = np.full(y_true.shape[0], np.nan)
        for i in range(y_true.shape[0]):
            m = mask[i]
            out[i] = 100 * np.mean(np.abs(err[i, m] / y_true[i, m])) if m.any() else np.nan
        return out
    if metric == "r2":
        ss_res = np.sum(err ** 2, axis=1)
        ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1 - ss_res / ss_tot
        return r2
    raise ValueError(f"unknown metric: {metric}")


def paired_ttest(values_proposed: np.ndarray, values_baseline: np.ndarray, alpha: float = 0.05) -> dict:
    valid = ~(np.isnan(values_proposed) | np.isnan(values_baseline))
    t_stat, p_value = stats.ttest_rel(values_proposed[valid], values_baseline[valid])
    return {
        "n": int(valid.sum()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_diff": float(np.mean(values_proposed[valid] - values_baseline[valid])),
    }


def run_all_comparisons(results: dict, alpha: float = 0.05) -> dict:
    """`results[variant] = (y_true, y_pred)` for variants 'A','B','C','D',
    where y_true/y_pred are aligned (n_windows, horizon) arrays covering the
    SAME test windows for every variant. Returns Table-13-shaped output."""
    comparisons = {
        "RQ1_B_vs_A": ("B", "A"),
        "RQ2_C_vs_B": ("C", "B"),
        "RQ3_C_vs_D": ("C", "D"),
    }
    out = {}
    for rq, (proposed, baseline) in comparisons.items():
        y_true_p, y_pred_p = results[proposed]
        y_true_b, y_pred_b = results[baseline]
        out[rq] = {}
        for metric in ("rmse", "mae", "mape", "r2"):
            vals_p = per_window_metric(y_true_p, y_pred_p, metric)
            vals_b = per_window_metric(y_true_b, y_pred_b, metric)
            out[rq][metric] = paired_ttest(vals_p, vals_b, alpha)
    return out
