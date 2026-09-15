from __future__ import annotations

import argparse
import csv

import numpy as np

RESULTS_DIR = "results"
HORIZONS = [1, 6, 12, 24]   # T+1, T+6, T+12, T+24 per thesis Tables 9-11
VARIANTS = ["A", "B", "C", "D"]


def mape(y_true: np.ndarray, y_pred: np.ndarray, zero_exclusion: bool = True) -> float:
    if zero_exclusion:
        mask = y_true != 0
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    return 100.0 * float(np.mean(np.abs((y_true - y_pred) / y_true)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz_path = f"{RESULTS_DIR}/stage3_predictions_seed{args.seed}.npz"
    d = np.load(npz_path, allow_pickle=True)
    cities = d["cities"]
    actual = d["actual"]  # (N, 24)
    preds = {v: d[f"pred_{v}"] for v in VARIANTS}  # each (N, 24)

    unique_cities = sorted(set(cities.tolist()))
    print(f"Stations found ({len(unique_cities)}): {unique_cities}\n")

    # ---- Tables 9-11 style: per-station, per-horizon breakdown ----
    per_horizon_rows = []
    for variant in VARIANTS:
        y_hat_all = preds[variant]
        for city in unique_cities:
            mask = cities == city
            y_true_city = actual[mask]      # (n_city, 24)
            y_pred_city = y_hat_all[mask]    # (n_city, 24)
            for h in HORIZONS:
                col = h - 1  # T+1 -> index 0, T+24 -> index 23
                yt = y_true_city[:, col]
                yp = y_pred_city[:, col]
                rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
                mae = float(np.mean(np.abs(yt - yp)))
                m_mape = mape(yt, yp)
                m_r2 = r_squared(yt, yp)
                per_horizon_rows.append({
                    "variant": variant, "station": city, "forecast_horizon": f"T+{h}",
                    "rmse": rmse, "mae": mae, "mape": m_mape, "r2": m_r2,
                    "mean_predicted": float(yp.mean()), "mean_actual": float(yt.mean()),
                    "n_windows": int(mask.sum()),
                })

    per_horizon_path = f"{RESULTS_DIR}/tables_9_11_per_station_per_horizon_seed{args.seed}.csv"
    with open(per_horizon_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_horizon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_horizon_rows)
    print(f"Tables 9-11 (per-station, per-horizon) written to {per_horizon_path}")

    # ---- Table 12 style: full 24h RMSE/MAE/MAPE/R2 per station, then
    # mean +/- std across the 10 stations, per variant ----
    aggregate_rows = []
    for variant in VARIANTS:
        y_hat_all = preds[variant]
        per_station_rmse, per_station_mae, per_station_mape, per_station_r2 = [], [], [], []
        for city in unique_cities:
            mask = cities == city
            yt = actual[mask]       # (n_city, 24) -- full horizon
            yp = y_hat_all[mask]
            per_station_rmse.append(np.sqrt(np.mean((yt - yp) ** 2)))
            per_station_mae.append(np.mean(np.abs(yt - yp)))
            per_station_mape.append(mape(yt.ravel(), yp.ravel()))
            per_station_r2.append(r_squared(yt.ravel(), yp.ravel()))
        aggregate_rows.append({
            "variant": variant,
            "rmse_mean": float(np.mean(per_station_rmse)), "rmse_std": float(np.std(per_station_rmse)),
            "mae_mean": float(np.mean(per_station_mae)), "mae_std": float(np.std(per_station_mae)),
            "mape_mean": float(np.nanmean(per_station_mape)), "mape_std": float(np.nanstd(per_station_mape)),
            "r2_mean": float(np.mean(per_station_r2)), "r2_std": float(np.std(per_station_r2)),
        })

    aggregate_path = f"{RESULTS_DIR}/table_12_aggregate_seed{args.seed}.csv"
    with open(aggregate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"Table 12 (aggregate mean+/-std across stations) written to {aggregate_path}\n")

    for row in aggregate_rows:
        print(f"{row['variant']}: RMSE={row['rmse_mean']:.4f}+/-{row['rmse_std']:.4f}  "
              f"R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}")


if __name__ == "__main__":
    main()
