from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import torch
import yaml
from scipy import stats

from src.models.variants import (
    VariantA_SingleBranchUnified,
    VariantB_DualBranchConcat,
    VariantC_AIRPRED,
    VariantD_PM25Only,
)

DATA_DIR = "data/final"
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"

VARIANTS = {
    "A": VariantA_SingleBranchUnified,
    "B": VariantB_DualBranchConcat,
    "C": VariantC_AIRPRED,
    "D": VariantD_PM25Only,
}


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


def load_pm25_inverse_transform(data_dir: str):
    """The scaler is a single StandardScaler fit across all 8 columns
    (PM2.5 + 7 met vars) together, keyed by feature_names_in_ -- NOT one
    scaler per feature. Y (the forecast target) is assumed to have been
    normalized with this same PM2.5 column's mean/scale, since it's the
    same physical quantity as the PM2.5 input stream, just future
    timesteps. This is a reasonable inference, not a documented thesis
    fact -- the printed sanity-check range after inverse-transforming is
    there so you can visually confirm it lands in a plausible PM2.5 range
    rather than trusting this silently."""
    scaler = joblib.load(f"{data_dir}/scaler.pkl")
    pm25_idx = list(scaler.feature_names_in_).index("pm25")
    mean, scale = float(scaler.mean_[pm25_idx]), float(scaler.scale_[pm25_idx])
    print(f"PM2.5 scaler: mean={mean:.4f}  scale={scale:.4f}  (column index {pm25_idx})")

    def inverse(arr: np.ndarray) -> np.ndarray:
        return arr * scale + mean

    return inverse


def build_model(name: str, ModelClass, horizon: int, cfg: dict):
    if name != "C":
        return ModelClass(horizon=horizon)
    return ModelClass(
        horizon=horizon,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_attention_heads"],
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray, zero_exclusion: bool = True) -> float:
    if zero_exclusion:
        mask = y_true != 0
        y_true, y_pred = y_true[mask], y_pred[mask]
    return 100.0 * float(np.mean(np.abs((y_true - y_pred) / y_true)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def per_window_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """One RMSE per test window (row), averaged over the 24-hour horizon.
    This is the 'per_window_rmse' unit config.yaml specifies for pairing."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=1))


def run_inference(model, X_pm25, X_met, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_pm25), batch_size):
            xp = torch.as_tensor(X_pm25[i:i + batch_size]).to(device)
            xm = torch.as_tensor(X_met[i:i + batch_size]).to(device)
            y_hat = model(xp, xm)
            preds.append(y_hat.cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Which seed's checkpoints to load, e.g. checkpoints/variant_a_seed{SEED}_best.pt",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizon = cfg["data"]["forecast_horizon"]
    print(f"Using device: {device}")
    print(f"Evaluating seed-{args.seed} checkpoints on the held-out test set.\n")

    X_pm25, X_met, Y, cities = load_split("test")
    print(f"test: X_pm25={X_pm25.shape}  X_met={X_met.shape}  Y={Y.shape}\n")

    inverse_pm25 = load_pm25_inverse_transform(DATA_DIR)
    Y = inverse_pm25(Y)
    print(
        f"Real-unit PM2.5 range (test set actuals): "
        f"min={Y.min():.2f}  max={Y.max():.2f}  mean={Y.mean():.2f}  µg/m³\n"
        f"(sanity check -- should look like plausible Philippine PM2.5 levels, "
        f"roughly single-to-low-double digits on average, not near-zero or in "
        f"the hundreds on average)\n"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    per_window: dict[str, np.ndarray] = {}

    for name, ModelClass in VARIANTS.items():
        model = build_model(name, ModelClass, horizon, cfg)
        ckpt_path = f"{CKPT_DIR}/variant_{name.lower()}_seed{args.seed}_best.pt"
        assert os.path.exists(ckpt_path), (
            f"Checkpoint not found: {ckpt_path}. Run train_local.py with "
            f"--variants {name} --seeds {args.seed} --keep-checkpoints first."
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        y_hat = run_inference(model, X_pm25, X_met, device)
        y_hat = inverse_pm25(y_hat)
        predictions[name] = y_hat

        rmse = float(np.sqrt(((Y - y_hat) ** 2).mean()))
        mae = float(np.abs(Y - y_hat).mean())
        m_mape = mape(Y, y_hat, zero_exclusion=cfg["evaluation"]["mape_zero_exclusion"])
        m_r2 = r_squared(Y, y_hat)
        metrics[name] = {"rmse": rmse, "mae": mae, "mape": m_mape, "r2": m_r2}
        per_window[name] = per_window_rmse(Y, y_hat)

        print(f"Variant {name}: RMSE={rmse:.4f} µg/m³  MAE={mae:.4f} µg/m³  "
              f"MAPE={m_mape:.2f}%  R2={m_r2:.4f}")

    # Paired t-tests, exactly as specified in config.yaml's statistical_tests block.
    print(f"\n{'='*60}\nPaired t-tests (alpha={cfg['evaluation']['alpha']})\n{'='*60}")
    alpha = cfg["evaluation"]["alpha"]
    tests = cfg["statistical_tests"]
    rq_results = {}
    for rq, spec in tests.items():
        if rq in ("test", "paired_unit"):
            continue
        proposed, baseline = spec["proposed"], spec["baseline"]
        t_stat, p_val = stats.ttest_rel(per_window[proposed], per_window[baseline])
        significant = bool(p_val < alpha)
        rq_results[rq] = {
            "proposed": proposed,
            "baseline": baseline,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_at_alpha": significant,
            "mean_rmse_proposed": float(per_window[proposed].mean()),
            "mean_rmse_baseline": float(per_window[baseline].mean()),
        }
        print(
            f"{rq}: {proposed} vs {baseline}  -- "
            f"t={t_stat:.4f}  p={p_val:.4f}  significant={significant}"
        )

    out_path = f"{RESULTS_DIR}/stage3_metrics_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"seed": args.seed, "units": "µg/m³ (inverse-transformed via scaler.pkl)",
             "metrics": metrics, "paired_t_tests": rq_results},
            f, indent=2,
        )
    print(f"\nMetrics + t-tests written to {out_path}")

    # Predictions themselves (not just summary metrics) are also saved -- these
    # are what the airpred-web dashboard's Model Comparison page will need
    # once we wire up the JSON export for it.
    np.savez(
        f"{RESULTS_DIR}/stage3_predictions_seed{args.seed}.npz",
        cities=cities,
        actual=Y,
        **{f"pred_{name}": pred for name, pred in predictions.items()},
    )
    print(f"Raw predictions written to {RESULTS_DIR}/stage3_predictions_seed{args.seed}.npz")


if __name__ == "__main__":
    main()