from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.models.variants import VariantC_AIRPRED

DATA_DIR = "data/final"
CKPT_DIR = "checkpoints"
OUT_DIR = "results/attention_plots"


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of high-pollution test windows to visualize (highest peak forecast PM2.5)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizon = cfg["data"]["forecast_horizon"]

    model = VariantC_AIRPRED(
        horizon=horizon,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_attention_heads"],
    )
    ckpt_path = f"{CKPT_DIR}/variant_c_seed{args.seed}_best.pt"
    assert os.path.exists(ckpt_path), (
        f"Checkpoint not found: {ckpt_path}. This must be the FIXED AIRPRED "
        f"checkpoint (post attention.py correction), not the pre-fix one."
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    X_pm25, X_met, Y, cities = load_split("test")

    # "High-pollution episodes" -- windows whose actual forecast horizon
    # contains the highest peak PM2.5 values in the test set (top N, by peak).
    peak_per_window = Y.max(axis=1)  # (N,) -- worst forecasted hour in each window
    top_idx = np.argsort(peak_per_window)[-args.n_samples:][::-1]

    os.makedirs(OUT_DIR, exist_ok=True)

    for rank, idx in enumerate(top_idx, start=1):
        xp = torch.as_tensor(X_pm25[idx:idx + 1]).to(device)
        xm = torch.as_tensor(X_met[idx:idx + 1]).to(device)
        with torch.no_grad():
            _, attn_weights = model(xp, xm, return_attention=True)
        # attn_weights: (1, num_heads, L, L) -- dim0=query pos (PM2.5 hour),
        # dim1=key pos (meteorological hour)
        attn = attn_weights[0].cpu().numpy()  # (num_heads, L, L)
        n_heads = attn.shape[0]

        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
        if n_heads == 1:
            axes = [axes]
        city_label = cities[idx]
        peak_val = peak_per_window[idx]
        fig.suptitle(
            f"{city_label} -- test window #{idx} "
            f"(peak forecast PM2.5={peak_val:.2f}, scaled units) -- "
            f"cross-modal attention per head"
        )
        for h in range(n_heads):
            im = axes[h].imshow(attn[h], aspect="auto", cmap="viridis")
            axes[h].set_title(f"Head {h}")
            axes[h].set_xlabel("Meteorological hour (key)")
            if h == 0:
                axes[h].set_ylabel("PM2.5 hour (query)")
            fig.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

        out_path = f"{OUT_DIR}/attention_{city_label}_window{idx}_rank{rank}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\n{args.n_samples} attention plots saved to {OUT_DIR}/")
    print(
        "Reminder: these show TEMPORAL attention over meteorological hours, "
        "not per-variable attention -- see script docstring before writing this up."
    )


if __name__ == "__main__":
    main()
