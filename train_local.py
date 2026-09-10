"""Local equivalent of AIRPRED_Stage2_Colab.ipynb -- trains all four variants
on a local machine's GPU instead of Colab's. Same logic, same config.yaml,
no Colab-specific code (no Drive mount, no upload widgets).

USAGE:
    python train_local.py

PREREQUISITES:
  1. Stage 1 must have already run (in Colab, as before) and produced
     train.npz, val.npz, test.npz, scaler.pkl.
  2. Download those 4 files from Drive (AIRPRED/data/final/) onto this
     machine, into this repo's data/final/ folder.
  3. pip install -r requirements.txt in an activated venv.
  4. Confirm GPU is visible: python -c "import torch; print(torch.cuda.is_available())"
"""
from __future__ import annotations

import os
import yaml
import joblib
import numpy as np
import torch

from src.models.variants import (
    VariantA_SingleBranchUnified,
    VariantB_DualBranchConcat,
    VariantC_AIRPRED,
    VariantD_PM25Only,
)
from src.training.train import train_variant

DATA_DIR = "data/final"
CKPT_DIR = "checkpoints"


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["training"]["device"] = device
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected -- training will be much slower than expected.")

    assert os.path.exists(f"{DATA_DIR}/train.npz"), (
        f"{DATA_DIR}/train.npz not found. Download Stage 1's output from "
        f"Drive (AIRPRED/data/final/) into this repo's {DATA_DIR}/ folder first."
    )

    train_data = load_split("train")[:3]
    val_data = load_split("val")[:3]
    print(f"train: {[a.shape for a in train_data]}")
    print(f"val:   {[a.shape for a in val_data]}")
    print("Check: val sample count should be ~18,000-21,000 (post BLH-fix), not ~950.\n")

    VARIANTS = {
        "A": VariantA_SingleBranchUnified,
        "B": VariantB_DualBranchConcat,
        "C": VariantC_AIRPRED,
        "D": VariantD_PM25Only,
    }
    horizon = cfg["data"]["forecast_horizon"]
    os.makedirs(CKPT_DIR, exist_ok=True)

    best_val_rmses = {}
    for name, ModelClass in VARIANTS.items():
        print(f"\n{'='*60}\nTraining Variant {name}\n{'='*60}")
        model = (
            ModelClass(horizon=horizon)
            if name != "C"
            else ModelClass(
                horizon=horizon,
                d_model=cfg["model"]["d_model"],
                num_heads=cfg["model"]["num_attention_heads"],
            )
        )
        ckpt_path = f"{CKPT_DIR}/variant_{name.lower()}_best.pt"
        best_rmse = train_variant(model, train_data, val_data, cfg, ckpt_path)
        best_val_rmses[name] = best_rmse
        print(f"Variant {name} best val RMSE (scaled units): {best_rmse:.4f}")

    print("\nAll four variants trained. Checkpoints saved to", CKPT_DIR)
    print(best_val_rmses)


if __name__ == "__main__":
    main()
