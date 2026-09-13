"""Local equivalent of AIRPRED_Stage2_Colab.ipynb -- trains all four variants
on a local machine's GPU instead of Colab's. Now runs each variant across
multiple random seeds, since RQ2/RQ3 are close enough that a single seed's
result can't be trusted on its own (see prior run: direction flipped between
an unseeded and a seeded run).

USAGE:
    python train_local.py                          # default seed sweep (42, 123, 2024, 7, 99)
    python train_local.py --seeds 42                # single seed, old behavior
    python train_local.py --seeds 42 123 7 2024 99  # explicit seed list
    python train_local.py --keep-checkpoints        # keep all (variant, seed) checkpoints,
                                                      # not just each variant's best seed
    python train_local.py --variants D --seeds 42   # run just one variant, one seed
                                                      # (e.g. a quick diagnostic re-run)

PREREQUISITES:
  1. Stage 1 must have already run (in Colab, as before) and produced
     train.npz, val.npz, test.npz, scaler.pkl.
  2. Download those 4 files from Drive (AIRPRED/data/final/) onto this
     machine, into this repo's data/final/ folder.
  3. pip install -r requirements.txt in an activated venv.
  4. Confirm GPU is visible: python -c "import torch; print(torch.cuda.is_available())"
"""
from __future__ import annotations

import argparse
import csv
import os
import yaml
import numpy as np
import torch

from src.models.variants import (
    VariantA_SingleBranchUnified,
    VariantB_DualBranchConcat,
    VariantC_AIRPRED,
    VariantD_PM25Only,
)
from src.training.train import train_variant, set_seed

DATA_DIR = "data/final"
CKPT_DIR = "checkpoints"
RESULTS_CSV = "results/multiseed_val_rmse.csv"

DEFAULT_SEEDS = [42, 123, 2024, 7, 99]

VARIANTS = {
    "A": VariantA_SingleBranchUnified,
    "B": VariantB_DualBranchConcat,
    "C": VariantC_AIRPRED,
    "D": VariantD_PM25Only,
}


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


def build_model(name: str, ModelClass, horizon: int, cfg: dict):
    if name != "C":
        return ModelClass(horizon=horizon)
    return ModelClass(
        horizon=horizon,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_attention_heads"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to run each variant with (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help="Keep every (variant, seed) checkpoint instead of only each "
             "variant's best-seed checkpoint.",
    )
    parser.add_argument(
        "--variants", type=str, nargs="+", default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help=f"Which variants to run (default: all {list(VARIANTS.keys())}). "
             f"E.g. --variants D for a quick single-variant diagnostic run.",
    )
    args = parser.parse_args()
    variants_to_run = {name: VARIANTS[name] for name in args.variants}

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

    horizon = cfg["data"]["forecast_horizon"]
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    print(f"Seeds this run: {args.seeds}\n")

    # (variant, seed) -> best_val_rmse
    all_results: dict[tuple[str, int], float] = {}

    for seed in args.seeds:
        for name, ModelClass in variants_to_run.items():
            print(f"\n{'='*60}\nTraining Variant {name} -- seed {seed}\n{'='*60}")
            set_seed(seed)  # same placement as the earlier fix: before model construction
            model = build_model(name, ModelClass, horizon, cfg)
            ckpt_path = f"{CKPT_DIR}/variant_{name.lower()}_seed{seed}_best.pt"
            best_rmse = train_variant(model, train_data, val_data, cfg, ckpt_path)
            all_results[(name, seed)] = best_rmse
            print(f"Variant {name}, seed {seed} -- best val RMSE (scaled): {best_rmse:.4f}")

    # Every (variant, seed) result goes to CSV so nothing has to be re-derived from logs later.
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seed", "best_val_rmse"])
        for (name, seed), rmse in all_results.items():
            writer.writerow([name, seed, f"{rmse:.6f}"])
    print(f"\nPer-(variant, seed) results written to {RESULTS_CSV}")

    # Aggregate: mean +/- std per variant across seeds.
    print(f"\n{'='*60}\nAggregate across {len(args.seeds)} seeds\n{'='*60}")
    summary = {}
    for name in variants_to_run:
        vals = np.array([all_results[(name, s)] for s in args.seeds])
        summary[name] = (float(vals.mean()), float(vals.std()))
        print(f"Variant {name}: mean={vals.mean():.4f}  std={vals.std():.4f}  (n={len(vals)})")

    # Housekeeping: by default, keep only the checkpoint for each variant's
    # best-performing seed and delete the rest. A full sweep produces
    # len(VARIANTS) x len(seeds) checkpoints -- each only a few MB for a
    # model this size -- so this is a convenience default, not a necessity.
    if not args.keep_checkpoints:
        for name in variants_to_run:
            best_seed = min(args.seeds, key=lambda s: all_results[(name, s)])
            for seed in args.seeds:
                if seed == best_seed:
                    continue
                path = f"{CKPT_DIR}/variant_{name.lower()}_seed{seed}_best.pt"
                if os.path.exists(path):
                    os.remove(path)
        print("\nKept only each variant's best-seed checkpoint (pass --keep-checkpoints to keep all).")

    print("\nDone.")
    print(summary)


if __name__ == "__main__":
    main()