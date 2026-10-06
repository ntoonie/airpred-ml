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

C+ (improved AIRPRED) and ablations -- see docs/variant_c_plus.md:
    python train_local.py --variants CPLUS --seeds 42             # improved C+
    python train_local.py --variants C CMS CATT CPLUS CPLUSBI --seeds 42 123 7
    python train_local.py --variants C --residual --seeds 42      # C + residual forecasting
    python train_local.py --variants CPLUS --lr 3e-4 --weight-decay 1e-5 --attn-dropout 0.2 --tag _lr3e-4_wd1e-5_ad0.2
  Every run that changes lr / weight decay / dropout / residual MUST get its own --tag
  (a residual run gets "_resid" automatically); the tag goes into the checkpoint, results
  CSV and run-config JSON names so nothing overwrites the baseline files. Selection is by
  validation RMSE only -- this script never reads the test split.

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

import json

from src.models.variants import VARIANT_KEYS, build_variant
from src.training.train import train_variant, set_seed

DATA_DIR = "data/final"
CKPT_DIR = "checkpoints"
RESULTS_CSV = "results/multiseed_val_rmse.csv"

DEFAULT_SEEDS = [42, 123, 2024, 7, 99]

DEFAULT_VARIANTS = ["A", "B", "C", "D"]   # default run unchanged; C+ keys are opt-in via --variants


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


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
        "--variants", type=str, nargs="+", default=DEFAULT_VARIANTS,
        choices=VARIANT_KEYS,
        help=f"Which variants to run (default: {DEFAULT_VARIANTS}). C+ ablation keys: "
             f"CMS (C + multi-scale TCN), CATT (C + improved attention), CPLUS (both), "
             f"CPLUSBI (CPLUS + bidirectional attention).",
    )
    parser.add_argument("--lr", type=float, default=None, help="override training.learning_rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="override training.weight_decay (Adam L2)")
    parser.add_argument("--max-epochs", type=int, default=None, help="override training.max_epochs")
    parser.add_argument("--patience", type=int, default=None, help="override training.early_stopping_patience")
    parser.add_argument("--attn-dropout", type=float, default=None,
                        help="override model.c_plus.attention_dropout (C+ variants that use dropout; "
                             "CMS pins it to 0 on purpose, original C has none)")
    parser.add_argument("--residual", action="store_true",
                        help="residual forecasting: out = last observed PM2.5 + model(x). Adds _resid to names.")
    parser.add_argument("--tag", default="", help="suffix for checkpoint/CSV/JSON names, e.g. _lr3e-4")
    args = parser.parse_args()
    tag = ("_resid" if args.residual else "") + args.tag
    variants_to_run = list(args.variants)

    cfg = yaml.safe_load(open("configs/config.yaml"))
    if args.lr is not None: cfg["training"]["learning_rate"] = args.lr
    if args.weight_decay is not None: cfg["training"]["weight_decay"] = args.weight_decay
    if args.max_epochs is not None: cfg["training"]["max_epochs"] = args.max_epochs
    if args.patience is not None: cfg["training"]["early_stopping_patience"] = args.patience
    overrides = {"attention_dropout": args.attn_dropout} if args.attn_dropout is not None else None
    # C+ keys without an explicit tag still get their own CSV, so the baseline results file is never overwritten.
    csv_tag = tag or ("_cplus" if any(v not in DEFAULT_VARIANTS for v in variants_to_run) else "")
    results_csv = RESULTS_CSV.replace(".csv", f"{csv_tag}.csv")
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
    if tag:
        print(f"Run tag: '{tag}'  (baseline files are not touched)")

    print(f"Seeds this run: {args.seeds}\n")

    # (variant, seed) -> best_val_rmse
    all_results: dict[tuple[str, int], float] = {}
    param_counts: dict[str, int] = {}

    for seed in args.seeds:
        for name in variants_to_run:
            print(f"\n{'='*60}\nTraining Variant {name} -- seed {seed}\n{'='*60}")
            set_seed(seed)  # same placement as the earlier fix: before model construction
            model = build_variant(name, horizon, cfg, residual=args.residual, overrides=overrides)
            n_params = sum(p.numel() for p in model.parameters())
            param_counts[name] = n_params
            print(f"  parameters: {n_params:,}")
            ckpt_path = f"{CKPT_DIR}/variant_{name.lower()}_seed{seed}{tag}_best.pt"
            best_rmse = train_variant(model, train_data, val_data, cfg, ckpt_path)
            all_results[(name, seed)] = best_rmse
            print(f"Variant {name}, seed {seed} -- best val RMSE (scaled): {best_rmse:.4f}")

    # Every (variant, seed) result goes to CSV so nothing has to be re-derived from logs later.
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seed", "best_val_rmse"])
        for (name, seed), rmse in all_results.items():
            writer.writerow([name, seed, f"{rmse:.6f}"])
    print(f"\nPer-(variant, seed) results written to {results_csv}")
    # Sidecar: everything needed to say later exactly what produced these numbers.
    with open(results_csv.replace(".csv", "_config.json"), "w") as f:
        json.dump({"variants": variants_to_run, "seeds": args.seeds, "residual": args.residual, "tag": tag,
                   "training": cfg["training"], "model": cfg["model"], "attention_dropout_override": args.attn_dropout,
                   "parameters": param_counts}, f, indent=2, default=str)

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
                path = f"{CKPT_DIR}/variant_{name.lower()}_seed{seed}{tag}_best.pt"
                if os.path.exists(path):
                    os.remove(path)
        print("\nKept only each variant's best-seed checkpoint (pass --keep-checkpoints to keep all).")

    print("\nDone.")
    print(summary)


if __name__ == "__main__":
    main()