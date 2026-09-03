# AIRPRED — Implementation Skeleton

This is the runnable code skeleton that accompanies `AIRPRED_Implementation_Guide.md`.
Read that document first — it explains every design decision, flags everything
the thesis leaves unspecified, and lists the critical items (PM2.5 formula,
d_model, receptive field) that need to be resolved before this pipeline
produces numbers you'd want in Chapter 4.

## Quick orientation

```
configs/config.yaml          all experiment parameters (TODOs = unresolved thesis gaps)
src/data/                    MERRA-2 (GEE) + ERA5 (Open-Meteo) acquisition
src/preprocessing/           PM2.5 derivation, merge, clean, split, scale, window
src/models/                  TCN encoder, cross-modal attention, all 4 variants
src/training/                shared training loop (early stopping, checkpointing)
src/evaluation/               inference, metrics, paired t-tests
src/visualization/           plotting helpers for Chapter 4 figures
```

## Verified so far

- `python3 src/models/tcn.py` — TCN encoder shape checks pass.
- `python3 src/models/attention.py` — cross-modal attention shape checks pass.
- `python3 -m src.models.variants` — all four variants forward-pass and
  produce `(batch, 24)` output; parameter counts scale sensibly
  (A ≈ 240K, D ≈ 239K, B ≈ 476K, C/AIRPRED ≈ 691K).
- Preprocessing pipeline (`derive_pm25_thesis_formula`, `merge_datasets`,
  `chronological_split`, `fit_transform_split`, `build_dataset`) ran
  end-to-end on a synthetic 2-city dataset and produced correctly-shaped,
  non-leaking train/val/test tensors, correctly dropping windows that
  crossed a split boundary.

## NOT yet run against real data

Nothing here has touched actual MERRA-2 or ERA5 data. Before you do:

1. Resolve the PM2.5 formula question (guide Section 7 / `preprocessing.py`
   docstring) — it determines which GEE band to export.
2. Fill in every `TODO` in `configs/config.yaml`.
3. Confirm `d_model` (guide Section 16/19) with your adviser.

## Running training once data is ready

```python
from src.preprocessing.preprocessing import build_dataset  # etc.
from src.models.variants import VariantA_SingleBranchUnified, VariantB_DualBranchConcat, \
    VariantC_AIRPRED, VariantD_PM25Only
from src.training.train import train_variant
import yaml

cfg = yaml.safe_load(open("configs/config.yaml"))
# ds = build_dataset(...)  -> {"train": (Xp, Xm, Y, cities), "val": ..., "test": ...}
model = VariantC_AIRPRED(horizon=cfg["data"]["forecast_horizon"])
train_variant(model, ds["train"][:3], ds["val"][:3], cfg, "checkpoints/variant_c/best.pt")
```

Repeat for Variants A, B, D with identical `cfg` (fair-comparison requirement).
