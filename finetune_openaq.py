"""Fine-tunes the trained AIRPRED variants (A-D) on OpenAQ ground-sensor data.

Put this in the airpred-ml repo root (next to train_local.py). It reuses your own
code -- src.models.variants, src.preprocessing.preprocessing, src.evaluation.evaluate --
so cleaning, scaling and windowing match the original pipeline. Defaults reproduce the
first fine-tune run (MSE, Adam-equivalent, lr 1e-4, single seed). Everything new is an
opt-in flag, so existing results and checkpoints are not changed unless you ask.

INPUT: data/openaq/<City>.csv files made by fetch_openaq_history.py (copy that
folder into this repo). Columns: datetime, pm25, then the 7 met variables.

WHAT IT DOES
  1. Loads the ORIGINAL scaler (data/final/scaler.pkl) -- not refit.
  2. Cleans like the original pipeline: pm25 < 0 or > 500 -> NaN, gaps <= 3 h
     interpolated, windows touching a NaN dropped.
  3. Global chronological split 70/10/20. Test is the LATEST period and is never used
     for early stopping or for choosing settings -- pick settings on VAL only.
  4. Per variant and per fine-tune seed: loads checkpoints/variant_<x>_seed<S>_best.pt,
     scores it as-is ("zero_shot"), fine-tunes with early stopping on val RMSE, scores
     again ("fine_tuned"), plus a naive "persistence" baseline.
  5. Writes results/openaq_finetune_results<tag>.csv; with several seeds also prints
     mean +/- std per variant, so you can see whether C-vs-D gaps are real or noise.

OPT-IN IMPROVEMENTS (all for the OpenAQ extension only; Chapter 4 is untouched)
  --seeds 42 123 7      repeat fine-tuning with different shuffling/dropout seeds
  --weight-decay 1e-4   L2-style regularisation (AdamW; 0 = plain Adam as before)
  --attn-dropout 0.1    dropout inside C's cross-modal attention (ignored for A/B/D)
  --freeze-encoders     tune only fusion + head
  --loss huber|peak     huber: less dominated by outliers. peak: MSE that weights
                        above-average targets more, to fight flat/under-predicting
                        forecasts on dirty days (--peak-alpha sets the strength)
  --residual            predict the CHANGE from the last observed PM2.5. The output
                        layer is zero-initialised, so training starts from "repeat the
                        last hour" (persistence) and learns corrections. Saved
                        checkpoints are the base model; they need the residual wrapper
                        (out = last_pm25 + model(x)) at inference -- tell me if this wins
                        and I will wire it into main.py.

CHECKPOINT NAMES
  No flags, one seed : variant_<x>_seed<S>_openaq_ft.pt           (what main.py loads)
  Otherwise          : variant_<x>_seed<S>_openaq_ft<tag>_fs<seed>.pt
  so experiments never overwrite the deployed checkpoints. <tag> is auto-built from
  the flags (e.g. _resid_huber_wd1e-04) unless you pass --tag.

USAGE
    python finetune_openaq.py                                   # same as before
    python finetune_openaq.py --variants C D --seeds 42 123 7   # is C-vs-D real?
    python finetune_openaq.py --variants C --freeze-encoders --weight-decay 1e-4 --attn-dropout 0.1
    python finetune_openaq.py --variants C D --residual --seeds 42 123 7

HOW TO READ THE RESULT
  - fine_tuned should beat zero_shot AND persistence on TEST.
  - Choose settings by VAL results; look at TEST once at the end.
  - Report this as a separate OpenAQ extension, NOT a replacement for Chapter 4.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluate import compute_metrics, evaluate_model, inverse_scale_pm25
from src.models.variants import VARIANT_KEYS, build_variant
from src.preprocessing.preprocessing import (
    FEATURE_COLS,
    build_dataset,
    chronological_cutoffs,
    interpolate_gaps,
)
from src.training.train import set_seed

VARIANTS = VARIANT_KEYS                       # A-D as before, plus the C+ ablation keys (CMS, CATT, CPLUS, CPLUSBI)
DEFAULT_VARIANTS = ["A", "B", "C", "D"]


class ResidualWrapper(nn.Module):
    """out = last observed (scaled) PM2.5 + base(x). Same call signature as the variants."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x_pm25, x_met=None):
        return self.base(x_pm25, x_met) + x_pm25[:, -1, :]      # (B,1) broadcasts over horizon


def build_model(name: str, horizon: int, cfg: dict):
    return build_variant(name, horizon, cfg)


def load_city_frames(data_dir: str, cities: list[str] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        city = os.path.splitext(os.path.basename(path))[0]
        if city == "station_summary" or (cities and city not in cities):
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{path} is missing columns: {missing}")
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        bad = (df["pm25"] < 0) | (df["pm25"] > 500)
        df.loc[bad, "pm25"] = np.nan
        df = interpolate_gaps(df[["datetime"] + FEATURE_COLS], max_gap_hours=3)
        df["city"] = city
        n_ok = int(df["pm25"].notna().sum())
        print(f"  {city}: {len(df)} hours, {n_ok} with pm25 ({100 * n_ok / len(df):.0f}%), "
              f"{df['datetime'].min():%Y-%m-%d} -> {df['datetime'].max():%Y-%m-%d}")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No city CSVs found in {data_dir}/")
    return pd.concat(frames, ignore_index=True)


def score(model, split, scaler, device):
    xp, xm, y, cities = split
    preds, trues, _ = evaluate_model(model, xp, xm, y, cities, scaler, device)
    return compute_metrics(trues.ravel(), preds.ravel())


def persistence_metrics(split, scaler):
    xp, _, y, _ = split
    last = inverse_scale_pm25(xp[:, -1, 0], scaler)
    preds = np.repeat(last[:, None], y.shape[1], axis=1)
    trues = inverse_scale_pm25(y, scaler)
    return compute_metrics(trues.ravel(), preds.ravel())


def make_loss(kind: str, y_train: np.ndarray, peak_alpha: float):
    if kind == "mse":
        return nn.MSELoss()
    if kind == "huber":
        return nn.HuberLoss(delta=1.0)             # delta in scaled (z-score) units
    mu, sd = float(y_train.mean()), float(y_train.std() + 1e-8)

    def peak_loss(pred, target):
        w = 1.0 + peak_alpha * torch.relu((target - mu) / sd)
        return (w * (pred - target) ** 2).mean()
    return peak_loss


def fit(model, train, val, cfg, args, loss_fn, ckpt_path, device, seed):
    """Same recipe as src.training.train.train_variant (early stopping on val RMSE in
    scaled units, best epoch saved) plus weight decay and a pluggable loss."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    bs = cfg["training"]["batch_size"]
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(TensorDataset(*[torch.as_tensor(a) for a in train]), batch_size=bs, shuffle=True, generator=g)
    vl = DataLoader(TensorDataset(*[torch.as_tensor(a) for a in val]), batch_size=bs, shuffle=False)
    best, patience = float("inf"), args.patience
    base = model.base if isinstance(model, ResidualWrapper) else model
    for epoch in range(args.epochs):
        model.train()
        tr_se, tr_n = 0.0, 0
        for xp, xm, y in tl:
            xp, xm, y = xp.to(device), xm.to(device), y.to(device)
            opt.zero_grad()
            yh = model(xp, xm)
            loss_fn(yh, y).backward()
            opt.step()
            tr_se += ((yh.detach() - y) ** 2).sum().item()
            tr_n += y.numel()
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for xp, xm, y in vl:
                xp, xm, y = xp.to(device), xm.to(device), y.to(device)
                se += ((model(xp, xm) - y) ** 2).sum().item()
                n += y.numel()
        val_rmse = (se / n) ** 0.5
        print(f"epoch {epoch:3d}  train_rmse={(tr_se / tr_n) ** 0.5:.4f}  val_rmse={val_rmse:.4f}  patience_left={patience}")
        if val_rmse < best:
            best, patience = val_rmse, args.patience
            torch.save(base.state_dict(), ckpt_path)       # base weights; residual wrapper is code, not weights
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch}, best val RMSE={best:.4f}")
                break
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/openaq")
    ap.add_argument("--scaler", default="data/final/scaler.pkl")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42, help="which ORIGINAL checkpoint seed to start from (default 42, what main.py serves)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="fine-tuning seeds (shuffling/dropout); several -> mean +/- std")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=VARIANTS)
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--attn-dropout", type=float, default=0.0, help="dropout in C's cross-modal attention (C only)")
    ap.add_argument("--freeze-encoders", action="store_true", help="freeze the TCN encoders; tune only fusion + head")
    ap.add_argument("--loss", choices=["mse", "huber", "peak"], default="mse")
    ap.add_argument("--peak-alpha", type=float, default=1.0, help="strength of --loss peak (default 1.0)")
    ap.add_argument("--residual", action="store_true", help="predict the change from the last observed PM2.5 (zero-initialised head)")
    ap.add_argument("--tag", default=None, help="suffix for checkpoint/result names (auto-built from flags if omitted)")
    args = ap.parse_args()

    parts = []
    if args.residual: parts.append("resid")
    if args.loss != "mse": parts.append(args.loss)
    if args.weight_decay > 0: parts.append(f"wd{args.weight_decay:.0e}")
    if args.attn_dropout > 0: parts.append(f"ad{args.attn_dropout:g}")
    if args.freeze_encoders: parts.append("frz")
    auto_tag = ("_" + "_".join(parts)) if parts else ""
    tag = args.tag if args.tag is not None else auto_tag
    plain = (tag == "" and len(args.seeds) == 1)
    out_csv = f"results/openaq_finetune_results{tag}.csv"

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizon = cfg["data"]["forecast_horizon"]
    L = cfg["data"]["input_window"]
    print(f"Device: {device}   tag: '{tag}'   fine-tune seeds: {args.seeds}")

    scaler = joblib.load(args.scaler)
    assert list(scaler.feature_names_in_) == FEATURE_COLS, "scaler columns differ from FEATURE_COLS"

    print("\nLoading OpenAQ city data:")
    df = load_city_frames(args.data_dir, args.cities)
    if df["datetime"].min() < pd.Timestamp("2025-01-01"):
        print("\nNOTE: data before 2025-01-01 overlaps the MERRA-2 training period; don't compare "
              "these numbers directly to the Chapter 4 test results.")

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])         # ORIGINAL scaler, not refit
    c1, c2 = chronological_cutoffs(df["datetime"], cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    print(f"\nSplit cutoffs: train < {pd.Timestamp(c1):%Y-%m-%d %H:%M} <= val < {pd.Timestamp(c2):%Y-%m-%d %H:%M} <= test")
    ds = build_dataset(df, c1, c2, L=L, H=horizon)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(ds[k][2])} windows")
    if min(len(ds[k][2]) for k in ds) == 0:
        raise SystemExit("A split has zero windows -- not enough continuous OpenAQ data for fine-tuning.")
    if len(ds["train"][2]) < 2000:
        print("WARNING: fewer than ~2000 training windows -- expect a noisy, overfit-prone result.")

    os.makedirs("results", exist_ok=True)
    rows = []

    def add(variant, split_name, model_tag, m, n, fs=""):
        rows.append({"variant": variant, "split": split_name, "model": model_tag, "ft_seed": fs, "n_windows": n,
                     **{k: round(v, 4) for k, v in m.items()}})

    for name in args.variants:
        print(f"\n{'=' * 60}\nVariant {name}\n{'=' * 60}")
        orig = os.path.join(args.ckpt_dir, f"variant_{name.lower()}_seed{args.seed}_best.pt")
        if not os.path.isfile(orig):
            print(f"  {orig} not found -- skipped")
            continue
        base = build_model(name, horizon, cfg)
        base.load_state_dict(torch.load(orig, map_location=device))
        base.to(device)
        for split_name in ("val", "test"):
            n = len(ds[split_name][2])
            add(name, split_name, "zero_shot", score(base, ds[split_name], scaler, device), n)
            if name == args.variants[0]:
                add("-", split_name, "persistence", persistence_metrics(ds[split_name], scaler), n)

        for fs in args.seeds:
            print(f"\n-- {name}: fine-tune seed {fs} --")
            set_seed(fs)
            net = copy.deepcopy(base)
            if args.attn_dropout > 0 and hasattr(net, "fusion"):
                for m in net.fusion.modules():                   # C has one MHA; CPLUSBI has two
                    if isinstance(m, nn.MultiheadAttention):
                        m.dropout = args.attn_dropout            # nn.MultiheadAttention reads .dropout in train mode
            if args.freeze_encoders:
                for pname, p in net.named_parameters():
                    if "encoder" in pname:
                        p.requires_grad = False
            if args.residual:
                nn.init.zeros_(net.head.fc2.weight)              # start exactly at persistence
                nn.init.zeros_(net.head.fc2.bias)
                net = ResidualWrapper(net)
            n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"  trainable parameters: {n_train:,}")
            ckpt = os.path.join(args.ckpt_dir, f"variant_{name.lower()}_seed{args.seed}_openaq_ft"
                                + ("" if plain else f"{tag}_fs{fs}") + ".pt")
            loss_fn = make_loss(args.loss, ds["train"][2], args.peak_alpha)
            best_val = fit(net, ds["train"][:3], ds["val"][:3], cfg, args, loss_fn, ckpt, device, fs)

            tuned_base = build_model(name, horizon, cfg)
            tuned_base.load_state_dict(torch.load(ckpt, map_location=device))
            tuned = ResidualWrapper(tuned_base) if args.residual else tuned_base
            tuned.to(device)
            for split_name in ("val", "test"):
                add(name, split_name, "fine_tuned", score(tuned, ds[split_name], scaler, device),
                    len(ds[split_name][2]), fs)
            print(f"  saved {ckpt}  (best val RMSE, scaled units: {best_val:.4f})")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "split", "model", "ft_seed", "n_windows", "rmse", "mae", "mape", "r2"])
        w.writeheader()
        w.writerows(rows)

    res = pd.DataFrame(rows)
    for split_name in ("val", "test"):
        print(f"\n{'=' * 60}\n{split_name.upper()} results on OpenAQ (ug/m3)\n{'=' * 60}")
        sub = res[res["split"] == split_name]
        print(f"{'variant':8s}{'model':13s}{'RMSE':>14s}{'MAE':>8s}{'R2':>8s}")
        for (v, m), g in sub.groupby(["variant", "model"], sort=False):
            rm = f"{g['rmse'].mean():.2f}" + (f" +/-{g['rmse'].std(ddof=0):.2f}" if len(g) > 1 else "")
            print(f"{v:8s}{m:13s}{rm:>14s}{g['mae'].mean():8.2f}{g['r2'].mean():8.3f}")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()