"""End-to-end preprocessing pipeline.

Order of operations (matches thesis Data Generation Procedure steps 1-8):
  1. PM2.5 derivation + unit conversion       (derive_pm25_*)
  2. Dataset merging                          (merge_datasets)
  3. Data validation                          (validate_merged)
  4. Missing value handling                   (interpolate_gaps)
  5. Chronological split                      (chronological_split)
  6. Normalization (fit on TRAIN only)        (fit_transform_split)
  7. Sliding window generation                (make_windows_for_city / build_dataset)

*** Resolve implementation guide Section 7 (PM2.5 formula) before trusting
    the output of this pipeline. Both candidate formulas are implemented
    below so you can run either once you've decided. ***
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KG_M3_TO_UG_M3 = 1e9  # valid only if inputs are genuinely kg/m3

FEATURE_COLS = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]
MET_COLS = FEATURE_COLS[1:]  # the 7 meteorological columns, PM2.5 excluded


# ---------------------------------------------------------------------------
# 1. PM2.5 derivation
# ---------------------------------------------------------------------------
def derive_pm25_thesis_formula(df: pd.DataFrame) -> pd.Series:
    """As literally written in the thesis:
    PM2.5 = BCSMASS + 1.8*OCSMASS + SO4CMASS + DUSMASS25 + SSSMASS25
    Verify SO4CMASS is really the intended band, and that the raw values
    are kg/m3 (not kg/m2), before trusting this output -- see Section 7."""
    c = df[["BCSMASS", "OCSMASS", "SO4CMASS", "DUSMASS25", "SSSMASS25"]] * KG_M3_TO_UG_M3
    return c["BCSMASS"] + 1.8 * c["OCSMASS"] + c["SO4CMASS"] + c["DUSMASS25"] + c["SSSMASS25"]


def derive_pm25_nasa_gmao_formula(df: pd.DataFrame) -> pd.Series:
    """NASA GMAO published formula (requires SO4SMASS, not SO4CMASS):
    PM2.5 = DUSMASS25 + OCSMASS + BCSMASS + SSSMASS25 + SO4SMASS*1.375"""
    c = df[["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]] * KG_M3_TO_UG_M3
    return c["DUSMASS25"] + c["OCSMASS"] + c["BCSMASS"] + c["SSSMASS25"] + c["SO4SMASS"] * 1.375


# ---------------------------------------------------------------------------
# 2. Merging
# ---------------------------------------------------------------------------
def merge_datasets(pm25_df: pd.DataFrame, met_df: pd.DataFrame) -> pd.DataFrame:
    merged = pm25_df.merge(met_df, on=["city", "datetime"], how="inner")
    merged = merged.sort_values(["city", "datetime"]).reset_index(drop=True)
    assert not merged.duplicated(["city", "datetime"]).any(), "duplicate city-hour rows after merge"
    return merged


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------
def validate_merged(df: pd.DataFrame) -> dict:
    report = {
        "n_cities": df["city"].nunique(),
        "n_rows": len(df),
        "negative_pm25": int((df["pm25"] < 0).sum()),
        "extreme_pm25": int((df["pm25"] > 500).sum()),
        "rh_out_of_range": int(((df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100)).sum()),
        "wind_dir_out_of_range": int(((df["wind_direction_10m"] < 0) | (df["wind_direction_10m"] > 360)).sum()),
        "nulls_per_col": df[FEATURE_COLS].isna().sum().to_dict(),
    }
    return report


# ---------------------------------------------------------------------------
# 4. Missing value handling
# ---------------------------------------------------------------------------
def interpolate_gaps(city_df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Linear interpolation for gaps <= max_gap_hours; longer gaps are left
    as NaN so downstream windowing can exclude them (thesis Step 5)."""
    city_df = city_df.set_index("datetime").sort_index()
    full_index = pd.date_range(city_df.index.min(), city_df.index.max(), freq="h")
    city_df = city_df.reindex(full_index)
    city_df[FEATURE_COLS] = city_df[FEATURE_COLS].interpolate(
        method="linear", limit=max_gap_hours, limit_area="inside"
    )
    city_df = city_df.rename_axis("datetime").reset_index()
    return city_df


# ---------------------------------------------------------------------------
# 5. Chronological split
# ---------------------------------------------------------------------------
def chronological_cutoffs(all_datetimes: pd.Series, train_frac=0.70, val_frac=0.10):
    unique_ts = np.sort(all_datetimes.unique())
    n = len(unique_ts)
    cutoff_1 = unique_ts[int(n * train_frac)]
    cutoff_2 = unique_ts[int(n * (train_frac + val_frac))]
    return cutoff_1, cutoff_2


def chronological_split(df: pd.DataFrame, cutoff_1, cutoff_2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["datetime"] < cutoff_1].copy()
    val = df[(df["datetime"] >= cutoff_1) & (df["datetime"] < cutoff_2)].copy()
    test = df[df["datetime"] >= cutoff_2].copy()
    assert train["datetime"].max() < val["datetime"].min()
    assert val["datetime"].max() < test["datetime"].min()
    return train, val, test


# ---------------------------------------------------------------------------
# 6. Scaling (fit on TRAIN only)
# ---------------------------------------------------------------------------
def fit_transform_split(train_df, val_df, test_df, cols: Sequence[str] = FEATURE_COLS):
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[cols] = scaler.fit_transform(train_df[cols])
    val_df[cols] = scaler.transform(val_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, val_df, test_df, scaler


# ---------------------------------------------------------------------------
# 7. Sliding window generation
# ---------------------------------------------------------------------------
def make_windows_for_city(city_df: pd.DataFrame, L: int = 48, H: int = 24):
    """city_df: single city, sorted by datetime, already scaled.
    Returns X_pm25 (n,L,1), X_met (n,L,7), Y (n,H), start_ts (n,), end_ts (n,)
    -- the first input timestep and last target timestep of each window,
    used to assign windows to a split (and to detect boundary-crossing
    windows). Windows containing any NaN (from an un-interpolated long
    gap) are dropped."""
    pm25 = city_df["pm25"].to_numpy()
    met = city_df[MET_COLS].to_numpy()
    dt = city_df["datetime"].to_numpy()
    n = len(city_df)

    X_pm25, X_met, Y, start_ts, end_ts = [], [], [], [], []
    for start in range(0, n - L - H + 1):
        in_slice = slice(start, start + L)
        out_slice = slice(start + L, start + L + H)
        x_pm = pm25[in_slice]
        x_met_w = met[in_slice]
        y = pm25[out_slice]
        if np.isnan(x_pm).any() or np.isnan(x_met_w).any() or np.isnan(y).any():
            continue  # drop windows spanning an un-interpolated gap
        X_pm25.append(x_pm[:, None])
        X_met.append(x_met_w)
        Y.append(y)
        start_ts.append(dt[start])
        end_ts.append(dt[start + L + H - 1])

    return (
        np.array(X_pm25, dtype=np.float32),
        np.array(X_met, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(start_ts),
        np.array(end_ts),
    )


def _assign_split(t, cutoff_1, cutoff_2) -> str:
    if t < cutoff_1:
        return "train"
    if t < cutoff_2:
        return "val"
    return "test"


def build_dataset(scaled_df: pd.DataFrame, cutoff_1, cutoff_2, L: int = 48, H: int = 24):
    """Generate windows PER CITY (never crossing a city boundary). A window
    is assigned to a split only if its start AND end timestamps fall in the
    SAME split; windows whose span crosses a split boundary are dropped
    entirely (implementation guide Section 15 -- this rule is a documented
    ASSUMPTION, not literally specified in the thesis)."""
    splits = {"train": [], "val": [], "test": []}
    n_dropped_boundary = 0
    for city, city_df in scaled_df.groupby("city"):
        city_df = city_df.sort_values("datetime").reset_index(drop=True)
        X_pm25, X_met, Y, start_ts, end_ts = make_windows_for_city(city_df, L, H)
        for i in range(len(end_ts)):
            split_start = _assign_split(start_ts[i], cutoff_1, cutoff_2)
            split_end = _assign_split(end_ts[i], cutoff_1, cutoff_2)
            if split_start != split_end:
                n_dropped_boundary += 1
                continue
            splits[split_start].append((X_pm25[i], X_met[i], Y[i], city))
    if n_dropped_boundary:
        print(f"Dropped {n_dropped_boundary} windows crossing a split boundary.")

    out = {}
    for split, rows in splits.items():
        if not rows:
            out[split] = (np.empty((0, L, 1)), np.empty((0, L, len(MET_COLS))), np.empty((0, H)), np.empty((0,)))
            continue
        xp, xm, y, cities = zip(*rows)
        out[split] = (np.stack(xp), np.stack(xm), np.stack(y), np.array(cities))
    return out
