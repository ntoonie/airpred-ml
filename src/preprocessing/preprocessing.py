from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

KG_M3_TO_UG_M3 = 1e9  # kg/m3 only

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
MET_COLS = FEATURE_COLS[1:]  # met only


# ---------------------------------------------------------------------------
# 1. PM2.5 derivation
# ---------------------------------------------------------------------------
def derive_pm25_thesis_formula(df: pd.DataFrame) -> pd.Series:
    # thesis formula, SO4CMASS
    c = df[["BCSMASS", "OCSMASS", "SO4CMASS", "DUSMASS25", "SSSMASS25"]] * KG_M3_TO_UG_M3
    return c["BCSMASS"] + 1.8 * c["OCSMASS"] + c["SO4CMASS"] + c["DUSMASS25"] + c["SSSMASS25"]


def derive_pm25_nasa_gmao_formula(df: pd.DataFrame) -> pd.Series:
    # NASA formula, SO4SMASS
    c = df[["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]] * KG_M3_TO_UG_M3
    return c["DUSMASS25"] + c["OCSMASS"] + c["BCSMASS"] + c["SSSMASS25"] + c["SO4SMASS"] * 1.375


# ---------------------------------------------------------------------------
# 1b. Bias validation
# ---------------------------------------------------------------------------
def compute_bias_metrics(model_values: pd.Series, obs_values: pd.Series) -> dict:
    # Pearson r, MBE, RMSE
    r, _ = stats.pearsonr(model_values, obs_values)
    mbe = float((model_values - obs_values).mean())
    rmse = float(np.sqrt(((model_values - obs_values) ** 2).mean()))
    return {"n": len(model_values), "pearson_r": float(r), "mbe": mbe, "rmse": rmse}


# ---------------------------------------------------------------------------
# 1c. BLH gap-fill
# ---------------------------------------------------------------------------
def compare_blh_sources(era5_blh: pd.DataFrame, merra2_pblh: pd.DataFrame) -> dict:
    # bias check, overlap only
    merged = era5_blh.merge(merra2_pblh, on=["city", "datetime"], how="inner")
    if len(merged) == 0:
        raise ValueError("no overlap rows")
    diff = merged["PBLH"] - merged["boundary_layer_height"]
    return {
        "n_overlap_hours": len(merged),
        "era5_mean": float(merged["boundary_layer_height"].mean()),
        "merra2_mean": float(merged["PBLH"].mean()),
        "mean_diff_merra2_minus_era5": float(diff.mean()),
        "std_diff": float(diff.std()),
        "ratio_merra2_over_era5": float(merged["PBLH"].mean() / merged["boundary_layer_height"].mean()),
    }


def fill_blh_gap(
    era5_df: pd.DataFrame,
    merra2_pblh: pd.DataFrame,
    bias_correction: float = 0.0,
) -> pd.DataFrame:
    # fill NaNs, tag source
    df = era5_df.copy()
    df["blh_source"] = np.where(df["boundary_layer_height"].isna(), "merra2_gapfill", "era5")

    pblh_lookup = merra2_pblh.set_index(["city", "datetime"])["PBLH"]
    missing_mask = df["boundary_layer_height"].isna()
    keys = list(zip(df.loc[missing_mask, "city"], df.loc[missing_mask, "datetime"]))
    fill_values = pblh_lookup.reindex(keys).to_numpy() - bias_correction

    n_unfillable = np.isnan(fill_values).sum()
    if n_unfillable > 0:
        print(f"WARNING: {n_unfillable} rows unfilled")  # no match found

    df.loc[missing_mask, "boundary_layer_height"] = fill_values
    return df


# ---------------------------------------------------------------------------
# 2. Merging
# ---------------------------------------------------------------------------
def merge_datasets(pm25_df: pd.DataFrame, met_df: pd.DataFrame) -> pd.DataFrame:
    merged = pm25_df.merge(met_df, on=["city", "datetime"], how="inner")
    merged = merged.sort_values(["city", "datetime"]).reset_index(drop=True)
    assert not merged.duplicated(["city", "datetime"]).any(), "dup rows"
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
    # fill short gaps only
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
    assert train["datetime"].max() < val["datetime"].min()  # no leak
    assert val["datetime"].max() < test["datetime"].min()  # no leak
    return train, val, test


# ---------------------------------------------------------------------------
# 6. Scaling
# ---------------------------------------------------------------------------
def fit_transform_split(train_df, val_df, test_df, cols: Sequence[str] = FEATURE_COLS):
    scaler = StandardScaler()  # fit train only
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[cols] = scaler.fit_transform(train_df[cols])
    val_df[cols] = scaler.transform(val_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, val_df, test_df, scaler


# ---------------------------------------------------------------------------
# 7. Sliding windows
# ---------------------------------------------------------------------------
def make_windows_for_city(city_df: pd.DataFrame, L: int = 48, H: int = 24):
    # one city, 48-in 24-out
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
            continue  # skip gap window
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
    # per-city windows, drop boundary crossers
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