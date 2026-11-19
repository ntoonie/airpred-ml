"""MERRA-2 vs OpenAQ (Manila) bias validation, as described in Chapter 3.

Usage (from the project root):
    python validate_openaq_manila.py --openaq PATH_TO_OPENAQ_MANILA.csv --merra PATH_TO_MERRA2_Manila.csv

Assumptions you must check:
  * OpenAQ CSV has a timestamp column (default 'datetime') and a PM2.5 column
    (default 'pm25'). If the timestamps have no UTC offset they are ASSUMED to be UTC.
  * MERRA-2 CSV is the GEE export (columns: datetime, BCSMASS, OCSMASS, SO4SMASS,
    DUSMASS25, SSSMASS25), timestamps in UTC.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--openaq", required=True)
ap.add_argument("--merra", required=True)
ap.add_argument("--oaq-dt-col", default="datetime")
ap.add_argument("--oaq-val-col", default="pm25")
ap.add_argument("--station-col", default=None, help="column that names the station/location (needed if the file has several stations)")
ap.add_argument("--station", default=None, help="value in --station-col to keep, e.g. Manila")
ap.add_argument("--start", default="2023-09-06 00:00")
ap.add_argument("--end", default="2023-10-23 23:59")
ap.add_argument("--max-valid", type=float, default=200.0)
args = ap.parse_args()

# ---- OpenAQ (observations) ----
oaq = pd.read_csv(args.openaq)
if args.station_col:
    print("Stations in file:", sorted(oaq[args.station_col].astype(str).unique()))
    oaq = oaq[oaq[args.station_col].astype(str) == args.station]
    print(f"Rows kept for station '{args.station}': {len(oaq)}")
oaq["datetime"] = pd.to_datetime(oaq[args.oaq_dt_col], utc=True).dt.tz_convert(None)
oaq["obs"] = pd.to_numeric(oaq[args.oaq_val_col], errors="coerce")
oaq = oaq.dropna(subset=["obs"])
oaq = oaq[(oaq["datetime"] >= args.start) & (oaq["datetime"] <= args.end)]
n_window = len(oaq)
oaq = oaq[oaq["obs"] <= args.max_valid]
print(f"OpenAQ readings in window: {n_window}  | removed >{args.max_valid:g}: {n_window - len(oaq)}  | kept: {len(oaq)}")
oaq["hour"] = oaq["datetime"].dt.floor("h")
obs = oaq.groupby("hour")["obs"].mean()

# ---- MERRA-2 (model), NASA GMAO PM2.5 formula used by the pipeline ----
m = pd.read_csv(args.merra)
m["datetime"] = pd.to_datetime(m["datetime"], utc=True).dt.tz_convert(None)
c = m[["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]] * 1e9
m["merra"] = c["DUSMASS25"] + c["OCSMASS"] + c["BCSMASS"] + c["SSSMASS25"] + 1.375 * c["SO4SMASS"]
m["hour"] = m["datetime"].dt.floor("h")
merra = m.groupby("hour")["merra"].mean()

# ---- merge and metrics ----
df = pd.concat([obs, merra], axis=1, join="inner").dropna()
print(f"Matched hours: {len(df)}")
r, _ = stats.pearsonr(df["merra"], df["obs"])
mbe = float((df["merra"] - df["obs"]).mean())
rmse = float(np.sqrt(((df["merra"] - df["obs"]) ** 2).mean()))
print(f"Mean OpenAQ = {df['obs'].mean():.2f} ug/m3 | Mean MERRA-2 = {df['merra'].mean():.2f} ug/m3")
print(f"Pearson r = {r:.4f}")
print(f"MBE (MERRA-2 - OpenAQ) = {mbe:.2f} ug/m3")
print(f"RMSE = {rmse:.2f} ug/m3")