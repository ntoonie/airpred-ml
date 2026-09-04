"""Pull ERA5 boundary_layer_height directly from the Copernicus Climate Data
Store (CDS) -- the authoritative source Open-Meteo itself wraps. Used to
check/fix the ~6-month BLH gap found in the Open-Meteo pull (Jan-Jun 2024).

WHY THIS INSTEAD OF MERRA-2 PBLH: this is the exact same ERA5 product your
other 6 meteorological variables already come from -- same definition ("blh",
parameter 159, bulk-Richardson-number method), same ~0.25 degree resolution.
No bias correction needed if it agrees with your existing data, unlike the
MERRA-2 substitution (different definition, coarser grid, city-invariant,
seasonally-varying bias found in chat).

SETUP (one-time, separate from your GEE and Open-Meteo accounts):
  1. Register at https://cds.climate.copernicus.eu
  2. Accept the ERA5 single-levels dataset license on the CDS website
     (Dataset page -> "Terms of use" tab) -- requests fail without this.
  3. pip install cdsapi
  4. Get your API key from your CDS profile page, save to ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api
       key: <your-uid>:<your-api-key>

USAGE:
  python cds_era5_blh.py
"""
from __future__ import annotations

import cdsapi
import xarray as xr
import pandas as pd

CITIES: dict[str, tuple[float, float]] = {
    "Manila": (14.5995, 120.9842),
    "Quezon_City": (14.6760, 121.0437),
    "Caloocan": (14.6760, 120.9663),
    "Valenzuela": (14.7011, 120.9830),
    "Pasig": (14.5764, 121.0851),
    "Makati": (14.5547, 121.0244),
    "Mandaluyong": (14.5794, 121.0359),
    "Navotas": (14.6667, 120.9417),
    "Pasay": (14.5378, 120.9972),
    "San_Juan": (14.6000, 121.0333),
}

# Pull a window that OVERLAPS existing good ERA5 months (for validation)
# plus the full gap, rather than just the gap alone.
YEARS = ["2023", "2024"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]


def fetch_blh_for_city(client: cdsapi.Client, city: str, lat: float, lon: float, out_path: str):
    # CDS area format: [North, West, South, East]. ERA5's native grid is
    # 0.25 degrees -- a box smaller than that can fail CDS's area-crop step
    # entirely ("non-empty area crop/mask" MARS error) if it doesn't cleanly
    # straddle a real grid point. Use a box comfortably larger than one grid
    # cell, then pick the nearest actual grid point to the city coordinate
    # client-side (see netcdf_to_dataframe) rather than averaging the box.
    area = [lat + 0.3, lon - 0.3, lat - 0.3, lon + 0.3]
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "boundary_layer_height",
            "year": YEARS,
            "month": MONTHS,
            "day": DAYS,
            "time": TIMES,
            "area": area,
            "data_format": "netcdf",  # 'format' is deprecated by CDS as of 2026
        },
        out_path,
    )


def netcdf_to_dataframe(nc_path: str, city: str, lat: float, lon: float) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    # Box now spans multiple grid cells (0.6deg box vs 0.25deg native grid) --
    # select the single nearest actual grid point to the city coordinate,
    # rather than averaging across the box (which would blur in neighboring
    # cells and no longer represent "this city's" value specifically).
    blh = ds["blh"].sel(latitude=lat, longitude=lon, method="nearest")
    df = blh.to_dataframe(name="blh_cds").reset_index()
    # The new CDS backend has used both "time" and "valid_time" as the
    # datetime dimension name across different dataset/format versions --
    # handle either rather than assume one and fail on the other.
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df[[time_col, "blh_cds"]].rename(columns={time_col: "datetime"})
    df["city"] = city
    return df


if __name__ == "__main__":
    client = cdsapi.Client()
    frames = []
    for city, (lat, lon) in CITIES.items():
        out_path = f"cds_blh_{city}.nc"
        print(f"Requesting {city}...")
        fetch_blh_for_city(client, city, lat, lon, out_path)
        frames.append(netcdf_to_dataframe(out_path, city, lat, lon))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv("era5_blh_from_cds.csv", index=False)
    print(f"Saved {len(result):,} rows to era5_blh_from_cds.csv")