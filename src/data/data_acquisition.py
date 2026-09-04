"""Dataset 2 (ERA5, via Open-Meteo Archive API) acquisition.

Dataset 1 (NASA MERRA-2 via Google Earth Engine) is pulled with a
JavaScript export script run in the GEE Code Editor -- see
`gee_export_merra2.js` in this same directory. It cannot be run from
this Python file because Earth Engine's table export/Drive-export flow
is normally driven from the Code Editor UI (Tasks tab).

IMPORTANT: resolve the PM2.5 formula / band-name question (implementation
guide Section 7) BEFORE running the GEE export -- it determines whether
you pull the SO4CMASS or SO4SMASS band.
"""
from __future__ import annotations

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

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

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]

DATE_START = "2022-08-01"
DATE_END = "2024-12-30"


def fetch_era5_for_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache_era5", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": DATE_START,
        "end_date": DATE_END,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Manila",  # NOTE: confirm this matches MERRA-2's timezone convention
    }
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    resp = responses[0]
    hourly = resp.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    data = {"datetime": dates, "city": city}
    for i, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def fetch_all_cities() -> pd.DataFrame:
    frames = [fetch_era5_for_city(city, lat, lon) for city, (lat, lon) in CITIES.items()]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_all_cities()
    df.to_csv("data/raw/era5_all_cities.csv", index=False)
    print(f"Saved {len(df):,} rows across {df.city.nunique()} cities.")
