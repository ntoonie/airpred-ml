"""Export the raw (X_pm25, X_met) INPUT windows the live inference backend
needs -- one representative 48-hour window per city (the most recent one in
the test set, same selection logic as export_dashboard_json.py). Unlike that
script, this does NOT run inference or save predictions -- the backend does
that itself, live, on every request.

This keeps the file the backend needs to ship tiny: 10 cities x 48 hours x
8 features is a few KB, versus ~70MB for the full test.npz.

USAGE:
    python export_demo_inputs.py
    python export_demo_inputs.py --out ../airpred-api/demo_inputs.npz
"""
from __future__ import annotations

import argparse

import numpy as np

DATA_DIR = "data/final"


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    return d["X_pm25"], d["X_met"], d["Y"], d["cities"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="demo_inputs.npz",
        help="Output path -- point this directly at the airpred-api backend folder.",
    )
    args = parser.parse_args()

    X_pm25, X_met, Y, cities = load_split("test")
    unique_cities = sorted(set(cities.tolist()))

    per_city_pm25, per_city_met, city_names = [], [], []
    for city in unique_cities:
        idx_for_city = np.where(cities == city)[0]
        last_idx = idx_for_city[-1]  # most recent window for this city
        per_city_pm25.append(X_pm25[last_idx])
        per_city_met.append(X_met[last_idx])
        city_names.append(city)

    X_pm25_stacked = np.stack(per_city_pm25)  # (10, 48, 1)
    X_met_stacked = np.stack(per_city_met)    # (10, 48, 7)

    np.savez(args.out, cities=np.array(city_names), X_pm25=X_pm25_stacked, X_met=X_met_stacked)
    print(f"Wrote {args.out}")
    print(f"Cities: {city_names}")
    print(f"X_pm25: {X_pm25_stacked.shape}  X_met: {X_met_stacked.shape}")


if __name__ == "__main__":
    main()
