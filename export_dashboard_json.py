"""Export a dashboard-ready JSON from the saved Stage 3 predictions.

For each of the 10 NCR cities, takes the LAST test-set window
(chronologically most recent, since windows are ordered with stride=1)
as the "current" 24-hour forecast to display, with actual values and all
four variants' predictions -- already in real µg/m³ (stage3_predictions
was saved post scaler-fix).

Output shape:
{
  "generated_at": "...",
  "seed": 42,
  "units": "µg/m³",
  "cities": {
    "Manila": {
      "actual": [24 values],
      "variants": {
        "A": {"predicted": [24 values]},
        "B": {"predicted": [24 values]},
        "C": {"predicted": [24 values]},
        "D": {"predicted": [24 values]}
      }
    },
    ...
  }
}

USAGE:
    python export_dashboard_json.py --seed 42
    python export_dashboard_json.py --seed 42 --out ../airpred-web/public/data/forecast.json
"""
from __future__ import annotations

import argparse
import datetime
import json

import numpy as np

RESULTS_DIR = "results"
VARIANTS = ["A", "B", "C", "D"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path. Default: results/dashboard_forecast_seed{seed}.json "
             "(copy or point this at airpred-web/public/data/ yourself, or pass "
             "a path directly into that folder).",
    )
    args = parser.parse_args()

    npz_path = f"{RESULTS_DIR}/stage3_predictions_seed{args.seed}.npz"
    d = np.load(npz_path, allow_pickle=True)
    cities = d["cities"]
    actual = d["actual"]  # (N, 24), real µg/m³
    preds = {v: d[f"pred_{v}"] for v in VARIANTS}

    unique_cities = sorted(set(cities.tolist()))
    print(f"Exporting {len(unique_cities)} cities: {unique_cities}")

    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "units": "µg/m³",
        "cities": {},
    }

    for city in unique_cities:
        idx_for_city = np.where(cities == city)[0]
        last_idx = idx_for_city[-1]  # most recent window for this city in the test set

        output["cities"][city] = {
            "actual": [round(float(v), 2) for v in actual[last_idx]],
            "variants": {
                v: {"predicted": [round(float(x), 2) for x in preds[v][last_idx]]}
                for v in VARIANTS
            },
        }

    out_path = args.out or f"{RESULTS_DIR}/dashboard_forecast_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Dashboard JSON written to {out_path}")
    print(f"File size: {len(json.dumps(output))} bytes -- small enough to commit directly to the web repo.")


if __name__ == "__main__":
    main()
