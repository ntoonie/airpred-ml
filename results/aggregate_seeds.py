import glob, json
import numpy as np

files = sorted(glob.glob("results/stage3_metrics_seed*.json"))
runs = [json.load(open(f)) for f in files]
print("Seeds found:", [r["seed"] for r in runs])

print("\nTable 12 (mean +/- sample SD across seeds)")
for v in "CABD":  # AIRPRED, A, B, D
    row = []
    for m in ("rmse", "mae", "mape", "r2"):
        x = np.array([r["metrics"][v][m] for r in runs])
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        row.append(f"{m.upper()} {x.mean():.4f} +/- {sd:.4f}")
    print(f"Variant {v}: " + " | ".join(row))

print("\nTable 13 (paired t-test on per-window RMSE, per seed)")
for key in runs[0]["paired_t_tests"]:
    ts = [r["paired_t_tests"][key]["t_statistic"] for r in runs]
    ps = [r["paired_t_tests"][key]["p_value"] for r in runs]
    p = runs[0]["paired_t_tests"][key]
    print(f'{key} ({p["proposed"]} vs {p["baseline"]}): t per seed = '
          + ", ".join(f"{t:.4f}" for t in ts)
          + f" | max p = {max(ps):.4g}")